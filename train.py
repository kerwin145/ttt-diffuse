import os
import random
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from os.path import join as pjoin
from transformers import CLIPModel, CLIPTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch.multiprocessing as mp

from PoseTransformer import PoseTransformer, CrossModalTransformer
from NoiseScheduler import NoiseScheduler
mp.set_start_method("spawn", force=True)

BATCH_SIZE = 8
EPOCHS = 16
SAVE_PATH = "model_output"

src_dir = "..\\HumanML3D"
train_list = open(pjoin(src_dir, "train.txt"), "r", encoding="utf-8")
val_list = open(pjoin(src_dir, "val.txt"), "r", encoding="utf-8")
test_list = open(pjoin(src_dir, "test.txt"), "r", encoding="utf-8")

clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

class PoseTextDataset(Dataset):
    """
        (data_dir): contains all the new_joint_vecs
        (train_list): the train.txt, as an opened file
        (max_len): max length of text descriptions
    """

    def __init__(self, src_dir, setting, tokenizer, max_len=256, use_percentage = 1.0):
        self.src_dir = src_dir
        self.max_len = max_len
        self.tokenizer = tokenizer

        if setting not in ["train", "train_temp", "val", "test", "all"]:
            print("Invalid setting. Must be train, val, test, or all.")
            raise

        with open(pjoin(src_dir, f"{setting}.txt"), "r") as f:
            self.file_list = [line.strip() for line in f.readlines()]

        if 0 < use_percentage < 1.0:
            random.shuffle(self.file_list)
            num_samples = int(len(self.file_list) * use_percentage)
            self.file_list = self.file_list[:num_samples]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file = self.file_list[index]

        # Load post data
        pose_path = pjoin(self.src_dir, "new_joint_vecs", f"{file}.npy")
        pose_data = np.load(pose_path)  # Shape: (frames, joints, features)
        pose_tensor = torch.tensor(pose_data, dtype=torch.float32)

        # Load text description
        text_path = pjoin(self.src_dir, "texts", f"{file}.txt")

        # The descriptions have a extra information such as time stamp and part of speech. I will get rid of that for now to keep things simple.
        with open(text_path, "r", encoding="utf-8") as f:
            text_descriptions = [
                re.split('#', line.strip())[0] for line in f.readlines() if line.strip()
            ]

        encoded_texts = self.tokenizer(
            text_descriptions,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        # Since tokenizer returns tensors in a dictionary, we can access them directly
        input_ids = encoded_texts["input_ids"]
        attention_mask = encoded_texts["attention_mask"]

        # Return a list of dictionaries, one per description
        return [{
            "pose": pose_tensor,
            "text": input_ids[i],  # Squeeze unnecessary dimensions
            "attention_mask": attention_mask[i]  # Squeeze unnecessary dimensions
        } for i in range(len(text_descriptions))]

def collate_fn(batch):
    """
    Pads all pose sequences to the maximum length in the batch.
    """
    flattened_batch = [item for sublist in batch for item in sublist]

    # Extract poses and text descriptions from the batch
    poses = [item["pose"] for item in flattened_batch]  # List of tensors (varied lengths)
    texts = [item["text"] for item in flattened_batch]  # List of tokenized text tensors
    attention_masks = [item.get("attention_mask", None) for item in flattened_batch]

    # Pad pose sequences (pad with zeros to max length in the batch)
    padded_poses = pad_sequence(poses, batch_first=True, padding_value=0.0)  # Shape: (batch_size, max_len, ...)
    pose_mask = (padded_poses.sum(dim=-1) != 0).float()  # Sum across features, Shape: (batch_size, max_pose_len)

    # Stack text tensors
    padded_texts = torch.stack(texts, dim=0)
    padded_attention_masks = torch.stack(attention_masks, dim=0)

    return {
        "pose": padded_poses,
        "pose_mask": pose_mask,
        "text": padded_texts,
        "attention_mask": padded_attention_masks,
    }

class Trainer:
    def __init__(self, options):
        self.device = options['device']
        self.clip_model = options['clip_model'].to(self.device)
        self.clip_tokenizer = options['clip_tokenizer']
        self.pose_transformer = options['pose_transformer'].to(self.device)
        self.text_cross_transformer = options['text_cross_transformer'].to(self.device)
        self.pose_features_dim = options['pose_features_dim']
        self.noise_predictor = torch.nn.Linear(options['embedding_dim'], self.pose_features_dim).to(self.device)

        self.noise_scheduler = NoiseScheduler(timesteps=1000)
        # clip is left out of the optimizer, as we won't be tuning CLIP model
        self.optimizer = torch.optim.AdamW(
            list(self.pose_transformer.parameters()) +
            list(self.text_cross_transformer.parameters()) +
            list(self.noise_predictor.parameters()),
            lr=1e-4,
            weight_decay=1e-2
        )
    def train(self, dataloader, optimizer = None):
        self.pose_transformer.train()
        self.text_cross_transformer.train()
        self.noise_predictor.train()
        if optimizer is None:  # Default to using the initialized optimizer
            optimizer = self.optimizer

        total_loss = 0  # Track total loss for the epoch
        num_batches = len(dataloader)

        for batch in tqdm(dataloader, leave=True):
            poses = batch["pose"].to(self.device)  # (batch_size, max_pose_len, features)
            pose_mask = batch["pose_mask"].to(self.device)
            texts = batch["text"].to(self.device)  # (batch_size, max_text_len)
            text_mask = batch["attention_mask"].to(self.device)  # Attention mask for text

            # Denoiser setup
            batch_size = poses.shape[0]
            noise = torch.randn_like(poses)
            timesteps = self.noise_scheduler.sample_timesteps(batch_size, device=self.device)
            noisy_poses = self.noise_scheduler.add_noise(poses, noise, timesteps)

            # Get text embeddings from CLIP
            text_inputs = {
                "input_ids": texts,
                "attention_mask": text_mask
            }
            text_embeddings = self.clip_model.get_text_features(**text_inputs)  # Shape: (batch_size, embedding_dim)

            pose_embeddings = self.pose_transformer(
                noisy_poses,
                pose_mask=pose_mask,
                timesteps=timesteps
            )

            # Cross-attention with text embeddings
            text_conditioned_embeddings = self.text_cross_transformer(
                pose_embeddings,
                text_embeddings,
                pose_mask=pose_mask,
                memory_mask=None
            )

            # Predict noise (output shape: batch_size, seq_len, pose_features)
            predicted_noise = self.noise_predictor(text_conditioned_embeddings).permute(1, 0, 2)
            # Compute MSE loss (equivalent to KL divergence in this context)
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        return total_loss / num_batches

    def eval(self, dataloader):
        self.pose_transformer.eval()
        self.text_cross_transformer.eval()
        self.noise_predictor.eval()
        with torch.no_grad():
            total_loss = 0
            for batch in tqdm(dataloader, leave=True):
                poses = batch["pose"].to(self.device)  # (batch_size, max_pose_len, features)
                pose_mask = batch["pose_mask"].to(self.device)
                texts = batch["text"].to(self.device)  # (batch_size, max_text_len)
                text_mask = batch["attention_mask"].to(self.device)  # Attention mask for text

                # Denoiser setup
                batch_size = poses.shape[0]
                noise = torch.randn_like(poses)
                timesteps = self.noise_scheduler.sample_timesteps(batch_size, device=self.device)
                noisy_poses = self.noise_scheduler.add_noise(poses, noise, timesteps)

                # Get text embeddings from CLIP
                text_inputs = {
                    "input_ids": texts,
                    "attention_mask": text_mask
                }
                text_embeddings = self.clip_model.get_text_features(**text_inputs)  # Shape: (batch_size, embedding_dim)

                pose_embeddings = self.pose_transformer(
                    noisy_poses,
                    pose_mask=pose_mask,
                    timesteps=timesteps
                )

                # Cross-attention with text embeddings
                text_conditioned_embeddings = self.text_cross_transformer(
                    pose_embeddings,
                    text_embeddings,
                    pose_mask=pose_mask,
                    memory_mask=None
                )

                # Predict noise (output shape: batch_size, seq_len, pose_features)
                predicted_noise = self.noise_predictor(text_conditioned_embeddings).permute(1, 0, 2)
                # Compute MSE loss (equivalent to KL divergence in this context)
                loss = torch.nn.functional.mse_loss(predicted_noise, noise)
                total_loss += loss.item()
        return total_loss/len(dataloader)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    SEED = 42
    random.seed(SEED)  # Python random module
    np.random.seed(SEED)  # NumPy random
    torch.manual_seed(SEED)  # PyTorch random

    EMBEDDING_DIM = 512

    trainDataset = PoseTextDataset(src_dir=src_dir,setting="train",tokenizer=clip_tokenizer,max_len=77,use_percentage=0.2)
    trainDataLoader = DataLoader(trainDataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,collate_fn=collate_fn)
    # persistent_workers=True,
    # multiprocessing_context="spawn"
    evalDataset = PoseTextDataset(src_dir=src_dir,setting="val",tokenizer=clip_tokenizer,max_len=77,use_percentage=1)
    evalDataLoader = DataLoader(evalDataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,collate_fn=collate_fn)

    pose_transformer = PoseTransformer(
        pose_dim=263,
        embedding_dim=EMBEDDING_DIM,  # Match CLIP's embedding dimension
        num_heads=8,  # Number of attention heads
        num_layers=6,  # Number of transformer layers
        dropout=0.1  # Dropout probability
    )

    text_cross_transformer = CrossModalTransformer(
        pose_dim = EMBEDDING_DIM,
        memory_dim= EMBEDDING_DIM,
        embedding_dim=EMBEDDING_DIM,
        num_heads=8,
        num_layers=6,
        use_decoder=True
    )

    trainer = Trainer({
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "clip_model": clip_model,
        "clip_tokenizer": clip_tokenizer,
        "pose_transformer": pose_transformer,
        "text_cross_transformer": text_cross_transformer,
        "pose_features_dim": 263,
        "embedding_dim": EMBEDDING_DIM
    })
    # Print for each model
    print(f"Pose Transformer Parameters: {count_parameters(trainer.pose_transformer)}")
    print(f"Text-Cross Transformer Parameters: {count_parameters(trainer.text_cross_transformer)}")
    print(f"Noise Predictor Parameters: {count_parameters(trainer.noise_predictor)}")

    # Total trainable parameters
    total_params = sum([
        count_parameters(trainer.pose_transformer),
        count_parameters(trainer.text_cross_transformer),
        count_parameters(trainer.noise_predictor),
    ])
    print(f"Total Trainable Parameters: {total_params}")
    train_losses = []
    eval_losses = []
    for epoch in range(16):
        trainLoss = trainer.train(trainDataLoader)
        train_losses.append(trainLoss)
        evalLoss = trainer.eval(evalDataLoader)
        eval_losses.append(evalLoss)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), train_losses, marker='o', linestyle='-', color='b', label="Train Loss")
    plt.plot(range(1, EPOCHS + 1), eval_losses, marker='s', linestyle='--', color='r', label="Eval Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training & Evaluation Loss Curve")
    plt.legend()  # Show legend to differentiate train and eval losses
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()

