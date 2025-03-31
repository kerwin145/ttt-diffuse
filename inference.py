import torch
import os
from PoseTransformer import PoseTransformer, CrossModalTransformer
from NoiseScheduler import NoiseScheduler
from transformers import CLIPModel, CLIPTokenizer
from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion

MODEL_PATH = "model_saves/TT_0.75_percentdata_lr0.0001_wd0.01"
EMBEDDING_DIM = 512
POSE_FEATURES_DIM = 263
CLIP_MAX_LENGTH = 77

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load trained models
print("Loading models...")
pose_transformer = PoseTransformer(
    pose_dim=POSE_FEATURES_DIM,
    embedding_dim=EMBEDDING_DIM,
    num_heads=8,
    num_layers=6,
    dropout=0.1,
    use_decoder=False
).to(device)

text_cross_transformer = CrossModalTransformer(
    pose_dim=EMBEDDING_DIM,
    memory_dim=EMBEDDING_DIM,
    embedding_dim=EMBEDDING_DIM,
    num_heads=8,
    num_layers=6,
    use_decoder=True
).to(device)

noise_predictor = torch.nn.Linear(EMBEDDING_DIM, POSE_FEATURES_DIM).to(device)

# CLIP Model and Tokenizer
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model checkpoint not found at: {MODEL_PATH}")
# weights only to true to stop receiving the warning
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)

pose_transformer.load_state_dict(checkpoint["pose_transformer"])
text_cross_transformer.load_state_dict(checkpoint["text_cross_transformer"])
noise_predictor.load_state_dict(checkpoint["noise_predictor"])
print("Model weights loaded successfully.")

pose_transformer.eval()
text_cross_transformer.eval()
noise_predictor.eval()
clip_model.eval()

noise_scheduler = NoiseScheduler(timesteps=1000)

@torch.no_grad()
def infer(text_input, seq_length=60, batch_size=1):

    # Tokenize text input and get text embeddings
    encoded_texts = clip_tokenizer(
        text_input,
        truncation=True,
        padding="max_length",
        max_length=CLIP_MAX_LENGTH,
        return_tensors="pt"
    )
    text_inputs = {
        "input_ids": encoded_texts["input_ids"].to(device),
        "attention_mask": encoded_texts["attention_mask"].to(device)
    }

    text_embeddings = clip_model.get_text_features(**text_inputs)
    
    if text_embeddings.ndim == 2:
         text_embeddings = text_embeddings.unsqueeze(1)

    # Initialize random noise as the starting pose
    pose_shape = (batch_size, seq_length, POSE_FEATURES_DIM)
    current_pose = torch.randn(pose_shape, device=device)
    pose_mask = torch.ones((batch_size, seq_length), device=device, dtype=torch.float32) # <-- Create mask

    print("Starting denoising loop...")

    # Perform reverse diffusion (denoising)
    for t in reversed(range(noise_scheduler.timesteps)):
        timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # Get pose embeddings from transformer
        pose_embeddings = pose_transformer(
            current_pose,
            pose_mask=pose_mask,
            timesteps=timesteps
        )

        # Cross-attention with text embeddings
        text_conditioned_embeddings = text_cross_transformer(
            pose_embeddings,
            text_embeddings,
            pose_mask, 
            memory_mask=None
        )

        # Predict noise
        predicted_noise = noise_predictor(text_conditioned_embeddings)

        beta_t = noise_scheduler.betas[t].to(device)
        alpha_t = noise_scheduler.alphas[t].to(device)
        sqrt_one_minus_alpha_cumprod_t = noise_scheduler.sqrt_one_minus_alphas_cumprod[t].to(device)
        sqrt_alpha_t = torch.sqrt(alpha_t) 

        # Calculate the term multiplying the predicted noise
        noise_coeff = beta_t / sqrt_one_minus_alpha_cumprod_t

        # Calculate the mean of x_{t-1}
        mean_x_t_minus_1 = (1.0 / sqrt_alpha_t) * (current_pose - noise_coeff * predicted_noise)

        # Add noise (variance term) - except for the last step (t=0)
        if t > 0:
            variance = beta_t # Use beta_t for variance (sigma_t = sqrt(beta_t))
            std_dev = torch.sqrt(variance)
            noise = torch.randn_like(current_pose)
            current_pose = mean_x_t_minus_1 + std_dev * noise # Update current_pose to x_{t-1}
        else:
            current_pose = mean_x_t_minus_1

    return current_pose.cpu()

# Example usage
text_prompt = "A person walking slowly."
generated_poses = infer(text_prompt, seq_length=120, batch_size=1)
pose_single = generated_poses[0].numpy()
print(pose_single.shape)
joints_num = 22
kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
# print(example_data_ml3d[0:5])

joints = recover_from_ric(torch.from_numpy(pose_single), joints_num).numpy()
trajectory = joints[:, 0, :]
print("Generated Pose Shape:", generated_poses.shape, joints.shape)
plot_3d_motion("output/test_ani.mp4", kinematic_chain, joints, title="Testing!", fps=20)
print(joints)