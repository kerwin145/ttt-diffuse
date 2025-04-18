import torch.nn as nn
import torch
import math

class BaseTransformer(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_heads, num_layers, use_encoder = True, use_decoder = True, max_seq_len=512, dropout=0.1):
        super(BaseTransformer, self).__init__()
        self.embedding_dim = embedding_dim

        # Project pose features to the embedding dimension
        self.input_projection = nn.Linear(input_dim, embedding_dim)

        # Positional encoding
        pos_enc = self._generate_positional_encoding(max_seq_len, embedding_dim)
        self.register_buffer('positional_encoding', pos_enc, persistent=False)

        self.encoder_layer = None
        self.decoder_layer = None

        # Transformer Encoder
        if use_encoder:
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dropout=dropout,
                batch_first=True
            )
        if use_decoder:
            self.decoder_layer = nn.TransformerDecoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dropout=dropout,
                batch_first=True
            )

    def _generate_positional_encoding(self, max_seq_len, embedding_dim):
        """
        Generate sinusoidal positional encodings as described in Vaswani et al.
        Returns:
            torch.Tensor: Positional encoding matrix of shape (1, max_seq_len, embedding_dim).
        """
        positional_encoding = torch.zeros(max_seq_len, embedding_dim)
        positions = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # Shape: (max_seq_len, 1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        # Apply sin to even indices
        positional_encoding[:, 0::2] = torch.sin(positions * div_term)
        # Apply cos to odd indices
        positional_encoding[:, 1::2] = torch.cos(positions * div_term)

        return positional_encoding.unsqueeze(0)  # Shape: (1, max_seq_len, embedding_dim)


    def add_positional_encoding(self, x, positional_encoding):
        seq_len = x.size(1)
        return x + positional_encoding[:, :seq_len, :]


class PoseTransformer(BaseTransformer):
    def __init__(self, pose_dim, embedding_dim, num_heads, num_layers, num_timesteps = 1000, use_encoder = True, use_decoder= False, max_seq_len=512, dropout=0.1):
        super().__init__(pose_dim, embedding_dim, num_heads, num_layers, use_encoder, use_decoder, max_seq_len, dropout)

        self.timestep_embedding = nn.Embedding(num_timesteps, embedding_dim)

        if use_encoder:
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        else:
            raise ValueError("PoseTransformer requires an encoder, but use_encoder=False was given.")

    def forward(self, poses, timesteps, pose_mask=None):
        # Project pose features to embedding space
        projected_poses = self.input_projection(poses) # Shape: (batch_size, seq_len, embedding_dim)
        pose_embeddings = self.add_positional_encoding(projected_poses, self.positional_encoding)
        t_emb = self.timestep_embedding(timesteps)  # (batch_size, embedding_dim)
        pose_embeddings += t_emb.unsqueeze(1)  # Broadcast to (batch_size, seq_len, embedding_di

        encoded_poses = self.transformer_encoder(pose_embeddings, src_key_padding_mask=pose_mask)

        return encoded_poses 
        # return encoded_poses

class CrossModalTransformer(BaseTransformer):
    def __init__(self, pose_dim, memory_dim, embedding_dim, num_heads, num_layers, use_encoder = False, use_decoder= True, max_seq_len=512, dropout=0.1):
        super().__init__(pose_dim, embedding_dim, num_heads, num_layers, use_encoder, use_decoder,  max_seq_len, dropout)

        if use_decoder:
            self.memory_projection = nn.Linear(memory_dim, embedding_dim)
            self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        else:
            raise ValueError("CrossModalTransformer requires a decoder, but use_decoder=False was given.")

    def forward(self, poses, memory, pose_mask=None, memory_mask=None):
        # pose_embeddings = self.add_positional_encoding(poses, self.positional_encoding)
        memory_embeddings = self.memory_projection(memory)
        # Cross-attend poses with memory
        output = self.transformer_decoder(poses, memory_embeddings, tgt_key_padding_mask=pose_mask, memory_key_padding_mask=memory_mask)

        return output