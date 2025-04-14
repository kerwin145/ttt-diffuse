# compute_stats.py
import numpy as np
import torch
from os.path import join as pjoin
from tqdm import tqdm
import os

# --- Configuration ---
src_dir = "..\\HumanML3D"  # Adjust if needed
pose_features_dim = 263   # Dimension of features in the .npy files
# --- End Configuration ---

print(f"Source directory: {src_dir}")
print(f"Expected pose feature dimension: {pose_features_dim}")

train_list_path = pjoin(src_dir, "train.txt")
if not os.path.exists(train_list_path):
    raise FileNotFoundError(f"train.txt not found at {train_list_path}")

with open(train_list_path, "r") as f:
    file_list = [line.strip() for line in f.readlines()]

all_pose_data = []
print(f"Found {len(file_list)} files in train.txt. Loading data for stats calculation...")

skipped_files = 0
loaded_files = 0
for file in tqdm(file_list, desc="Loading Files"):
    pose_path = pjoin(src_dir, "new_joint_vecs", f"{file}.npy")
    if not os.path.exists(pose_path):
        print(f"Warning: Skipping {file}, file not found at {pose_path}")
        skipped_files += 1
        continue
    try:
        pose_data = np.load(pose_path) # Shape: (frames, pose_features_dim)
        if pose_data.ndim != 2 or pose_data.shape[1] != pose_features_dim:
             print(f"Warning: Skipping {file}, unexpected shape {pose_data.shape}")
             skipped_files += 1
             continue
        if np.isnan(pose_data).any():
             print(f"Warning: Skipping {file}, contains NaN values")
             skipped_files += 1
             continue
        all_pose_data.append(pose_data)
        loaded_files += 1
    except Exception as e:
        print(f"Warning: Could not load or process {file}: {e}")
        skipped_files += 1

print(f"Loaded data from {loaded_files} files. Skipped {skipped_files} files.")

if not all_pose_data:
    raise ValueError("No valid pose data loaded. Check paths, data integrity, and expected dimension.")

# Concatenate all data along the time dimension (axis 0)
print("Concatenating data...")
all_pose_data_np = np.concatenate(all_pose_data, axis=0) # Shape: (total_frames, pose_features_dim)
del all_pose_data # Free memory

print(f"Calculating mean and std on final data shape: {all_pose_data_np.shape}")
# Calculate mean and std across all frames for each feature dimension
mean = np.mean(all_pose_data_np, axis=0)
std = np.std(all_pose_data_np, axis=0)

# Add a small epsilon to std to prevent division by zero
std[std == 0] = 1e-6

# Save stats
stats_path = pjoin("pose_stats.npz")
np.savez(stats_path, mean=mean, std=std)
print(f"Saved mean and std (shape: {mean.shape}, {std.shape}) to {stats_path}")

# Verify saved stats
loaded_stats = np.load(stats_path)
print(f"Verification: Loaded Mean shape: {loaded_stats['mean'].shape}, Std shape: {loaded_stats['std'].shape}")

"""
stats_file = "pose_stats.npz"
        if os.path.exists(stats_file):
            stats = np.load(stats_file)
            if 'mean' in stats and 'std' in stats:
                 self.mean = torch.tensor(stats['mean'], dtype=torch.float32).unsqueeze(0) # Shape (1, pose_features_dim)
                 self.std = torch.tensor(stats['std'], dtype=torch.float32).unsqueeze(0)   # Shape (1, pose_features_dim)
                 if self.mean.shape[1] != self.pose_features_dim or self.std.shape[1] != self.pose_features_dim:
                      print(f"Warning: Loaded stats shape mismatch! Mean: {self.mean.shape}, Std: {self.std.shape}. Expected dim {self.pose_features_dim}")
                      self.mean = None
                      self.std = None
                 else:
                     print(f"Loaded normalization stats for {setting} dataset.")
            else:
                 print("Warning: 'mean' or 'std' not found in stats file.")
        else:
            print(f"Warning: Normalization stats file not found at {stats_file}. Data will not be normalized.")
"""