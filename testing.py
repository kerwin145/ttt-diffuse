from os.path import join as pjoin
import numpy as np

src_dir = "..\\HumanML3D"
train_list = open(pjoin(src_dir, "train_temp.txt"), "r", encoding="utf-8")
file_list = [line.strip() for line in train_list]

for file in file_list:
    pose_path = pjoin(src_dir, "new_joint_vecs", f"{file}.npy")
    pose_data = np.load(pose_path)  # Shape: (frames, joints, features)
    print(pose_data.shape)
