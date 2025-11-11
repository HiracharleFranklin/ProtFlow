import os
import torch
import numpy as np
from torch.utils.data import Dataset

class ReflowDataset(Dataset):
    def __init__(self, mode, base_path, num_files):
        self.mode = mode
        self.num_files = num_files
        if mode == "train":
            self.z0_data  = []
            self.z1_data  = []
            for i in range(num_files):
                print(f"Loading train data block{i}")
                z0 = np.load(os.path.join(base_path, f"z0_train{i}.npy"))
                z1 = np.load(os.path.join(base_path, f"z1_train{i}.npy"))
                self.z0_data.append(z0)
                self.z1_data.append(z1)
            self.z0_data = np.concatenate(self.z0_data, axis=0)
            self.z1_data = np.concatenate(self.z1_data, axis=0)
        else:
            self.z0_data = np.load(os.path.join(base_path, f"z0_valid.npy"))
            self.z1_data = np.load(os.path.join(base_path, f"z1_valid.npy"))
        self.num_samples = self.z0_data.shape[0]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        z0_sample = self.z0_data[idx]
        z1_sample = self.z1_data[idx]
        z0_tensor = torch.tensor(z0_sample, dtype=torch.float32)
        z1_tensor = torch.tensor(z1_sample, dtype=torch.float32)
        return z0_tensor, z1_tensor