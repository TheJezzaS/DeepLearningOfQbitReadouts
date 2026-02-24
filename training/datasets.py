import torch
from torch.utils.data import Dataset
import numpy as np

class LatentDataset(Dataset):
    def __init__(self, latent_dim: int, size: int):
        self.latent_dim = latent_dim
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        z = torch.randn(self.latent_dim, dtype=torch.float32)
        return z
