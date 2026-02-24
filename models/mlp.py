import torch
import torch.nn as nn
from .base import BaseModel

class PulseMLP(BaseModel):
    def __init__(self, latent_dim, pulse_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, pulse_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)
