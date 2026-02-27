import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, pulse_dim, hidden=256):
        super().__init__()

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, pulse_dim),
        )

        # Log std for Gaussian policy
        self.log_std = nn.Parameter(torch.zeros(pulse_dim))

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state):
        mean = self.actor(state)
        std = torch.exp(self.log_std)
        value = self.critic(state)
        return mean, std, value
