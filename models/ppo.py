import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.distributions as D

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
        """
        Returns:
            dist  : torch.distributions.Distribution
            value : state-value estimate
        """
        mean = self.actor(state)                 # (B, action_dim)
        std = torch.exp(self.log_std)             # (action_dim,)
        std = std.expand_as(mean)                 # match batch shape

        dist = D.Normal(mean, std)                # policy distribution
        value = self.critic(state).squeeze(-1)    # (B,)

        return dist, value

    def act(self, state):
        """
        Used during rollout
        """
        dist, value = self(state)

        action = dist.sample()                    # (B, action_dim)
        log_prob = dist.log_prob(action).sum(-1)  # (B,)

        return action, log_prob, value

    def evaluate(self, state, action):
        dist, value = self(state)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value