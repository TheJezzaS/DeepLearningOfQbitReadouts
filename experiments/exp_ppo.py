import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
from physics.sim_main import ReadoutPhysics
from models.ppo import PPOActorCritic
from training.ppo_trainer import PPOTrainer

device = torch.device("cpu")

# --- config ---
state_dim = 16        # dummy state
pulse_dim = 121       # N_ACTION
batch_size = 32
epochs = 5000
gamma = 0.99

# --- systems ---
physics = ReadoutPhysics()
model = PPOActorCritic(state_dim, pulse_dim).to(device)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

trainer = PPOTrainer(model, physics, opt)

print("PPO initialized")

# Dummy state (stateless env like paper)
def sample_states(B):
    return torch.zeros(B, state_dim)

for epoch in range(epochs):
    states = sample_states(batch_size)

    actions, log_probs, values, rewards, infos = trainer.rollout(states)

    # returns = rewards (single-step env like paper)
    returns = rewards.detach()
    advantages = returns - values.detach()

    loss = trainer.ppo_update(
        states, actions, log_probs.detach(),
        returns, advantages
    )

    if epoch % 50 == 0:
        avg_fid = torch.mean(torch.tensor([i["fidelity"] for i in infos])).item()
        print(f"[{epoch}] loss={loss:.4f}  avg_fidelity={avg_fid:.4f}")
