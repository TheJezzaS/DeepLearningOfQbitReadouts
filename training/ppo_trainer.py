import torch
from torch.distributions import Normal
import torch.nn.functional as F

class PPOTrainer:
    def __init__(self, model, physics, optimizer,
                 clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):

        self.model = model
        self.physics = physics
        self.opt = optimizer
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

    # training/ppo_trainer.py

    def rollout(self, states):
        """
        Single-step rollout  (stateless env like the paper)
        """
        with torch.no_grad():
            # Policy forward
            dist, values = self.model(states)

            # Sample actions (pulses)
            actions = dist.sample()  # (B, pulse_dim)
            log_probs = dist.log_prob(actions).sum(-1)  # (B,)

            rewards = []
            infos = []

            for i in range(actions.shape[0]):
                out = self.physics.simulate(actions[i])  # ✅ correct API
                rewards.append(out["reward"])
                infos.append(out)

            rewards = torch.stack(rewards).to(states.device)

        return actions, log_probs, values, rewards, infos

    def ppo_update(self, states, actions, old_log_probs, returns, advantages):
        """
        PPO policy + value update
        """
        # Forward pass
        dist, values = self.model(states)

        # New log probs
        new_log_probs = dist.log_prob(actions).sum(-1)

        # Ratio for PPO clipping
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(values.squeeze(-1), returns)

        # ✅ Entropy bonus (for exploration)
        entropy = dist.entropy().sum(-1).mean()

        # Total PPO loss
        loss = actor_loss + self.vf_coef * value_loss - self.ent_coef * entropy

        # Optimize
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()
