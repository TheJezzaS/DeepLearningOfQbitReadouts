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

    def rollout(self, states):
        mean, std, values = self.model(states)
        dist = Normal(mean, std)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(-1)

        rewards = []
        infos = []

        for a in actions:
            out = self.physics.simulate(a)
            rewards.append(out["reward"])
            infos.append(out)

        rewards = torch.stack(rewards)

        return actions, log_probs, values.squeeze(-1), rewards, infos

    def ppo_update(self, states, actions, old_log_probs, returns, advantages):
        mean, std, values = self.model(states)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1).mean()

        ratio = torch.exp(log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(values.squeeze(-1), returns)

        loss = actor_loss + self.vf_coef * value_loss - self.ent_coef * entropy

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()
