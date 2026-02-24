import torch
from physics.batching import simulate_batch

class PhysicsTrainer:
    def __init__(self, model, physics, loss_fn, optimizer, device="cpu"):
        self.model = model.to(device)
        self.physics = physics
        self.loss_fn = loss_fn
        self.opt = optimizer
        self.device = device

    def step(self, z_batch):
        z_batch = z_batch.to(self.device)

        pulses = self.model(z_batch)  # NN → pulses

        metrics_batch = simulate_batch(self.physics, pulses)

        losses = []
        for m in metrics_batch:
            losses.append(self.loss_fn.compute(m))

        # IMPORTANT: use stack, not torch.tensor
        loss = torch.stack(losses).mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item(), metrics_batch

