import torch
from physics.sim_main import ReadoutPhysics
from training.losses import PhysicsLoss
from training.trainer import PhysicsTrainer
from training.datasets import LatentDataset
from models.mlp import PulseMLP

# --- config ---
latent_dim = 32
pulse_dim = 128
batch_size = 32
epochs = 100

# --- systems ---
physics = ReadoutPhysics()
model = PulseMLP(latent_dim, pulse_dim)
loss_fn = PhysicsLoss(w_fid=1.0, w_bw=0.05, w_sm=0.05)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

trainer = PhysicsTrainer(model, physics, loss_fn, opt)

dataset = LatentDataset(latent_dim, size=10000)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# --- training ---
for epoch in range(epochs):
    for z in loader:
        loss, metrics = trainer.step(z)

    print(f"Epoch {epoch} | Loss: {loss:.6f} | Fidelity: {metrics[0]['fidelity']:.4f}")
