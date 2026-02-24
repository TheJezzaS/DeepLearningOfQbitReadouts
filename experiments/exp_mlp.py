import torch
from physics.sim_main import ReadoutPhysics
from training.losses import PhysicsLoss
from training.trainer import PhysicsTrainer
from training.datasets import LatentDataset
from models.mlp import PulseMLP
import time

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


print('------------------------------INITIALIZATIONS FINISHED-------------------------------')
# --- training ---
global_start = time.time()
for epoch in range(epochs):
    epoch_start = time.time()
    print(f'starting epoch {epoch}/{epochs} | previous epoch took {time.time() - epoch_start} ------------------------')
    batch_count = 0
    for z in loader:
        batch_count += 1
        batch_start = time.time()
        loss, metrics = trainer.step(z)
        print(f'batch {batch_count}/{len(loader)} | previous batch took {time.time() - batch_start} | (epoch, {epoch}/{epochs}) | global run time: {time.time() - global_start}')

    print(f"Epoch {epoch} | Loss: {loss:.6f} | Fidelity: {metrics[0]['fidelity']:.4f}")
