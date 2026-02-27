import sys
import os
# Add the parent directory to Python's import path to fix ModuleNotFoundError
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler


from physics.sim_core import evaluate_pulse, t_action, t_sim

from physics.sim_main import ReadoutPhysics
from training.losses import PhysicsLoss
from training.trainer import PhysicsTrainer
from training.datasets import LatentDataset
from models.mlp import PulseMLP

# --- Device Configuration ---
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")

# --- config ---
latent_dim = 32
pulse_dim = 128
batch_size = 2
epochs = 20
size_per_epoch = 10

# --- systems ---
physics = ReadoutPhysics()
model = PulseMLP(latent_dim, pulse_dim)

# Note: Ensure your PhysicsLoss in losses.py matches the updated code we discussed
loss_fn = PhysicsLoss() 

# --- Optimizer with Weight Decay & Higher Initial LR ---
# Starting at 3e-4 to match the initial momentum used in the research paper
opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)

# --- Adaptive Learning Rate Scheduler ---
# Drops the learning rate by half (factor=0.5) if the physics loss stalls for 3 epochs (patience=3)
scheduler = lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)

# Pass the device to the trainer
trainer = PhysicsTrainer(model, physics, loss_fn, opt, device=device)
dataset = LatentDataset(latent_dim, size=size_per_epoch)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

print('------------------------------INITIALIZATIONS FINISHED-------------------------------')

# --- Tracking arrays for the Learning Curve ---
history_loss = []
history_fidelity = []


save_dir = "trained_models"
os.makedirs(save_dir, exist_ok=True) # Creates the folder safely
best_fidelity = 0.0
best_model_path = os.path.join(save_dir, "best_pulse_mlp.pth")

# --- training ---
global_start = time.time()
for epoch in range(epochs):
    epoch_start = time.time()
    batch_count = 0
    
    # Temporarily store the batch loss and fidelity to average them per epoch
    epoch_losses = []
    epoch_fidelities = []
    
    for z in loader:
        batch_count += 1
        batch_start = time.time()
        
        # Trainer step handles moving z to the correct device internally
        loss, metrics = trainer.step(z)
        
        epoch_losses.append(loss)
        epoch_fidelities.append(metrics[0]['fidelity'].item()) 
        
        # Reduced the print frequency so the console is easier to read
        if batch_count % 100 == 0:
            print(f'batch {batch_count}/{len(loader)} | previous batch took {time.time() - batch_start:.4f}s')

    # Calculate epoch averages
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    avg_fid = sum(epoch_fidelities) / len(epoch_fidelities)

    history_loss.append(avg_loss)
    history_fidelity.append(avg_fid)
    
    # --- Step the scheduler based on the average epoch loss ---
    scheduler.step(avg_loss)
    current_lr = opt.param_groups[0]['lr']

    print(f"Epoch {epoch}/{epochs-1} | Avg Loss: {avg_loss:.6f} | Avg Fidelity: {avg_fid:.4f} | LR: {current_lr:.6f} | Time: {time.time() - epoch_start:.2f}s")

    if avg_fid > best_fidelity:
        best_fidelity = avg_fid
        torch.save(model.state_dict(), best_model_path)
        print(f"New best fidelity! Model saved to {best_model_path}")



print("Training Complete. Generating Graphs...")

# ==========================================================
# --- Plotting the Article's Graphs ---
# ==========================================================
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval() 
with torch.no_grad():
    # Generate one final test pulse
    test_z = torch.randn(1, latent_dim).to(device)
    final_pulse = model(test_z).squeeze()
    
    # Run it through the physics simulator
   
    
    reward, info, pulse_out, ng, ne, F, res = evaluate_pulse(final_pulse)

# Move tensors back to CPU for matplotlib compatibility
pulse_out = pulse_out.cpu().numpy()
ng = ng.cpu().numpy()
ne = ne.cpu().numpy()
F = F.cpu().numpy()
res = res.cpu().numpy()

# Extract Real (I) and Imaginary (Q) components for Ground and Excited states
# res shape is (Time, 4): [Ground_Real, Ground_Imag, Excited_Real, Excited_Imag]
I_g = res[:, 0]
Q_g = res[:, 1]
I_e = res[:, 2]
Q_e = res[:, 3]

# Create a 3-panel figure
plt.figure(figsize=(14, 10))

# --- Plot 1: Learning Curve ---
plt.subplot(3, 1, 1)
plt.plot(range(len(history_fidelity)), history_fidelity, label='Fidelity', color='green', linewidth=2)
plt.ylabel('Fidelity')
plt.xlabel('Epochs')
plt.title('Training Convergence')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# --- Plot 2: The Generated Waveform ---
plt.subplot(3, 1, 2)
plt.plot(t_action.cpu().numpy(), pulse_out, label='Optimized Waveform', color='blue', linewidth=2)
plt.ylabel('Amplitude (a.u.)')
plt.xlabel('Time ($\mu$s)')
plt.title('Learned Readout Pulse')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# --- Plot 3: Photon Population and Fidelity over Time ---
plt.subplot(3, 1, 3)
plt.plot(t_sim.cpu().numpy(), ng, label='Ground Photons ($n_g$)', color='cyan')
plt.plot(t_sim.cpu().numpy(), ne, label='Excited Photons ($n_e$)', color='magenta')
plt.plot(t_sim.cpu().numpy(), F, label='Instantaneous Fidelity', color='black', linestyle='--')
plt.axhline(info['fidelity'].item(), color='red', linestyle=':', label=f"Max Fidelity: {info['fidelity'].item():.4f}")
plt.ylabel('Magnitude')
plt.xlabel('Time ($\mu$s)')
plt.title('Resonator Dynamics')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()


# --- Plot 4: The Phase-Space "Butterfly" ---
# If using a 2x2 grid, this would be plt.subplot(2, 2, 4)
plt.figure(figsize=(6, 6))
plt.plot(I_g, Q_g, label='Ground State $|g\\rangle$', color='cyan')
plt.plot(I_e, Q_e, label='Excited State $|e\\rangle$', color='magenta')

# Mark the starting point (vacuum) and ending point
plt.scatter([0], [0], color='black', marker='x', label='Start (Vacuum)')
plt.scatter(I_g[-1], Q_g[-1], color='blue', marker='o', label='End $|g\\rangle$')
plt.scatter(I_e[-1], Q_e[-1], color='red', marker='o', label='End $|e\\rangle$')

plt.xlabel('In-Phase (I)')
plt.ylabel('Quadrature (Q)')
plt.title('Cavity Phase-Space Trajectory')
plt.axhline(0, color='grey', linestyle='--', alpha=0.5)
plt.axvline(0, color='grey', linestyle='--', alpha=0.5)
plt.legend()
plt.grid(True)


save_path = "trained_pulse_mlp.pth"
torch.save(model.state_dict(), save_path)
print(f"Model successfully saved to {save_path}")


plt.show()