import torch
from models.mlp import PulseMLP

# 1. Define the exact same dimensions you used during training
latent_dim = 32
pulse_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Initialize a fresh, untrained model
loaded_model = PulseMLP(latent_dim, pulse_dim)

# 3. Load the saved weights into the model
loaded_model.load_state_dict(torch.load("trained_pulse_mlp.pth", map_location=device))

# 4. Move it to the GPU/CPU and set it to evaluation mode
loaded_model.to(device)
loaded_model.eval()

print("Model loaded and ready for inference!")

# Now you can generate a pulse instantly:
with torch.no_grad():
    z = torch.randn(1, latent_dim).to(device)
    best_pulse = loaded_model(z)