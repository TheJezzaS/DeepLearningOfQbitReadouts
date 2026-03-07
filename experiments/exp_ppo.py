# import sys, os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
# import torch
# import time
# from physics.sim_main import ReadoutPhysics
# from models.ppo import PPOActorCritic
# from training.ppo_trainer import PPOTrainer
#
# device = torch.device("cpu")
#
# # --- config ---
# state_dim = 16        # dummy state
# pulse_dim = 121       # N_ACTION
# batch_size = 32
# epochs = 100
# gamma = 0.99
#
# # --- systems ---
# physics = ReadoutPhysics()
# model = PPOActorCritic(state_dim, pulse_dim).to(device)
# opt = torch.optim.Adam(model.parameters(), lr=3e-4)
#
# trainer = PPOTrainer(model, physics, opt)
#
# print("PPO initialized")
#
# # Dummy state (stateless env like paper)
# def sample_states(B):
#     return torch.zeros(B, state_dim)
#
# for epoch in range(epochs):
#     states = sample_states(batch_size)
#
#     actions, log_probs, values, rewards, infos = trainer.rollout(states)
#
#     # returns = rewards (single-step env like paper)
#     returns = rewards.detach()
#     advantages = returns - values.detach()
#
#     loss = trainer.ppo_update(
#         states, actions, log_probs.detach(),
#         returns, advantages
#     )
#
#     if epoch % 10 == 0:
#         avg_fid = torch.mean(torch.tensor([i["fidelity"] for i in infos])).item()
#         print(f"[{epoch}] loss={loss:.4f}  avg_fidelity={avg_fid:.4f}")



import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.sim_main import ReadoutPhysics
from models.ppo import PPOActorCritic
from training.ppo_trainer import PPOTrainer
from physics.sim_core import evaluate_pulse, t_action, t_sim

import matplotlib.pyplot as plt
import torch

# ==========================================================
# Device
# ==========================================================
device = torch.device("cpu")   # change to cuda later if needed


# ==========================================================
# Config
# ==========================================================
CONFIG = {
    "state_dim": 16,       # dummy state (stateless env, like paper)
    "pulse_dim": 121,      # action dimension (pulse samples)
    "batch_size": 32,
    "epochs": 1000,
    "gamma": 0.99,
    "lr": 3e-4,
    "log_interval": 10
}


# ==========================================================
# Environment abstraction (stateless physics env)
# ==========================================================
class ReadoutEnv:
    """
    Stateless environment wrapper around physics simulator.
    This allows future extension to multi-step RL without refactor.
    """

    def __init__(self, physics: ReadoutPhysics, state_dim: int, device):
        self.physics = physics
        self.state_dim = state_dim
        self.device = device

    def reset(self, batch_size: int):
        """
        Returns initial dummy state (stateless environment).
        """
        return torch.zeros(batch_size, self.state_dim, device=self.device)

    def step(self, actions: torch.Tensor):
        """
        Executes physics simulation using action = pulse.
        """
        infos = []
        rewards = []

        for pulse in actions:
            out = self.physics.simulate(pulse)
            infos.append(out)
            rewards.append(out["reward"])

        rewards = torch.stack(rewards).to(self.device)
        next_state = torch.zeros_like(self.reset(len(actions)))  # stateless
        done = torch.ones(len(actions), dtype=torch.bool, device=self.device)

        return next_state, rewards, done, infos


# ==========================================================
# Initialization
# ==========================================================
def init_system():
    physics = ReadoutPhysics()
    model = PPOActorCritic(CONFIG["state_dim"], CONFIG["pulse_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    trainer = PPOTrainer(model, physics, optimizer)
    env = ReadoutEnv(physics, CONFIG["state_dim"], device)

    return model, trainer, env


# ==========================================================
# Training Loop
# ==========================================================
def train():
    model, trainer, env = init_system()

    # --- tracking ---
    history = {
        "loss": [],
        "reward": [],
        "fidelity": [],
        "smoothness": [],
        "bandwidth": []
    }

    print("--------------- PPO INITIALIZED ---------------")

    for epoch in range(CONFIG["epochs"]):

        # -------- Rollout --------
        states = env.reset(CONFIG["batch_size"])

        actions, log_probs, values, rewards, infos = trainer.rollout(states)

        # -------- Advantage computation --------
        # single-step env => return = reward
        returns = rewards.detach()
        advantages = returns - values.detach()

        # -------- PPO Update --------
        loss = trainer.ppo_update(
            states,
            actions,
            log_probs.detach(),
            returns,
            advantages
        )

        # -------- Logging --------
        with torch.no_grad():
            avg_reward = rewards.mean().item()
            avg_fidelity = torch.mean(torch.tensor([i["fidelity"] for i in infos])).item()
            avg_smoothness = torch.mean(torch.tensor([i["smoothness"] for i in infos])).item()
            avg_bandwidth = torch.mean(torch.tensor([i["bandwidth"] for i in infos])).item()

        history["loss"].append(loss)
        history["reward"].append(avg_reward)
        history["fidelity"].append(avg_fidelity)
        history["smoothness"].append(avg_smoothness)
        history["bandwidth"].append(avg_bandwidth)

        if epoch % CONFIG["log_interval"] == 0:
            print(
                f"[{epoch:04d}] "
                f"loss={loss:.4f} | "
                f"reward={avg_reward:.4f} | "
                f"fid={avg_fidelity:.4f} | "
                f"smooth={avg_smoothness:.4f}"
            )

    return model, env, history


# ==========================================================
# Final Pulse Generation
# ==========================================================
def generate_final_pulse(model, env):
    model.eval()

    with torch.no_grad():
        state = env.reset(1)
        action, log_prob, value = model.act(state)   # PPO policy output
        pulse = action.squeeze(0)

        # Physics simulation
        reward, info, pulse_out, ng, ne, F, res = evaluate_pulse(pulse)

    return {
        "pulse": pulse_out,
        "ng": ng,
        "ne": ne,
        "F": F,
        "res": res,
        "info": info
    }




def plot_final_pulse(model, physics, state_dim, device):
    """
    Generate one pulse from the trained PPO policy and plot:
    - waveform
    - photon dynamics
    - fidelity
    - phase-space trajectory
    """

    model.eval()

    with torch.no_grad():
        # Stateless env → dummy state
        state = torch.zeros(1, state_dim).to(device)

        # PPO policy output
        dist, value = model(state)
        action = dist.mean.squeeze(0)  # deterministic PPO policy

        # Run physics
        reward, info, pulse, ng, ne, F, res = evaluate_pulse(action)

    # Move to CPU numpy
    pulse = pulse.cpu().numpy()
    ng = ng.cpu().numpy()
    ne = ne.cpu().numpy()
    F = F.cpu().numpy()
    res = res.cpu().numpy()

    # Phase-space
    I_g = res[:, 0]
    Q_g = res[:, 1]
    I_e = res[:, 2]
    Q_e = res[:, 3]

    # =========================
    # Plots
    # =========================

    # --- 1) Waveform ---
    plt.figure(figsize=(10, 4))
    plt.plot(t_action.cpu().numpy(), pulse, linewidth=2)
    plt.title("Learned Readout Pulse (PPO)")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

    # --- 2) Dynamics ---
    plt.figure(figsize=(12, 6))
    plt.plot(t_sim.cpu().numpy(), ng, label="Ground photons $n_g$")
    plt.plot(t_sim.cpu().numpy(), ne, label="Excited photons $n_e$")
    plt.plot(t_sim.cpu().numpy(), F, label="Fidelity", linestyle="--")
    plt.axhline(info["fidelity"].item(), color="red", linestyle=":",
                label=f"Max Fidelity = {info['fidelity'].item():.4f}")
    plt.title("Resonator Dynamics (PPO)")
    plt.xlabel("Time")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 3) Phase space ---
    plt.figure(figsize=(6, 6))
    plt.plot(I_g, Q_g, label="Ground $|g⟩$")
    plt.plot(I_e, Q_e, label="Excited $|e⟩$")
    plt.scatter([0], [0], c="black", marker="x", label="Vacuum")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title("Phase Space Trajectory (PPO)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Print metrics ---
    print("\nFinal pulse metrics:")
    print(f"Fidelity   : {info['fidelity'].item():.6f}")
    print(f"Reset time : {info['t_reset'].item():.6f}")
    print(f"Smoothness : {info['smoothness'].item():.6f}")
    print(f"Bandwidth  : {info['bandwidth'].item():.6f}")
    print(f"Reward     : {reward.item():.6f}")

# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":

    model, env, history = train()

    print("--------------- TRAINING COMPLETE ---------------")

    results = generate_final_pulse(model, env)

    # Save model
    os.makedirs("trained_models", exist_ok=True)
    torch.save(model.state_dict(), "trained_models/ppo_readout_policy.pth")

    print("Model saved to trained_models/ppo_readout_policy.pth")

    # Results ready for plotting
    # history -> training curves
    # results["pulse"] -> final waveform
    # results["ng"], results["ne"], results["F"] -> dynamics
    # results["res"] -> IQ phase space

    physics = ReadoutPhysics()
    plot_final_pulse(model, physics, CONFIG["state_dim"], device)