import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from .sim_core import evaluate_pulse

class QubitReadoutEnv(gym.Env):
    """
    Gymnasium environment for Qubit Readout.
    Treats the differentiable physics simulator as a black-box RL environment.
    """
    def __init__(self):
        super().__init__()
        # Action Space: The 128-point microwave pulse. Bounded between -1.0 and 1.0.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(128,), dtype=np.float32)
        
        # Observation Space: A dummy observation. 
        # Because the agent generates the entire pulse in one shot, there is no "state" to observe.
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Return the dummy starting state
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        # 1. Convert the agent's numpy array action back to a PyTorch tensor
        pulse_tensor = torch.tensor(action, dtype=torch.float32)

        # 2. Run the physics simulator!
        # CRITICAL: We use torch.no_grad() because RL does NOT use differentiable gradients.
        # This prevents the ODE solver from hoarding memory.
        with torch.no_grad():
            reward_tensor, info, pulse, ng, ne, F, res = evaluate_pulse(pulse_tensor)

        # 3. Extract the scalar reward
        reward = reward_tensor.item()

        # 4. The episode ends immediately after one pulse is evaluated
        terminated = True
        truncated = False
        obs = np.array([1.0], dtype=np.float32) # Dummy next state

        # 5. Clean up the info dict for Stable-Baselines3 (must be standard Python floats)
        clean_info = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in info.items()}

        return obs, reward, terminated, truncated, clean_info