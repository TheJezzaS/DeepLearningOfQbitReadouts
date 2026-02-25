"""
High-level physics interface for ML training.

Provides a clean abstraction layer between deep learning models
and the differentiable physics simulator.
"""
import torch

from .sim_core import evaluate_pulse

class ReadoutPhysics:
    """
    Physics wrapper class for ML/DL integration.

    Provides a stable interface (`simulate`) that maps control pulses
    to physical metrics and rewards for training pipelines.

    Clean physics interface for ML/DL usage
    """

    def simulate(self, pulse: torch.Tensor) -> dict:
        """
        Simulate a control pulse through the physics model.

        Args:
            pulse (torch.Tensor): Control pulse signal.

        Returns:
            dict: Dictionary containing physical observables, metrics, and reward.
        """
        reward, info, pulse, ng, ne, F = evaluate_pulse(pulse)

        return {
            "fidelity": info["fidelity"],
            "max_pf": info["max_pf"],
            "max_photon": info["max_photon"],
            "bandwidth": info["bandwidth"],
            "smoothness": info["smoothness"],
            "t_reset": info["t_reset"],
            "reward": reward,
            "pulse": pulse,
            "ng": ng,
            "ne": ne
        }

