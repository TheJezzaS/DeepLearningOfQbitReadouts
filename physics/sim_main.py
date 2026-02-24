import numpy as np
import torch

from .sim_core import evaluate_pulse

class ReadoutPhysics:
    """
    Clean physics interface for ML/DL usage
    """

    def simulate(self, pulse: torch.Tensor) -> dict:
        reward, info, pulse, ng, ne, F = evaluate_pulse(pulse)

        return {
            "fidelity": torch.mean(F).detach().item(), # there are multiple F vals, so take mean
            "ng": ng,
            "ne": ne,
            "reward": reward,
            "snr": info.get("snr", 0.0),
            "bandwidth": info.get("bandwidth", 0.0),
            "smoothness": info.get("smoothness", 0.0),
            "photon_time": info.get("photon_time", 0.0),
            "pulse": pulse,
        }
