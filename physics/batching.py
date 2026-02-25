"""
Batch simulation utilities.

Provides helper functions to run the physics simulator on batches of control pulses.
Used to bridge ML batch outputs with single-trajectory physics simulation.
"""

import torch

def simulate_batch(physics, pulses: torch.Tensor):
    """
    Simulate a batch of control pulses through the physics model.

    Args:
        physics: Physics simulator object implementing a `.simulate(pulse)` method.
        pulses (torch.Tensor): Tensor of shape (B, T) where B is batch size and
                               T is pulse length.

    Returns:
        list[dict]: List of physics simulation outputs, one per pulse.
                    Each element is the output dict returned by physics.simulate().
    """
    results = []
    for p in pulses:
        results.append(physics.simulate(p))
    return results