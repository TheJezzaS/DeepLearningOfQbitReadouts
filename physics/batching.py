import torch

def simulate_batch(physics, pulses: torch.Tensor):
    results = []
    for p in pulses:
        results.append(physics.simulate(p))
    return results
