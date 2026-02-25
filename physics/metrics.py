"""
Metric vectorization utilities.

Converts physics output dictionaries into fixed-size torch tensors
for use in neural network training, logging, and evaluation pipelines.
"""


import torch

def metrics_to_vector(m: dict) -> torch.Tensor:
    """
    Convert physics metric dictionary into a fixed-order tensor vector.

    Expected keys in `m`:
        - "fidelity"
        - "max_pf"
        - "max_photon"
        - "bandwidth"
        - "smoothness"
        - "t_reset"

    Args:
        m (dict): Dictionary of scalar physics metrics.

    Returns:
        torch.Tensor: Tensor of shape (5,) containing metrics in fixed order.

    Note:
        The ordering is fixed and must remain consistent with downstream
        training pipelines and logging systems.
    """
    return torch.stack([
        m["fidelity"],
        m["max_pf"],
        m["max_photon"],
        m["bandwidth"],
        m["smoothness"],
        m["t_reset"],
    ])

