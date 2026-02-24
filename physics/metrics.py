import torch
def metrics_to_vector(m: dict) -> torch.Tensor:
    return torch.Tensor([
        m["fidelity"],
        m["snr"],
        m["bandwidth"],
        m["smoothness"],
        m["photon_time"]
    ], dtype=np.float32)
