import matplotlib.pyplot as plt
import numpy as np
from torchdiffeq import odeint
import torch
import torch.nn.functional as F

# ==============================
# Time grids
# ==============================
T1 = 0.6  # total simulation time
N_SIM = 361  # simulation resolution
N_ACTION = 121  # pulse resolution

t_sim = torch.linspace(0, T1, N_SIM)
t_action = torch.linspace(0, T1, N_ACTION)

# ==============================
# Physical parameters, taken from source paper
# ==============================
kappa = 1.0  # resonator decay rate
chi = 0.2  # dispersive shift
kerr = 0.002  # Kerr nonlinearity
gamma_I = 1 / 26  # qubit decay
photon_gamma = 1 / 300  # photon induced decay

a0 = 1.0  # drive scaling
mu = 2.3  # amplitude limit
ideal_photon = 0.05  # target photon number after reset


# ==============================
# Utility functions
# ==============================
def gaussian_kernel(length=15, std=2.0):
    x = torch.linspace(-length // 2, length // 2, length)
    g = torch.exp(-x ** 2 / (2 * std ** 2))
    return g / g.sum()




def smooth_signal(signal, kernel):
    """
    signal: 1D torch tensor
    kernel: 1D torch tensor
    returns: 1D torch tensor (same length as signal)
    """

    # Ensure tensors
    signal = signal.float()
    kernel = kernel.float()

    # Reshape for conv1d
    signal = signal.view(1, 1, -1)          # (batch=1, channels=1, length)
    kernel = kernel.view(1, 1, -1)          # (out_ch=1, in_ch=1, kernel_size)

    # SAME padding
    padding = (kernel.shape[-1] - 1) // 2

    smoothed = F.conv1d(signal, kernel, padding=padding)

    return smoothed.view(-1)



def normalize_pulse(pulse, max_amp):
    scale = torch.clamp(max_amp / torch.abs(pulse), 0, 1)
    return pulse * scale


def gradient_clip(pulse, max_grad, dt):
    dp = torch.diff(pulse) / dt
    dp = torch.clamp(dp, -max_grad, max_grad)
    out = torch.cumsum(torch.cat((pulse[0:1], dp * dt)), dim=0)
    return out


def bandwidth_limit(pulse, max_freq, t_action):
    """
    pulse: 1D torch tensor
    max_freq: scalar
    t_action: 1D torch tensor (time axis)
    """

    pulse = pulse.float()
    dt = t_action[1] - t_action[0]

    # Torch FFT frequency bins
    freqs = torch.fft.fftfreq(len(pulse), d=dt)

    # Forward FFT
    F = torch.fft.fft(pulse)

    # Frequency mask
    mask = torch.abs(freqs) < max_freq

    # Apply filter
    F_filtered = F * mask

    # Inverse FFT
    filtered = torch.fft.ifft(F_filtered)

    return filtered.real



# ==============================
# Langevin ODE model
# ==============================
def resonator_ode(t, y, drive_func):
    rg_r, rg_i, re_r, re_i = y
    drive = drive_func(t)

    rg = rg_r + 1j * rg_i
    re = re_r + 1j * re_i

    d_rg = rg * (-0.5 * kappa - 0.5j * chi + 1j * kerr * torch.abs(rg) ** 2) - 1j * drive
    d_re = re * (-0.5 * kappa + 0.5j * chi + 1j * kerr * torch.abs(re) ** 2) - 1j * drive

    return torch.stack([
        d_rg.real, d_rg.imag,
        d_re.real, d_re.imag
    ])


def simulate_resonator(drive):
    """
    drive: 1D torch tensor
    Returns: tensor of shape (time, 4)
    """

    device = drive.device
    dtype = drive.dtype

    # Ensure time tensors are torch tensors
    t_sim_torch = torch.as_tensor(t_sim, device=device, dtype=dtype)

    # Initial state
    y0 = torch.zeros(4, device=device, dtype=dtype)

    # Define drive interpolation (must use torch only)
    def drive_func(t):
        t_action_local = t_action.to(device=drive.device, dtype=drive.dtype)
        return interp1d_torch(drive, x=t_action_local, t=t.unsqueeze(0))[0]

    # ODE function
    def ode_func(t, y):
        return resonator_ode(t, y, drive_func)

    # Solve ODE (RK45 equivalent)
    sol = odeint(
        ode_func,
        y0,
        t_sim_torch,
        method="dopri5"   # equivalent to RK45
    )

    # torchdiffeq returns shape (time, state)
    return sol



# ==============================
# Physics metrics
# ==============================
def photons(res):
    rg = res[:, 0] + 1j * res[:, 1]
    re = res[:, 2] + 1j * res[:, 3]
    ng = torch.abs(rg) ** 2
    ne = torch.abs(re) ** 2
    return ng, ne


def fidelity(ng, ne, scale=10):
    sep = torch.abs(torch.sqrt(ng) - torch.sqrt(ne))
    F = 0.5 * (1 + torch.erf(sep * scale))
    return F


def smoothness_metric(pulse):
    # second derivative (curvature)
    second_derivative = torch.diff(pulse, n=2)

    # square curvature
    curvature_energy = second_derivative**2

    # integrate over time
    smoothness = torch.trapz(curvature_energy, dx=1.0)
    return smoothness



def bandwidth_metric(pulse, t_action, thresh=0.05):
    """
    pulse: 1D torch tensor
    t_action: 1D torch tensor (time axis)
    """

    pulse = pulse.float()
    dt = t_action[1] - t_action[0]

    # FFT + shift
    F = torch.abs(torch.fft.fftshift(torch.fft.fft(pulse)))

    # Frequency axis
    freqs = torch.fft.fftshift(
        torch.fft.fftfreq(len(pulse), d=dt)
    )

    maxF = torch.max(F)

    idx = torch.where(F > thresh * maxF)[0]

    if idx.numel() == 0:
        return torch.tensor(0.0, device=pulse.device)

    bandwidth = torch.abs(freqs[idx[-1]] - freqs[idx[0]])
    return bandwidth



# ==============================
# Reward function
# ==============================
def compute_reward(F, ng, pulse):
    max_pf = -torch.log10(1 - torch.max(F) + 1e-9)
    max_photon = torch.max(ng)
    smooth = smoothness_metric(pulse)
    bw = bandwidth_metric(pulse, t_action)

    reward = (
            + 10 * max_pf
            - 5 * max_photon
            - 2 * smooth
            - 2 * bw
    )
    return reward, {
        "max_pf": max_pf,
        "max_photon": max_photon,
        "smoothness": smooth,
        "bandwidth": bw
    }


# ==============================
# numpy in pytorch functions
# ==============================
def interp1d_torch(y, x, t):
    """
    True linear interpolation at positions t.
    Assumes x is sorted and uniform.
    """
    y = y.float()
    x = x.float()
    t = t.float()

    dt = x[1] - x[0]

    idx_float = (t - x[0]) / dt
    idx0 = torch.floor(idx_float).long()
    idx1 = torch.clamp(idx0 + 1, max=len(y) - 1)

    idx0 = torch.clamp(idx0, 0, len(y) - 1)

    w = idx_float - idx0.float()

    return (1 - w) * y[idx0] + w * y[idx1]


def resize_1d(y, N):
    """
    Resize 1D tensor y to length N in pytorch tensor
    Similar to np.resize: repeats or interpolates as needed.
    """
    y = y.float()
    y_reshaped = y.view(1, 1, -1)
    y_resized = F.interpolate(y_reshaped, size=N, mode='linear', align_corners=True)
    return y_resized.view(-1)

# ==============================
# Full pipeline
# ==============================
def evaluate_pulse(action):
    # Ensure input is correct length
    if len(action) != N_ACTION:
        action = resize_1d(action, N_ACTION)

    kernel = gaussian_kernel(15, 2.0)

    pulse = normalize_pulse(action, mu)
    pulse = smooth_signal(pulse, kernel)
    pulse = gradient_clip(pulse, max_grad=40, dt=t_action[1] - t_action[0])
    pulse = bandwidth_limit(pulse, max_freq=50, t_action=t_action)
    pulse = a0 * pulse

    res = simulate_resonator(pulse)
    ng, ne = photons(res)
    F = fidelity(ng, ne)

    reward, info = compute_reward(F, ng, pulse)

    # Ensure outputs are fixed-size arrays for DL
    pulse = resize_1d(pulse, N_ACTION)
    ng = resize_1d(ng, N_SIM)
    ne = resize_1d(ne, N_SIM)
    F = resize_1d(F, N_SIM)

    return reward, info, pulse, ng, ne, F



if __name__ == "__main__":
    # random pulse
    action = (torch.rand(N_ACTION) * 2) - 1

    reward, info, pulse, ng, ne, F = evaluate_pulse(action)

    print("Reward:", reward)
    print("Info:", info)

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t_action, pulse)
    plt.title("Drive pulse")

    plt.subplot(3, 1, 2)
    plt.plot(t_sim, ng, label="ground photons")
    plt.plot(t_sim, ne, label="excited photons")
    plt.legend()
    plt.title("Photon number")

    plt.subplot(3, 1, 3)
    plt.plot(t_sim, F)
    plt.title("Fidelity vs time")

    plt.tight_layout()
    plt.show()
