FUNCTION step(action):

    # 1. Construct Drive Pulse
    drive = a0 * action

    drive = clip_amplitude(drive, mu)

    drive = apply_bandwidth_filter(drive)

    drive = smooth_if_needed(drive)

    # 2. Simulate Cavity ODE
    alpha_g, alpha_e = solve_ode(
        dα/dt = (-κ/2 ± iχ/2 + iK|α|²)α - i drive(t)
    )

    # 3. Compute Photon Numbers
    n_g = |alpha_g|²
    n_e = |alpha_e|²
    n = max(n_g, n_e)

    n_max = max(n)

    # 4. Compute IQ Separation
    separation = |alpha_g - alpha_e|

    # 5. Compute Fidelity
    F = 0.5 * (1 + erf(λ * separation))

    Γ = Γ_I + Γ_ph * n_g
    survival = exp(- cumulative_integral(Γ))

    F = F * survival

    pF = -log10(1 - F)

    pF_max = max(pF)
    t_pf = argmax(pF)

    # 6. Compute Reset Time
    n0 = photon_at_pulse_end
    t_reset = t_end + (1/κ) * log(n0 / n_ideal)

    # 7. Compute Smoothness
    smoothness = integral( (second_derivative(drive))² )

    # 8. Compute Bandwidth
    BW = compute_fft_bandwidth(drive)

    # 9. Compute Reward
    reward =
        pf_reward(pF_max)
        - time_penalty(t_reset)
        - photon_penalty(n_max)
        - bandwidth_penalty(BW)
        - smoothness_penalty(smoothness)
        - amplitude_penalty(drive)

    RETURN reward
