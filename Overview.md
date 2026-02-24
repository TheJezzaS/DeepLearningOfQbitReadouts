# **Differentiable Physics–Deep Learning Framework**
## Project Architecture & Execution Flow

---

## 1. Project Purpose

This project implements a **physics-based simulator** of a driven quantum resonator system and integrates it with **deep learning architectures** to study learning-based control.

The goal is to:
- Learn optimal control pulses for a physical system  
- Apply different deep learning models to the *same* physics  
- Compare architectures in a controlled, scientific way  
- Enable reproducible benchmarking of ML methods on physical dynamics  

The system is designed so that:
- **Physics is fixed**
- **Models are interchangeable**
- **Training logic is shared**
- **Experiments are comparable**

---

## 2. High-Level Structure

physics/ → physical simulator (system dynamics)
models/ → deep learning architectures
training/ → training logic and optimizers
configs/ → hyperparameters and constants
experiments/ → executable entry points (main programs)


Each module has a single responsibility:

- `physics/` defines how the physical system behaves  
- `models/` defines how control signals are generated  
- `training/` defines how learning happens  
- `experiments/` defines which model is evaluated  

---

## 3. Experiments Folder = Main Entry Points

The `experiments/` folder contains the **main runnable programs**:


experiments/
exp_mlp.py
exp_cnn.py
exp_transformer.py
exp_diffusion.py
exp_rl.py
exp_baseline.py


Each file represents a **separate scientific experiment**:

- One model architecture  
- Same physics simulator  
- Same reward structure  
- Same training logic  
- Comparable outputs  

**Meaning:**

> Each experiment file runs one model family on the same physical system.

Running a file in `experiments/` is equivalent to running a controlled scientific experiment.

---

## 4. Conceptual Roles

| Component | Meaning |
|------|------|
| physics | the real system |
| models | hypotheses |
| training | learning mechanism |
| experiments | scientific trials |
| results | empirical comparison |

---

## 5. Runtime Execution Flow

When an experiment file is executed, the following pipeline occurs:

---

### Step 1 — Model Construction

A deep learning model is instantiated (e.g. MLP, CNN, Transformer, RL policy).

The model maps inputs → control actions.

---

### Step 2 — Physics Simulator Construction

A physics simulator is created with fixed physical parameters:

- resonator decay  
- dispersive coupling  
- Kerr nonlinearity  
- decay processes  
- measurement model  

This simulator represents the **physical world** and is identical across all experiments.

---

### Step 3 — Trainer Construction

A trainer connects:

model → physics → loss → optimizer


The trainer is model-agnostic and physics-agnostic.

---

### Step 4 — Training Loop

Each iteration follows the same pipeline:

---

#### 4.1 Model Forward Pass

action = model(input)


The model outputs a control signal (pulse vector).

---

#### 4.2 Pulse Construction

The raw model output is converted into a physical drive pulse:

- amplitude scaling  
- clipping  
- smoothing  
- gradient limiting  
- bandwidth filtering  

This produces a **physically valid control signal**.

---

#### 4.3 Physics Simulation

The pulse is injected into the simulator.

The simulator solves the cavity dynamics:

dα/dt = (-κ/2 ± iχ/2 + iK|α|²)α - i drive(t)


producing:

- ground-state amplitude α_g(t)
- excited-state amplitude α_e(t)

---

#### 4.4 Physical Observables

From the dynamics:


n_g(t) = |α_g(t)|²
n_e(t) = |α_e(t)|²
separation(t) = |α_g(t) - α_e(t)|


---

#### 4.5 Measurement Model

Measurement fidelity:

F(t) = 0.5 * (1 + erf(λ * separation(t)))


with decay and survival:

Γ(t) = Γ_I + Γ_ph * n_g(t)
survival(t) = exp(-∫ Γ(t) dt)
F_eff(t) = F(t) * survival(t)

Performance metric:

pF(t) = -log10(1 - F_eff(t))


Key values:

- `pF_max` = max performance  
- `t_pf` = time of max performance  

---

#### 4.6 Reset Dynamics

Residual photons determine reset time:

t_reset = t_end + (1/κ) * log(n0 / n_ideal)


---

#### 4.7 Regularization Metrics

Control quality penalties:

- smoothness  
- bandwidth  
- amplitude  

---

#### 4.8 Reward / Loss

reward =
+ w_pf * pF_max
- w_time * t_reset
- w_photon * n_max
- w_bw * bandwidth
- w_smooth * smoothness
- w_amp * amplitude


Training loss:

loss = -reward


---

#### 4.9 Optimization

loss.backward()
optimizer.step()
optimizer.zero_grad()


---

## 6. Architectural Properties

- Physics is shared across all experiments  
- Training logic is shared  
- Only the model architecture changes  
- Results are directly comparable  
- Experiments are reproducible  
- Models are plug-and-play  

---

## 7. Research Meaning

This framework enables **scientific benchmarking of deep learning architectures** on a fixed physical system:

- Same physics  
- Same metrics  
- Same reward  
- Same constraints  
- Different models  

This allows fair comparison between:
- MLPs
- CNNs
- Transformers
- RL policies
- generative models
- heuristic baselines

---

## 8. Design Philosophy

Physics ≠ Model
Model ≠ Training
Training ≠ Experiment
Experiment ≠ Architecture


All components are modular and interchangeable.

---

## 9. Final Summary

This project implements a:

> **Differentiable physical simulator + modular deep learning benchmarking framework**

where:

- `physics/` = reality  
- `models/` = hypotheses  
- `training/` = learning  
- `experiments/` = science  
- `results/` = knowledge  

Running a file in `experiments/` is not just running code —  
it is running a **scientific experiment** on a physical system using a specific learning architecture.
