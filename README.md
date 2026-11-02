# NeuroChip AR

**Software-first brain-inspired AI and neuromorphic computing prototype**

---

## Overview
NeuroChip AR is an open-source project exploring software-based neuromorphic computing and spiking neural networks (SNNs). The project implements biologically inspired neuron dynamics, synaptic connectivity, and local plasticity mechanisms in pure Python, establishing a verifiable computational foundation before considering embedded or hardware implementations.

---

## Core Components

### 1. LIF Neuron (`neurochip_ar.neuron`)
Implements leaky integrate-and-fire membrane dynamics using forward Euler integration:
$$\frac{dV}{dt} = \frac{V_{\text{rest}} - V + R \cdot I}{\tau_m}$$
- Resting potential ($V_{\text{rest}}$), reset potential ($V_{\text{reset}}$), and spike threshold ($V_{\text{threshold}}$).
- Absolute refractory period clamping membrane potential.
- Strict parameter and boundary validation.

### 2. Poisson Rate Encoding (`neurochip_ar.encoding`)
- Converts continuous normalized inputs in $[0, 1]$ into binary spike trains.
- Seeded pseudo-random number generator for deterministic reproducibility.
- Supports 1D channel inputs and 2D time-varying inputs.

### 3. Feed-Forward Spiking Network (`neurochip_ar.network`)
- Connects input spike trains to output LIF neurons through a weight matrix of shape `(n_output, n_input)`.
- Exponentially decaying synaptic current dynamics:
$$I_{\text{syn}}(t) = I_{\text{syn}}(t-1) \cdot \exp\left(-\frac{dt}{\tau_{\text{syn}}}\right) + W \cdot S_{\text{in}}(t)$$
- Deterministic simulation recording membrane potentials, synaptic currents, and output spikes.

### 4. Pair-based STDP (`neurochip_ar.stdp`)
- Implements spike-timing-dependent plasticity using presynaptic and postsynaptic eligibility traces.
- Pre-before-post spike timing triggers Long-Term Potentiation (LTP).
- Post-before-pre spike timing triggers Long-Term Depression (LTD).
- Configurable learning rates, decay time constants, and hard weight boundaries $[w_{\text{min}}, w_{\text{max}}]$.
- Documented simultaneous spike convention.

### 5. Recurrent Spiking Network & Online STDP (`neurochip_ar.recurrent`) *(v0.2)*
- Connects external inputs ($x[t]$) and recurrent outputs ($y[t-1]$) to a layer of LIF neurons.
- Explicit one-timestep delayed recurrent feedback without instantaneous self-connections:
$$I_{\text{syn}}(t) = I_{\text{syn}}(t-1) \cdot \exp\left(-\frac{dt}{\tau_{\text{syn}}}\right) + W_{\text{in}} \cdot x[t] + W_{\text{rec}} \cdot y[t-1]$$
- Strictly enforces non-negative recurrent weights and zero diagonal (no instantaneous self-excitation).
- **Online STDP**: updates both input and recurrent weight matrices inside the simulation loop at every timestep:
  - Input weights updated via pairings between $x[t]$ and $y[t]$.
  - Recurrent weights updated via pairings between delayed recurrent spikes $y[t-1]$ and current spikes $y[t]$.
  - Updated weights immediately influence subsequent network timesteps.
  - State resets clear membrane voltages and synaptic currents while preserving learned weights.

---

## Offline vs. Online STDP vs. Supervised Learning

| Mechanism | Update Frequency | Weight Effect | Learning Signal |
|---|---|---|---|
| **Offline STDP** | Post-simulation | Evaluated over recorded spike histories | Local spike timing |
| **Online STDP** | Per simulation timestep | Dynamically influences subsequent steps in real time | Local spike timing |
| **Supervised Learning** *(Future)* | Epoch / batch | Optimizes loss function via surrogate gradients | Global task target / labels |

---

## Repository Structure

```
Neuro_chip_ar/
├── pyproject.toml                     # Project configuration and dependencies
├── README.md                          # Project documentation
├── src/
│   └── neurochip_ar/
│       ├── __init__.py                # Package exports
│       ├── neuron.py                  # Leaky Integrate-and-Fire neuron model
│       ├── encoding.py                # Seeded Poisson rate encoder
│       ├── network.py                 # Feed-forward SNN with synaptic current decay
│       ├── stdp.py                    # Pair-based STDP learning rule
│       └── recurrent.py               # Recurrent SNN with delayed feedback and online STDP
├── examples/
│   ├── basic_neurochip_demo.py        # Feed-forward SNN & STDP demonstration
│   └── recurrent_neurochip_demo.py    # Recurrent network & online STDP demonstration
├── tests/
│   ├── test_neuron.py                 # LIF dynamics & refractory tests
│   ├── test_encoding.py               # Encoder dimensions & reproducibility tests
│   ├── test_network.py                # Feed-forward propagation & reset tests
│   ├── test_stdp.py                   # STDP LTP, LTD & boundary tests
│   └── test_recurrent.py              # Recurrent delay, feedback & online STDP tests
└── artifacts/
    ├── basic_neurochip_demo.png        # Feed-forward simulation visualization
    └── recurrent_neurochip_demo.png    # Recurrent network & online STDP visualization
```

---

## Setup & Installation

Requirements: Python 3.10+ (tested on Python 3.12 and 3.14).

```bash
# Clone the repository
git clone https://github.com/AryanSaxena20/Neuro_chip_ar.git
cd Neuro_chip_ar

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package and dependencies in editable mode
pip install -e ".[dev]"
```

---

## Running the Demonstrations

### 1. Basic Feed-Forward & STDP Demo
```bash
python examples/basic_neurochip_demo.py
```
- Simulates feed-forward integration through LIF output neurons.
- Executes controlled STDP pairing across spike timing delays ($\Delta t$).
- Saves 4-panel analysis plot to `artifacts/basic_neurochip_demo.png`.

### 2. Recurrent Network & Online STDP Demo
```bash
python examples/recurrent_neurochip_demo.py
```
- Simulates external Poisson inputs driving recurrent LIF neurons.
- Demonstrates one-timestep delayed feedback in a recurrent ring topology.
- Performs online STDP weight adaptation at every timestep.
- Saves raster plots, membrane potentials, and weight trajectories to `artifacts/recurrent_neurochip_demo.png`.

---

## Testing & Verification

The test suite covers unit dynamics, boundary conditions, parameter validation, and end-to-end simulation pipelines:

```bash
# Run automated test suite
python -m pytest -v

# Run linter and formatting checks
ruff check .
ruff format --check .
```

---

## Status and Scope

### Current Implementation (v0.2)
- Numerical Leaky Integrate-and-Fire neuron with forward Euler integration and configurable refractory period.
- Deterministic Poisson rate encoding with reproducible seeds.
- Feed-forward spiking network with exponential synaptic current decay.
- Recurrent spiking network with one-timestep delayed feedback and zero diagonal self-connection constraint.
- Pair-based Spike-Timing-Dependent Plasticity (STDP) with eligibility traces and hard weight clamping.
- Online STDP learning executed inside the simulation timestep loop for both input and recurrent connections.
- Comprehensive test suite (30 automated tests passing).
- Reproducible demonstrations with Matplotlib visualizations.

### Known Limitations
- Recurrent connections are currently excitatory-only (non-negative weights); inhibitory plasticity is planned for future milestones.
- Plasticity is local pair-based STDP without triplet interactions or homeostatic normalization.
- Pure Python and NumPy execution without hardware acceleration.

---

## Development Roadmap

- [x] **Milestone 0.1**: Scaffold, LIF neuron, Poisson encoding, feed-forward network, pair-based STDP.
- [x] **Milestone 0.2**: Recurrent network architecture, one-step delayed feedback, online STDP.
- [ ] **Milestone 0.3**: Temporal spike encoding schemes (latency coding, phase encoding) and lateral inhibition.
- [ ] **Milestone 0.4**: Surrogate gradient learning integration for supervised SNN tasks.
- [ ] **Milestone 0.5**: Benchmarking on standard neuromorphic datasets.
- [ ] **Long-term**: Lightweight embedded C/C++ inference runtime and FPGA acceleration exploration.

---

## License

This project is currently unlicensed. An open-source license will be selected before a formal public release.
