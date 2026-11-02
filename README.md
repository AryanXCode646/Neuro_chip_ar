# NeuroChip AR

**Software-first brain-inspired AI and neuromorphic computing prototype**

---

## Overview
NeuroChip AR is an exploratory open-source project exploring software-based neuromorphic computing and spiking neural networks (SNNs). The project implements biologically inspired neuron dynamics, synaptic connectivity, and local plasticity mechanisms in pure Python, establishing a verifiable computational foundation before considering embedded or hardware implementations.

## Core Components (v0.1 Implemented)

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

---

## Repository Structure

```
Neuro_chip_ar/
├── pyproject.toml              # Project configuration and dependencies
├── README.md                   # Project documentation
├── src/
│   └── neurochip_ar/
│       ├── __init__.py         # Package exports
│       ├── neuron.py           # Leaky Integrate-and-Fire neuron model
│       ├── encoding.py         # Seeded Poisson rate encoder
│       ├── network.py          # Feed-forward SNN with synaptic current decay
│       └── stdp.py             # Pair-based STDP learning rule
├── examples/
│   └── basic_neurochip_demo.py # End-to-end execution script
├── tests/
│   ├── test_neuron.py          # LIF dynamics & refractory tests
│   ├── test_encoding.py        # Encoder dimensions & reproducibility tests
│   ├── test_network.py         # Network propagation & reset tests
│   └── test_stdp.py            # STDP LTP, LTD & boundary tests
└── artifacts/
    └── basic_neurochip_demo.png # Generated simulation visualization
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

## Running the Demonstration

An end-to-end simulation demonstration is provided in `examples/basic_neurochip_demo.py`:

```bash
python examples/basic_neurochip_demo.py
```

The script:
1. Encodes continuous input stimuli into reproducible Poisson spike trains.
2. Simulates feed-forward integration through LIF output neurons.
3. Applies pair-based STDP updates to synaptic weights.
4. Executes a controlled STDP pairing experiment across a range of spike-timing delays ($\Delta t$).
5. Generates and saves a 4-panel analysis plot to `artifacts/basic_neurochip_demo.png`.

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

### Current Implementation (v0.1)
- Numerical Leaky Integrate-and-Fire neuron with forward Euler integration and configurable refractory period.
- Deterministic Poisson rate encoding with reproducible seeds.
- Feed-forward spiking network with exponential synaptic current decay.
- Pair-based Spike-Timing-Dependent Plasticity (STDP) with eligibility traces and hard weight clamping.
- Comprehensive unit test suite covering dynamics, edge cases, and end-to-end simulation.
- Interactive demonstration script with Matplotlib visualization.

### Known Limitations
- Network topologies are currently limited to single-layer feed-forward connections (no lateral inhibition or recurrence yet).
- Plasticity is pair-based STDP without homeostatic mechanisms.
- Pure Python and NumPy execution without hardware acceleration or specialized SNN backends.

---

## Future Roadmap

- **Milestone 0.2**: Recurrent spiking network topologies and lateral inhibition.
- **Milestone 0.3**: Temporal spike encoding schemes (time-to-first-spike, phase encoding).
- **Milestone 0.4**: Surrogate gradient learning integration for supervised SNN tasks.
- **Milestone 0.5**: Benchmarking on standard neuromorphic datasets.
- **Long-term**: Lightweight embedded C/C++ inference runtime and FPGA acceleration exploration.

---

## License

This project is currently unlicensed. An open-source license will be selected before a formal public release.
