# NeuroChip AR

**Software-first brain-inspired neuromorphic computing prototype**

---

## Overview

NeuroChip AR is an open-source project building a biologically grounded spiking neural network (SNN) foundation in pure Python. The goal is to establish a reproducible, testable computational substrate before considering any embedded or hardware implementation.

The project is **not** a simulation of the human brain and does not claim to replicate biological cognition. It implements well-defined mathematical models of neuron dynamics, synaptic plasticity, and temporal information processing, verifiable against analytical solutions and reproducible across runs.

---

## Architecture

```
Neuro_chip_ar/
├── src/neurochip_ar/
│   ├── __init__.py                   # Public package exports
│   ├── neuron.py                     # Leaky Integrate-and-Fire neuron model
│   ├── encoding.py                   # Seeded Poisson rate encoder
│   ├── network.py                    # Feed-forward SNN with synaptic decay
│   ├── stdp.py                       # Pair-based STDP plasticity rule
│   ├── recurrent.py                  # Recurrent SNN with delayed feedback + online STDP
│   └── temporal_features.py          # Temporal feature extraction utilities  [new]
├── examples/
│   ├── basic_neurochip_demo.py       # Feed-forward & offline STDP demo
│   ├── recurrent_neurochip_demo.py   # Recurrent network & online STDP demo
│   └── temporal_pattern_classification.py  # Temporal learning experiment  [new]
├── tests/
│   ├── test_neuron.py                # LIF dynamics & refractory tests
│   ├── test_encoding.py              # Encoder dimensions & reproducibility
│   ├── test_network.py               # Feed-forward propagation & reset
│   ├── test_stdp.py                  # STDP LTP, LTD & weight-bound tests
│   ├── test_recurrent.py             # Recurrent delay, feedback & STDP tests
│   ├── test_recurrent_stdp_bounds.py # STDP weight-bound edge cases
│   ├── test_stdp_configuration.py    # Independent STDP component configuration  [new]
│   └── test_temporal_features.py     # Temporal feature extraction tests  [new]
├── artifacts/
│   ├── basic_neurochip_demo.png      # Feed-forward visualisation
│   ├── recurrent_neurochip_demo.png  # Recurrent network visualisation
│   ├── confusion_matrix.png          # Test-set confusion matrix (full-STDP)  [new]
│   ├── temporal_classification_results.csv   # Per-seed accuracy table  [new]
│   └── temporal_classification_report.txt    # Human-readable summary  [new]
├── run_all.sh                        # Portable experiment runner
├── conftest.py                       # pytest configuration
└── pyproject.toml                    # Build & dependency configuration
```

---

## Core Components

### 1. LIF Neuron — `neurochip_ar.neuron`

Implements leaky integrate-and-fire membrane dynamics via forward Euler integration:

$$\frac{dV}{dt} = \frac{V_{\text{rest}} - V + R \cdot I}{\tau_m}$$

- Resting potential, reset potential, and spike threshold (all in mV).
- Absolute refractory period — membrane is clamped at $V_{\text{reset}}$ and cannot spike.
- Euler stability check: rejects configurations where $dt / \tau_m \notin (0, 2)$.
- `params` property for safe, typed introspection of neuron configuration:
  ```python
  neuron.params["v_rest"]       # -65.0
  neuron.params["v_threshold"]  # -50.0
  neuron.params["tau_m"]        # 20.0
  ```
- `reset()` restores membrane potential to `v_rest` and clears refractory state.

### 2. Poisson Rate Encoder — `neurochip_ar.encoding`

Converts continuous firing rates $r \in [0, 1]$ into binary spike trains via a seeded Bernoulli process.

- Deterministic across runs given the same seed.
- Supports 1-D channel inputs and 2-D time-varying inputs.

### 3. Feed-Forward Spiking Network — `neurochip_ar.network`

Connects input spike trains to a layer of LIF neurons through a weight matrix `(n_output, n_input)`.

Synaptic current dynamics with exponential decay:

$$I_{\text{syn}}(t) = \alpha \cdot I_{\text{syn}}(t-1) + W \cdot S_{\text{in}}(t), \quad \alpha = e^{-dt/\tau_{\text{syn}}}$$

### 4. Pair-based STDP — `neurochip_ar.stdp`

Implements spike-timing-dependent plasticity using eligibility traces:

| Spike order | Effect | Rule |
|---|---|---|
| Pre before post | LTP (strengthening) | $\Delta w \propto A_+ \cdot x_{\text{pre}}$ |
| Post before pre | LTD (weakening) | $\Delta w \propto -A_- \cdot x_{\text{post}}$ |

- Configurable learning rates ($A_+$, $A_-$), trace time constants ($\tau_+$, $\tau_-$), and hard weight bounds $[w_{\min}, w_{\max}]$.
- Weights are clamped to bounds after every update.

### 5. Recurrent Spiking Network — `neurochip_ar.recurrent`

Extends the feed-forward model with one-timestep delayed recurrent feedback:

$$I_{\text{syn}}(t) = \alpha \cdot I_{\text{syn}}(t-1) + W_{\text{in}} \cdot x[t] + W_{\text{rec}} \cdot y[t-1]$$

- Zero diagonal enforced — no instantaneous self-connections.
- Non-negative recurrent weights only (excitatory connections at this milestone).
- **Independent STDP components**: `stdp_input` and `stdp_rec` are constructed separately and only when their respective learning rates are strictly positive. Zero or negative rates leave the attribute as `None`.

```python
# Input-only STDP — recurrent weights stay fixed
net = RecurrentSpikingNetwork(
    ...,
    enable_stdp=True,
    stdp_input_params={"a_plus": 0.01, "a_minus": 0.012},
    stdp_rec_params=None,   # or omit entirely
)
# net.stdp_input is a PairSTDP instance
# net.stdp_rec  is None
```

- `reset()` clears membrane potentials, synaptic currents, previous spikes, and STDP traces — **without** touching learned weights.
- `run(spikes, record_weight_history=True)` returns a dict with `spikes`, `voltages`, `i_syn`, and optionally `weight_history`.

### 6. Temporal Feature Extraction — `neurochip_ar.temporal_features` *(v0.3)*

Three deterministic feature extractors for spike-train analysis:

| Function | Description | Output shape |
|---|---|---|
| `extract_total(spikes)` | Total spike count per neuron | `(n_neurons,)` |
| `extract_bins(spikes, bins=4)` | Concatenated binned spike counts | `(n_neurons × bins,)` |
| `extract_latency(spikes)` | First-spike latency per neuron; silent neurons get `T` | `(n_neurons,)` |

All three functions accept either:
- A raw spike matrix `np.ndarray` of shape `(T, n_neurons)`.
- A `(network, spikes_input)` pair — the network is reset before each sample, transient state is cleared, but learned weights are preserved.

---

## Temporal Pattern Classification Experiment *(v0.3.2)*

`examples/temporal_pattern_classification.py` demonstrates a complete temporal learning pipeline:

### Dataset
- **Task**: Distinguish which input channel fires first in a Poisson spike train.
- **Size**: 200 samples per class → 400 total.
- **Split**: Stratified 70 / 15 / 15:

| Partition | Class 0 | Class 1 | Total |
|---|---|---|---|
| Training | 140 | 140 | 280 |
| Validation | 30 | 30 | 60 |
| Testing | 30 | 30 | 60 |

- Deterministic via fixed seeds; no cross-partition overlap.

### Evaluated configurations
- **Fixed weights** — no STDP; random initialisation.
- **Input-STDP** — STDP applied to input synapses only.
- **Full STDP** — STDP on both input and recurrent synapses.
- **Raw baseline** — time-binned input rates, no network.

### Feature representations
- Total spike counts.
- Binned spike counts (4 windows).
- First-spike latency.

### Protocol
- Train STDP on training partition only.
- Freeze STDP during evaluation.
- Reset transient network state between independent samples.
- Evaluate on held-out test set.
- Repeat across 10 fixed seeds.

### Generated artifacts

| File | Description |
|---|---|
| `artifacts/temporal_classification_results.csv` | Accuracy per seed & configuration |
| `artifacts/temporal_classification_report.txt` | Human-readable summary |
| `artifacts/confusion_matrix.png` | Test-set confusion matrix (full-STDP config) |

---

## Setup & Installation

Requirements: **Python 3.10+** (tested on 3.12 and 3.14).

```bash
# Clone the canonical repository
git clone https://github.com/StellarResearch/Neuro-Chip-AR.git
cd Neuro-Chip-AR

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

---

## Running the Experiments

### All experiments (portable runner)
```bash
bash run_all.sh
```
Automatically resolves the repository root, prefers `.venv/bin/python`, falls back to system `python3`, checks required packages, and runs all three scripts. Stops immediately on any error (`set -euo pipefail`).

### Individual scripts
```bash
# Feed-forward SNN & offline STDP demo
python examples/basic_neurochip_demo.py

# Recurrent network & online STDP demo
python examples/recurrent_neurochip_demo.py

# Temporal pattern classification experiment
python examples/temporal_pattern_classification.py
```

---

## Testing

The test suite covers unit dynamics, boundary conditions, parameter validation, end-to-end simulation pipelines, and temporal feature correctness.

```bash
# Run full test suite
python -m pytest -v

# Run focused test suites
pytest tests/test_stdp_configuration.py -v   # STDP configuration correctness
pytest tests/test_temporal_features.py -v    # Temporal feature extraction

# Lint and format checks
ruff check src/neurochip_ar tests examples
ruff format --check src/neurochip_ar tests examples
```

**Current status: 48 tests, all passing.**

---

## Learning Mode Comparison

| Mechanism | Update trigger | Weight effect | Learning signal |
|---|---|---|---|
| **Offline STDP** | Post-simulation | Evaluated over recorded histories | Local spike timing |
| **Online STDP** (input) | Every timestep | Updates input weights in real time | Local spike timing |
| **Online STDP** (recurrent) | Every timestep | Updates recurrent weights in real time | Local spike timing |
| **Supervised (future)** | Epoch / batch | Optimises global loss | Task labels |

---

## Current Status — v0.3.2

### Implemented
- Forward Euler LIF neuron with absolute refractory period and stability validation.
- Deterministic Poisson rate encoder.
- Feed-forward SNN with exponential synaptic current decay.
- Recurrent SNN with one-timestep delayed feedback and zero-diagonal constraint.
- Pair-based STDP with independent guards for input and recurrent synapses.
- Temporal feature extraction module (`total`, `binned`, `latency`).
- Stratified temporal pattern classification experiment with 10-seed evaluation.
- Portable experiment runner (`run_all.sh`).
- 48 automated tests (all passing).

### Known limitations
- Recurrent connections are excitatory-only; inhibitory populations are not yet implemented.
- Plasticity is pair-based STDP only — no triplet interactions, homeostatic normalisation, or short-term facilitation/depression.
- Pure Python and NumPy; no GPU, FPGA, or hardware acceleration.
- Temporal classification accuracy is near chance on the current synthetic task — STDP alone is insufficient for this problem without additional mechanisms.

---

## Roadmap

- [x] **v0.1** — LIF neuron, Poisson encoding, feed-forward network, pair-based STDP.
- [x] **v0.2** — Recurrent network, one-step delayed feedback, online STDP.
- [x] **v0.3** — Temporal feature extraction, stratified experiment, independent STDP configuration.
- [ ] **v0.4** — Excitatory / inhibitory populations, sparse connectivity, homeostatic regulation.
- [ ] **v0.5** — Surrogate-gradient supervised learning.
- [ ] **v0.6** — Benchmarking on standard neuromorphic datasets (N-MNIST, SHD).
- [ ] **Long-term** — Lightweight C/C++ inference runtime; FPGA exploration.

---

## License

This project is currently unlicensed. An open-source license will be selected before a formal public release.
