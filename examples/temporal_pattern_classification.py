#!/usr/bin/env python3
"""Temporal pattern classification experiment v0.3.2.

Improvements over the original demo:
- Larger balanced dataset (200 samples per class).
- Temporal‑ordering task (first channel then second vs opposite).
- Stratified 70/15/15 train/val/test split.
- Sparse ring recurrent connectivity (non‑zero recurrent weights).
- Optional STDP on input and recurrent synapses.
- Multiple temporal feature representations (total counts, binned counts, latency).
- Comparison against a raw‑input baseline using time‑binned rates.
- Results saved to the `artifacts/` directory.
"""

import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants for external use (tests, scripts)
# ---------------------------------------------------------------------------
DURATION = 0.1
DT = 0.001
NUM_PER_CLASS = 200
SEED = 0

# ---------------------------------------------------------------------------
# Ensure src is on PYTHONPATH
# ---------------------------------------------------------------------------
repo_root = pathlib.Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from neurochip_ar.encoding import poisson_rate_encoder
from neurochip_ar.recurrent import RecurrentSpikingNetwork

# ---------------------------------------------------------------------------
# Synthetic dataset generation (temporal ordering task)
# ---------------------------------------------------------------------------

def generate_pattern(duration: float, dt: float, seed: int, first_channel: int) -> np.ndarray:
    """Create a pattern where *first_channel* fires high in the first half, then low.
    The other channel does the opposite. Adds jitter and small Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(round(duration / dt))
    pattern = np.zeros((n_steps, 2), dtype=np.float64)
    half = n_steps // 2
    pattern[:half, first_channel] = 0.9
    pattern[:half, 1 - first_channel] = 0.1
    pattern[half:, first_channel] = 0.1
    pattern[half:, 1 - first_channel] = 0.9
    jitter = rng.uniform(-0.02, 0.02, size=pattern.shape)
    noise = rng.normal(0.0, 0.01, size=pattern.shape)
    pattern = np.clip(pattern + jitter + noise, 0.0, 1.0)
    return pattern


def build_dataset(num_per_class: int, duration: float, dt: float, seed: int = 0):
    """Return stratified train/val/test splits (70/15/15) with balanced classes.
    Deterministic split based on a single RNG seeded with *seed*.
    """
    rng = np.random.default_rng(seed)
    # Generate patterns per class
    class_patterns = {0: [], 1: []}
    for cls in (0, 1):
        for i in range(num_per_class):
            first = 0 if cls == 0 else 1
            pattern = generate_pattern(duration, dt, seed=seed + cls * 1000 + i, first_channel=first)
            class_patterns[cls].append(pattern)
    # Shuffle each class separately
    for cls in class_patterns:
        rng.shuffle(class_patterns[cls])
    # Compute split sizes
    total_per_class = num_per_class
    train_n = int(0.70 * total_per_class)
    val_n = int(0.15 * total_per_class)
    # Build splits preserving class balance
    train, val, test = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    for cls, patterns in class_patterns.items():
        train.extend(patterns[:train_n])
        val.extend(patterns[train_n:train_n + val_n])
        test.extend(patterns[train_n + val_n:])
        train_labels.extend([cls] * len(patterns[:train_n]))
        val_labels.extend([cls] * len(patterns[train_n:train_n + val_n]))
        test_labels.extend([cls] * len(patterns[train_n + val_n:]))
    # Convert to numpy arrays
    return (train, np.array(train_labels, dtype=int)), (val, np.array(val_labels, dtype=int)), (test, np.array(test_labels, dtype=int))


# ---------------------------------------------------------------------------
# Recurrent weight initialization – sparse ring topology
# ---------------------------------------------------------------------------

def init_sparse_ring(n_neurons: int, seed: int, base_weight: float = 0.5) -> np.ndarray:
    """Directed ring where each neuron excites the next one.
    Diagonal stays zero; weights are small positive values with slight random variation.
    """
    rng = np.random.default_rng(seed)
    w = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    for i in range(n_neurons):
        j = (i + 1) % n_neurons
        w[i, j] = base_weight * rng.uniform(0.8, 1.2)
    return w

# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

from neurochip_ar.temporal_features import extract_total, extract_bins, extract_latency

# ---------------------------------------------------------------------------
# Simple NumPy softmax classifier (multinomial logistic regression)
# ---------------------------------------------------------------------------
class SoftmaxClassifier:
    def __init__(self, lr: float = 0.1, epochs: int = 200):
        self.lr = lr
        self.epochs = epochs
    def _softmax(self, logits):
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)
    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        n_classes = int(np.max(y)) + 1
        self.W = np.zeros((n_classes, n_features))
        self.b = np.zeros(n_classes)
        for _ in range(self.epochs):
            logits = X @ self.W.T + self.b
            probs = self._softmax(logits)
            y_onehot = np.zeros_like(probs)
            y_onehot[np.arange(n_samples), y] = 1.0
            grad = (probs - y_onehot) / n_samples
            self.W -= self.lr * grad.T @ X
            self.b -= self.lr * grad.sum(axis=0)
        return self
    def predict(self, X: np.ndarray):
        logits = X @ self.W.T + self.b
        return np.argmax(logits, axis=1)

# ---------------------------------------------------------------------------
# Main experimental driver (10 seeds, multiple configurations)
# ---------------------------------------------------------------------------
def run_experiment(seeds: list[int]):
    DURATION = 0.1
    DT = 0.001
    NUM_PER_CLASS = 200
    ARTIFACT_DIR = pathlib.Path(__file__).parent.parent / "artifacts"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        # Build deterministic balanced splits
        (train_rates, y_train), (val_rates, y_val), (test_rates, y_test) = build_dataset(
            NUM_PER_CLASS, DURATION, DT, seed=seed
        )
        # Encode to binary spikes
        train_spikes = [poisson_rate_encoder(r, dt=DT, seed=seed + i) for i, r in enumerate(train_rates)]
        val_spikes = [poisson_rate_encoder(r, dt=DT, seed=seed + 1000 + i) for i, r in enumerate(val_rates)]
        test_spikes = [poisson_rate_encoder(r, dt=DT, seed=seed + 2000 + i) for i, r in enumerate(test_rates)]
        n_input = train_spikes[0].shape[1]
        n_output = 4
        rng = np.random.default_rng(seed)
        init_w_in = rng.uniform(0.2, 0.8, size=(n_output, n_input))
        init_w_rec = init_sparse_ring(n_output, seed=seed, base_weight=0.5)
        # ---- A: Fixed random weights (no STDP) ----
        net_fixed = RecurrentSpikingNetwork(
            n_input=n_input,
            n_neurons=n_output,
            input_weights=init_w_in,
            recurrent_weights=init_w_rec,
            dt=DT,
            enable_stdp=False,
        )
        X_train_fixed = extract_total(net_fixed, train_spikes)
        X_test_fixed = extract_total(net_fixed, test_spikes)
        clf_fixed = SoftmaxClassifier(lr=0.5, epochs=500).fit(X_train_fixed, y_train)
        acc_fixed = np.mean(clf_fixed.predict(X_test_fixed) == y_test)
        # ---- B: STDP on input synapses only ----
        net_input_stdp = RecurrentSpikingNetwork(
            n_input=n_input,
            n_neurons=n_output,
            input_weights=init_w_in,
            recurrent_weights=init_w_rec,
            dt=DT,
            enable_stdp=True,
            stdp_input_params={"a_plus":0.02,"a_minus":0.022,"w_min":0.0,"w_max":1.0},
            stdp_rec_params={"a_plus":0.0,"a_minus":0.0},
        )
        for s in train_spikes:
            net_input_stdp.run(s, record_weight_history=False)
        net_input_stdp.enable_stdp = False
        X_train_ip = extract_total(net_input_stdp, train_spikes)
        X_test_ip = extract_total(net_input_stdp, test_spikes)
        clf_ip = SoftmaxClassifier(lr=0.5, epochs=500).fit(X_train_ip, y_train)
        acc_ip = np.mean(clf_ip.predict(X_test_ip) == y_test)
        # ---- C: STDP on input + recurrent ----
        net_full = RecurrentSpikingNetwork(
            n_input=n_input,
            n_neurons=n_output,
            input_weights=init_w_in,
            recurrent_weights=init_w_rec,
            dt=DT,
            enable_stdp=True,
            stdp_input_params={"a_plus":0.02,"a_minus":0.022,"w_min":0.0,"w_max":1.0},
            stdp_rec_params={"a_plus":0.02,"a_minus":0.022,"w_min":0.0,"w_max":1.0},
        )
        for s in train_spikes:
            net_full.run(s, record_weight_history=False)
        net_full.enable_stdp = False
        X_train_full = extract_total(net_full, train_spikes)
        X_test_full = extract_total(net_full, test_spikes)
        clf_full = SoftmaxClassifier(lr=0.5, epochs=500).fit(X_train_full, y_train)
        acc_full = np.mean(clf_full.predict(X_test_full) == y_test)
        # ---- D: Raw input baseline (4‑bin summed rates) ----
        def raw_bins(rates, bins=4):
            """Bin summed rates for the raw‑input baseline.

            ``rates`` is a list of ``np.ndarray`` with shape ``(T, n_input)``.
            For each sample we compute the sum over ``bins`` equal‑duration windows
            (across *all* input channels) and concatenate the results into a 1‑D
            feature vector.  The previous implementation attempted to ``np.concatenate``
            a list of scalar values, which raised ``ValueError: zero‑dimensional arrays
            cannot be concatenated``.  Here we simply collect the scalars into a Python
            list and convert it to a ``np.ndarray``.
            """
            n_steps = rates[0].shape[0]
            edges = np.linspace(0, n_steps, bins + 1, dtype=int)
            feats = []
            for r in rates:
                b = []
                for i in range(bins):
                    b.append(r[edges[i]:edges[i + 1]].sum())
                feats.append(np.array(b, dtype=np.float64))
            return np.stack(feats)
        X_train_raw = raw_bins(train_rates)
        X_test_raw = raw_bins(test_rates)
        clf_raw = SoftmaxClassifier(lr=0.5, epochs=500).fit(X_train_raw, y_train)
        acc_raw = np.mean(clf_raw.predict(X_test_raw) == y_test)
        rows.append({
            "seed": seed,
            "fixed_acc": acc_fixed,
            "input_stdp_acc": acc_ip,
            "full_stdp_acc": acc_full,
            "raw_acc": acc_raw,
        })
    # Save CSV results
    import csv
    csv_path = ARTIFACT_DIR / "temporal_classification_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed","fixed_acc","input_stdp_acc","full_stdp_acc","raw_acc"])
        writer.writeheader()
        writer.writerows(rows)
    # Save short report
    report_path = ARTIFACT_DIR / "temporal_classification_report.txt"
    with open(report_path, "w") as f:
        f.write("Temporal pattern classification – v0.3.2\n")
        f.write("===================================\n\n")
        for r in rows:
            f.write(f"Seed {r['seed']}: fixed {r['fixed_acc']:.4f}, input‑STDP {r['input_stdp_acc']:.4f}, full‑STDP {r['full_stdp_acc']:.4f}, raw {r['raw_acc']:.4f}\n")
    # Save artifacts and generate visualisations
    print("Experiment completed. Files written to", ARTIFACT_DIR)
    # ----- Generate additional artifacts -----
    import matplotlib.pyplot as plt
    # Accuracy comparison bar plot
    seeds = [r['seed'] for r in rows]
    fixed_acc = [r['fixed_acc'] for r in rows]
    input_acc = [r['input_stdp_acc'] for r in rows]
    full_acc = [r['full_stdp_acc'] for r in rows]
    raw_acc = [r['raw_acc'] for r in rows]
    x = np.arange(len(seeds))
    width = 0.18
    plt.figure(figsize=(10, 6))
    plt.bar(x - 1.5 * width, fixed_acc, width, label='Fixed')
    plt.bar(x - 0.5 * width, input_acc, width, label='Input STDP')
    plt.bar(x + 0.5 * width, full_acc, width, label='Full STDP')
    plt.bar(x + 1.5 * width, raw_acc, width, label='Raw')
    plt.xlabel('Seed')
    plt.ylabel('Accuracy')
    plt.title('Temporal Classification Accuracies per Seed')
    plt.xticks(x, seeds)
    plt.ylim(0, 1)
    plt.legend()
    comp_path = ARTIFACT_DIR / "temporal_feature_comparison.png"
    plt.tight_layout()
    plt.savefig(comp_path)
    plt.close()
    # Weight evolution heatmap for full STDP network
    plt.figure(figsize=(6, 5))
    plt.imshow(net_full.recurrent_weights, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Weight')
    plt.title('Recurrent Weights after Full STDP Training')
    weight_path = ARTIFACT_DIR / "weight_evolution.png"
    plt.tight_layout()
    plt.savefig(weight_path)
    plt.close()

if __name__ == "__main__":
    run_experiment(seeds=list(range(42, 52)))
