#!/usr/bin/env python3
"""Temporal pattern classification experiment.

- Balanced dataset (200 samples per class)
- Stratified 70/15/15 train/val/test split
- Sparse ring recurrent connectivity
- Optional STDP on input and recurrent synapses
- Feature extractors: total spikes, binned counts, latency
- Raw‑input baseline using time‑binned rates
- Results saved in `artifacts/`
"""

# Standard library imports
import csv
import pathlib
from typing import List, Tuple

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np

# Project imports
from neurochip_ar.constants import NETWORK_DT_MS
from neurochip_ar.encoding import poisson_rate_encoder
from neurochip_ar.recurrent import RecurrentSpikingNetwork
from neurochip_ar.temporal_features import extract_total, raw_bins

# Constants (also used by tests)
DURATION = 0.1
DT = 0.001
NUM_PER_CLASS = 200
SEED = 0


def generate_pattern(duration: float, dt: float, seed: int, first_channel: int) -> np.ndarray:
    """Create a two‑channel pattern where *first_channel* is high then low.

    Jitter and small Gaussian noise are added for realism.
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
    return np.clip(pattern + jitter + noise, 0.0, 1.0)


def build_dataset(
    num_per_class: int, duration: float, dt: float, seed: int = 0
) -> Tuple[
    Tuple[List[np.ndarray], np.ndarray],
    Tuple[List[np.ndarray], np.ndarray],
    Tuple[List[np.ndarray], np.ndarray],
]:
    """Return stratified (train, val, test) splits with balanced classes.

    Splits: 70 % train, 15 % validation, 15 % test.
    """
    rng = np.random.default_rng(seed)
    class_patterns = {0: [], 1: []}
    for cls in (0, 1):
        for i in range(num_per_class):
            first = 0 if cls == 0 else 1
            pat = generate_pattern(duration, dt, seed=seed + cls * 1000 + i, first_channel=first)
            class_patterns[cls].append(pat)
    for cls in class_patterns:
        rng.shuffle(class_patterns[cls])
    train_n = int(0.70 * num_per_class)
    val_n = int(0.15 * num_per_class)
    splits = {
        "train": [],
        "val": [],
        "test": [],
        "train_labels": [],
        "val_labels": [],
        "test_labels": [],
    }
    for cls, patterns in class_patterns.items():
        splits["train"].extend(patterns[:train_n])
        splits["val"].extend(patterns[train_n : train_n + val_n])
        splits["test"].extend(patterns[train_n + val_n :])
        splits["train_labels"].extend([cls] * train_n)
        splits["val_labels"].extend([cls] * val_n)
        splits["test_labels"].extend([cls] * (num_per_class - train_n - val_n))
    return (
        (splits["train"], np.array(splits["train_labels"], dtype=int)),
        (splits["val"], np.array(splits["val_labels"], dtype=int)),
        (splits["test"], np.array(splits["test_labels"], dtype=int)),
    )


def init_sparse_ring(n_neurons: int, seed: int, base_weight: float = 0.5) -> np.ndarray:
    """Create a directed ring connectivity matrix (no self‑connections)."""
    rng = np.random.default_rng(seed)
    w = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    for i in range(n_neurons):
        j = (i + 1) % n_neurons
        w[i, j] = base_weight * rng.uniform(0.8, 1.2)
    return w


class SoftmaxClassifier:
    """Simple multinomial logistic regression using NumPy.

    Gradient descent is performed with a fixed learning rate.
    """

    def __init__(self, lr: float = 0.1, epochs: int = 200):
        self.lr = lr
        self.epochs = epochs

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftmaxClassifier":
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.W.T + self.b
        return np.argmax(logits, axis=1)




def run_experiment(seeds: List[int]):
    artifact_dir = pathlib.Path(__file__).parent.parent / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        (train_rates, y_train), (_, _), (test_rates, y_test) = build_dataset(NUM_PER_CLASS, DURATION, DT, seed=seed)
        train_spikes = [poisson_rate_encoder(r, dt=DT, seed=seed + i) for i, r in enumerate(train_rates)]
        test_spikes = [poisson_rate_encoder(r, dt=DT, seed=seed + 2000 + i) for i, r in enumerate(test_rates)]
        n_input = train_spikes[0].shape[1]
        n_output = 4
        rng = np.random.default_rng(seed)
        init_w_in = rng.uniform(0.2, 0.8, size=(n_output, n_input))
        init_w_rec = init_sparse_ring(n_output, seed=seed)
        # Fixed weights (no STDP)
        net_fixed = RecurrentSpikingNetwork(
            n_input=n_input,
            n_neurons=n_output,
            input_weights=init_w_in,
            recurrent_weights=init_w_rec,
            dt=NETWORK_DT_MS,
            enable_stdp=False,
        )
        X_train_fixed = extract_total(net_fixed, train_spikes)
        X_test_fixed = extract_total(net_fixed, test_spikes)
        clf_fixed = SoftmaxClassifier(lr=0.5, epochs=500).fit(X_train_fixed, y_train)
        acc_fixed = np.mean(clf_fixed.predict(X_test_fixed) == y_test)
        # Input‑only STDP
        net_input = RecurrentSpikingNetwork(
            n_input=n_input,
            n_neurons=n_output,
            input_weights=init_w_in,
            recurrent_weights=init_w_rec,
            dt=NETWORK_DT_MS,
            enable_stdp=True,
            stdp_input_params={"a_plus": 0.02, "a_minus": 0.022, "w_min": 0.0, "w_max": 1.0},
            stdp_rec_params={"a_plus": 0.0, "a_minus": 0.0},
        )
        for s in train_spikes:
            net_input.run(s)
            net_input.reset()
        net_input.enable_stdp = False
        net_input.reset()
        acc_input = np.mean(
            SoftmaxClassifier(lr=0.5, epochs=500)
            .fit(extract_total(net_input, train_spikes), y_train)
            .predict(extract_total(net_input, test_spikes))
            == y_test
        )
        # Full STDP (input + recurrent)
        net_full = RecurrentSpikingNetwork(
            n_input=n_input,
            n_neurons=n_output,
            input_weights=init_w_in,
            recurrent_weights=init_w_rec,
            dt=NETWORK_DT_MS,
            enable_stdp=True,
            stdp_input_params={"a_plus": 0.02, "a_minus": 0.022, "w_min": 0.0, "w_max": 1.0},
            stdp_rec_params={"a_plus": 0.02, "a_minus": 0.022, "w_min": 0.0, "w_max": 1.0},
        )
        for s in train_spikes:
            net_full.run(s)
            net_full.reset()
        net_full.enable_stdp = False
        net_full.reset()
        acc_full = np.mean(
            SoftmaxClassifier(lr=0.5, epochs=500)
            .fit(extract_total(net_full, train_spikes), y_train)
            .predict(extract_total(net_full, test_spikes))
            == y_test
        )
        # Raw baseline
        X_train_raw = raw_bins(train_rates)
        X_test_raw = raw_bins(test_rates)
        clf_raw = SoftmaxClassifier(lr=0.5, epochs=500).fit(X_train_raw, y_train)
        acc_raw = np.mean(clf_raw.predict(X_test_raw) == y_test)
        rows.append(
            {
                "seed": seed,
                "fixed_acc": acc_fixed,
                "input_stdp_acc": acc_input,
                "full_stdp_acc": acc_full,
                "raw_acc": acc_raw,
            }
        )
    # Write CSV results
    csv_path = artifact_dir / "temporal_classification_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "fixed_acc", "input_stdp_acc", "full_stdp_acc", "raw_acc"])
        writer.writeheader()
        writer.writerows(rows)
    # Write short report
    report_path = artifact_dir / "temporal_classification_report.txt"
    with open(report_path, "w") as f:
        f.write("Temporal pattern classification – v0.3.2\n\n")
        for r in rows:
            f.write(
                f"Seed {r['seed']}: fixed {r['fixed_acc']:.4f}, input‑STDP {r['input_stdp_acc']:.4f}, "
                f"full‑STDP {r['full_stdp_acc']:.4f}, raw {r['raw_acc']:.4f}\n"
            )
    # Plot accuracies per seed
    seeds = [r["seed"] for r in rows]
    width = 0.18
    x = np.arange(len(seeds))
    plt.figure(figsize=(10, 6))
    plt.bar(x - 1.5 * width, [r["fixed_acc"] for r in rows], width, label="Fixed")
    plt.bar(x - 0.5 * width, [r["input_stdp_acc"] for r in rows], width, label="Input STDP")
    plt.bar(x + 0.5 * width, [r["full_stdp_acc"] for r in rows], width, label="Full STDP")
    plt.bar(x + 1.5 * width, [r["raw_acc"] for r in rows], width, label="Raw")
    plt.xlabel("Seed")
    plt.ylabel("Accuracy")
    plt.title("Temporal Classification Accuracies per Seed")
    plt.xticks(x, seeds)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifact_dir / "temporal_feature_comparison.png")
    plt.close()
    # Visualise final recurrent weights after full STDP training
    plt.figure(figsize=(6, 5))
    plt.imshow(net_full.recurrent_weights, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Weight")
    plt.title("Recurrent Weights after Full STDP Training")
    plt.tight_layout()
    plt.savefig(artifact_dir / "weight_evolution.png")
    plt.close()
    print("Experiment completed. Files written to", artifact_dir)


if __name__ == "__main__":
    run_experiment(seeds=list(range(42, 52)))
