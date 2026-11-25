#!/usr/bin/env python3
"""Temporal pattern classification experiment.

- Balanced dataset (200 samples per class)
- Stratified 70/15/15 train/val/test split with seed-controlled training shuffle
- Sparse ring recurrent connectivity
- Online PairSTDP on input and recurrent synapses (train split only)
- Feature extractors: total spikes, binned counts, first-spike latency
- Raw-input baseline using time-binned rates
- Validation-based readout hyperparameter selection and leak-free test evaluation
- Results and figures saved to ``artifacts/`` (or a caller-specified ``output_dir``)
"""

from __future__ import annotations

import csv
import importlib.metadata
import math
import os
import pathlib
from collections.abc import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neurochip_ar.constants import ENCODER_DT_SECONDS, NETWORK_DT_MS
from neurochip_ar.encoding import poisson_rate_encoder
from neurochip_ar.recurrent import RecurrentSpikingNetwork
from neurochip_ar.temporal_features import extract_bins, extract_latency, extract_total, raw_bins

DURATION = 0.1
DT = ENCODER_DT_SECONDS
NUM_PER_CLASS = 200
SEED = 0
N_BINS = 4
NEURON_PARAMS = {
    "r_membrane": 30.0,
    "tau_m": 15.0,
    "refractory_period": 2.0,
}

# Stable integer namespaces for independent SeedSequence derivation.
_NS_DATASET_PATTERN = 1
_NS_DATASET_CLASS_SHUFFLE = 2
_NS_DATASET_TRAIN_SHUFFLE = 3
_NS_ENCODER = 4
_NS_INIT_W_IN = 5
_NS_INIT_W_REC = 6

_SPLIT_TRAIN = 0
_SPLIT_VAL = 1
_SPLIT_TEST = 2


def _derive_seed_sequence(root_seed: int, *keys: int) -> np.random.SeedSequence:
    """Derive deterministic, independently keyed NumPy random streams from *root_seed* and integer namespaces."""
    return np.random.SeedSequence([int(root_seed), *(int(k) for k in keys)])


def _validate_int_non_bool(value: object, name: str) -> int:
    """Validate that *value* is a non-boolean integer and return it as ``int``."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return int(value)


def _validate_positive_finite_float(value: object, name: str) -> float:
    """Validate that *value* is a finite, strictly positive real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    val = float(value)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value}")
    return val


def generate_pattern(duration: float, dt: float, seed: int | np.random.SeedSequence, first_channel: int) -> np.ndarray:
    """Create a two-channel pattern where *first_channel* is high then low."""
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
) -> tuple[
    tuple[list[np.ndarray], np.ndarray],
    tuple[list[np.ndarray], np.ndarray],
    tuple[list[np.ndarray], np.ndarray],
]:
    """Return stratified (train, val, test) splits with balanced classes and shuffled training order."""
    num_per_class = _validate_int_non_bool(num_per_class, "num_per_class")
    if num_per_class <= 0:
        raise ValueError(f"num_per_class must be positive, got {num_per_class}")
    duration = _validate_positive_finite_float(duration, "duration")
    dt = _validate_positive_finite_float(dt, "dt")
    seed = _validate_int_non_bool(seed, "seed")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    n_steps = int(round(duration / dt))
    if n_steps < 2:
        raise ValueError(f"duration / dt must yield at least 2 timesteps, got {n_steps}")

    train_n = int(0.70 * num_per_class)
    val_n = int(0.15 * num_per_class)
    test_n = num_per_class - train_n - val_n
    if train_n <= 0 or val_n <= 0 or test_n <= 0:
        raise ValueError(
            f"num_per_class={num_per_class} is too small for 70/15/15 split "
            f"(train={train_n}, val={val_n}, test={test_n})"
        )

    class_patterns: dict[int, list[np.ndarray]] = {0: [], 1: []}
    for cls in (0, 1):
        for i in range(num_per_class):
            first = 0 if cls == 0 else 1
            pat_seed = _derive_seed_sequence(seed, _NS_DATASET_PATTERN, cls, i)
            pat = generate_pattern(duration, dt, seed=pat_seed, first_channel=first)
            class_patterns[cls].append(pat)
        cls_rng = np.random.default_rng(_derive_seed_sequence(seed, _NS_DATASET_CLASS_SHUFFLE, cls))
        cls_rng.shuffle(class_patterns[cls])

    splits: dict[str, list] = {
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
        splits["test_labels"].extend([cls] * test_n)

    # Shuffle combined training split deterministically to avoid class-block STDP bias.
    train_shuffle_rng = np.random.default_rng(_derive_seed_sequence(seed, _NS_DATASET_TRAIN_SHUFFLE))
    train_perm = train_shuffle_rng.permutation(len(splits["train"]))
    train_shuffled = [splits["train"][idx] for idx in train_perm]
    train_labels_shuffled = np.asarray(splits["train_labels"], dtype=int)[train_perm]

    return (
        (train_shuffled, train_labels_shuffled),
        (splits["val"], np.asarray(splits["val_labels"], dtype=int)),
        (splits["test"], np.asarray(splits["test_labels"], dtype=int)),
    )


def init_sparse_ring(n_neurons: int, seed: int | np.random.SeedSequence, base_weight: float = 0.4) -> np.ndarray:
    """Create a directed ring connectivity matrix (no self-connections)."""
    n_neurons = _validate_int_non_bool(n_neurons, "n_neurons")
    if n_neurons < 2:
        raise ValueError(f"n_neurons must be at least 2 for a directed ring without self-connections, got {n_neurons}")
    if isinstance(base_weight, bool) or not isinstance(base_weight, (int, float, np.number)):
        raise TypeError(f"base_weight must be a real number, got {type(base_weight).__name__}")
    weight = float(base_weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(f"base_weight must be finite and non-negative, got {base_weight}")

    rng = np.random.default_rng(seed)
    w = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    for i in range(n_neurons):
        j = (i + 1) % n_neurons
        w[i, j] = weight * rng.uniform(0.8, 1.2)
    return w


class SoftmaxClassifier:
    """Multinomial logistic regression with optional L2 regularization using NumPy."""

    def __init__(self, lr: float = 0.1, epochs: int = 400, l2: float = 0.0):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> SoftmaxClassifier:
        n_samples, n_features = X.shape
        n_classes = int(np.max(y)) + 1
        self.W = np.zeros((n_classes, n_features), dtype=np.float64)
        self.b = np.zeros(n_classes, dtype=np.float64)
        for _ in range(self.epochs):
            logits = X @ self.W.T + self.b
            probs = self._softmax(logits)
            y_onehot = np.zeros_like(probs)
            y_onehot[np.arange(n_samples), y] = 1.0
            grad = (probs - y_onehot) / n_samples
            self.W -= self.lr * ((grad.T @ X) + self.l2 * self.W)
            self.b -= self.lr * grad.sum(axis=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.W.T + self.b
        return np.argmax(logits, axis=1)


def _standardize(X_fit: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    """Standardize features using ``X_fit`` statistics only."""
    mean = X_fit.mean(axis=0, keepdims=True)
    std = X_fit.std(axis=0, keepdims=True)
    safe_std = np.where(std > 1e-8, std, 1.0)
    transformed = [(X_fit - mean) / safe_std]
    for X in others:
        transformed.append((X - mean) / safe_std)
    return tuple(transformed)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    """Compute accuracy, balanced accuracy, and binary confusion counts."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc = float(np.mean(y_true == y_pred))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = 0.5 * (tpr + tnr)
    return {
        "accuracy": acc,
        "balanced_accuracy": float(bal_acc),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _select_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, float | int], float, float, np.ndarray]:
    """Select readout hyperparameters on validation data, refit on train+val, evaluate on test."""
    X_tr_s, X_val_s = _standardize(X_train, X_val)
    best_val_acc = -1.0
    best_lr = 0.1
    best_l2 = 0.0
    for lr in (0.05, 0.2, 0.5):
        for l2 in (0.0, 1e-3, 1e-2):
            clf = SoftmaxClassifier(lr=lr, epochs=400, l2=l2).fit(X_tr_s, y_train)
            val_acc = float(np.mean(clf.predict(X_val_s) == y_val))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_lr = lr
                best_l2 = l2

    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    X_tv_s, X_te_s = _standardize(X_trainval, X_test)
    final_clf = SoftmaxClassifier(lr=best_lr, epochs=400, l2=best_l2).fit(X_tv_s, y_trainval)
    y_pred = final_clf.predict(X_te_s)
    metrics = _compute_metrics(y_test, y_pred)
    return metrics, best_lr, best_l2, y_pred


def _train_stdp_network(
    net: RecurrentSpikingNetwork,
    train_spikes: list[np.ndarray],
    init_w_in: np.ndarray,
    init_w_rec: np.ndarray,
) -> tuple[int, float, float, bool, list[tuple[float, float]]]:
    """Run online STDP over training samples with explicit sample-state resets."""
    total_spikes = 0
    trajectory: list[tuple[float, float]] = []
    for s in train_spikes:
        res = net.run(s, record_weight_history=False)
        total_spikes += int(res["spikes"].sum())
        net.reset()
        dw_in = float(np.linalg.norm(net.input_weights - init_w_in))
        dw_rec = float(np.linalg.norm(net.recurrent_weights - init_w_rec))
        trajectory.append((dw_in, dw_rec))

    net.enable_stdp = False
    net.reset()
    final_dw_in = float(np.linalg.norm(net.input_weights - init_w_in))
    final_dw_rec = float(np.linalg.norm(net.recurrent_weights - init_w_rec))
    stdp_updated = (final_dw_in > 0.0) or (final_dw_rec > 0.0)
    return total_spikes, final_dw_in, final_dw_rec, stdp_updated, trajectory


def _package_version() -> str:
    """Return the installed ``neurochip-ar`` package version if available."""
    try:
        return importlib.metadata.version("neurochip-ar")
    except importlib.metadata.PackageNotFoundError:
        return "0.2.0"


def run_experiment(
    seeds: Iterable[int],
    output_dir: pathlib.Path | str | None = None,
) -> list[dict[str, object]]:
    """Run the temporal pattern classification experiment and save reproducible artifacts."""
    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Iterable):
        raise TypeError(f"seeds must be an iterable of integers, got {type(seeds).__name__}")
    validated_seeds: list[int] = []
    for s in seeds:
        si = _validate_int_non_bool(s, "seed")
        if si < 0:
            raise ValueError(f"seed must be non-negative, got {si}")
        validated_seeds.append(si)
    if not validated_seeds:
        raise ValueError("seeds must not be empty")
    if len(set(validated_seeds)) != len(validated_seeds):
        raise ValueError("seeds must not contain duplicate values")

    if output_dir is not None:
        if not isinstance(output_dir, (str, os.PathLike)):
            raise TypeError(f"output_dir must be a path or string, got {type(output_dir).__name__}")
        if isinstance(output_dir, str) and not output_dir.strip():
            raise ValueError("output_dir must not be empty")
        artifact_dir = pathlib.Path(output_dir)
        if artifact_dir.exists() and not artifact_dir.is_dir():
            raise ValueError(f"output_dir exists and is not a directory: {artifact_dir}")
    else:
        artifact_dir = pathlib.Path(__file__).parent.parent / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    combos = [
        ("fixed", "total"),
        ("fixed", "bins"),
        ("fixed", "latency"),
        ("input_stdp", "total"),
        ("input_stdp", "bins"),
        ("input_stdp", "latency"),
        ("full_stdp", "total"),
        ("full_stdp", "bins"),
        ("full_stdp", "latency"),
        ("raw", "raw_bins"),
    ]

    rows: list[dict[str, object]] = []
    full_stdp_trajectories: list[list[tuple[float, float]]] = []
    agg_cms: dict[str, np.ndarray] = {f"{m} / {feat}": np.zeros((2, 2), dtype=int) for m, feat in combos}

    for seed in validated_seeds:
        (train_rates, y_train), (val_rates, y_val), (test_rates, y_test) = build_dataset(
            NUM_PER_CLASS, DURATION, DT, seed=seed
        )
        train_spikes = [
            poisson_rate_encoder(r, dt=DT, seed=_derive_seed_sequence(seed, _NS_ENCODER, _SPLIT_TRAIN, i))
            for i, r in enumerate(train_rates)
        ]
        val_spikes = [
            poisson_rate_encoder(r, dt=DT, seed=_derive_seed_sequence(seed, _NS_ENCODER, _SPLIT_VAL, i))
            for i, r in enumerate(val_rates)
        ]
        test_spikes = [
            poisson_rate_encoder(r, dt=DT, seed=_derive_seed_sequence(seed, _NS_ENCODER, _SPLIT_TEST, i))
            for i, r in enumerate(test_rates)
        ]

        n_input = train_spikes[0].shape[1]
        n_output = 4
        w_in_rng = np.random.default_rng(_derive_seed_sequence(seed, _NS_INIT_W_IN))
        init_w_in = w_in_rng.uniform(0.2, 0.8, size=(n_output, n_input))
        init_w_rec = init_sparse_ring(n_output, seed=_derive_seed_sequence(seed, _NS_INIT_W_REC))

        net_fixed = RecurrentSpikingNetwork(
            n_input=n_input,
            n_neurons=n_output,
            input_weights=init_w_in,
            recurrent_weights=init_w_rec,
            dt=NETWORK_DT_MS,
            neuron_params=NEURON_PARAMS,
            enable_stdp=False,
        )
        net_input = RecurrentSpikingNetwork(
            n_input=n_input,
            n_neurons=n_output,
            input_weights=init_w_in,
            recurrent_weights=init_w_rec,
            dt=NETWORK_DT_MS,
            neuron_params=NEURON_PARAMS,
            enable_stdp=True,
            stdp_input_params={"a_plus": 0.02, "a_minus": 0.022, "w_min": 0.0, "w_max": 1.0},
            stdp_rec_params={"a_plus": 0.0, "a_minus": 0.0},
        )
        net_full = RecurrentSpikingNetwork(
            n_input=n_input,
            n_neurons=n_output,
            input_weights=init_w_in,
            recurrent_weights=init_w_rec,
            dt=NETWORK_DT_MS,
            neuron_params=NEURON_PARAMS,
            enable_stdp=True,
            stdp_input_params={"a_plus": 0.02, "a_minus": 0.022, "w_min": 0.0, "w_max": 1.0},
            stdp_rec_params={"a_plus": 0.02, "a_minus": 0.022, "w_min": 0.0, "w_max": 1.0},
        )

        # Count spikes during fixed baseline pass for fair diagnostic comparison.
        fixed_spikes = 0
        for s in train_spikes:
            fixed_spikes += int(net_fixed.run(s, record_weight_history=False)["spikes"].sum())
            net_fixed.reset()

        in_spikes, in_dw_in, in_dw_rec, in_updated, _ = _train_stdp_network(
            net_input, train_spikes, init_w_in, init_w_rec
        )
        full_spikes, full_dw_in, full_dw_rec, full_updated, full_traj = _train_stdp_network(
            net_full, train_spikes, init_w_in, init_w_rec
        )
        full_stdp_trajectories.append(full_traj)

        models = [
            ("fixed", net_fixed, fixed_spikes, 0.0, 0.0, False),
            ("input_stdp", net_input, in_spikes, in_dw_in, in_dw_rec, in_updated),
            ("full_stdp", net_full, full_spikes, full_dw_in, full_dw_rec, full_updated),
        ]
        extractors = [
            ("total", lambda net, spk: extract_total(net, spk)),
            ("bins", lambda net, spk: extract_bins(net, spk, bins=N_BINS)),
            ("latency", lambda net, spk: extract_latency(net, spk, dt_ms=NETWORK_DT_MS)),
        ]

        for model_name, net, spk_cnt, dw_in, dw_rec, updated in models:
            for feat_name, fn in extractors:
                X_tr = fn(net, train_spikes)
                X_va = fn(net, val_spikes)
                X_te = fn(net, test_spikes)
                metrics, best_lr, best_l2, _ = _select_and_evaluate(X_tr, y_train, X_va, y_val, X_te, y_test)
                rows.append(
                    {
                        "seed": seed,
                        "model": model_name,
                        "feature": feat_name,
                        "accuracy": metrics["accuracy"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "tp": metrics["tp"],
                        "fp": metrics["fp"],
                        "fn": metrics["fn"],
                        "tn": metrics["tn"],
                        "best_lr": best_lr,
                        "best_l2": best_l2,
                        "delta_w_in_norm": dw_in,
                        "delta_w_rec_norm": dw_rec,
                        "train_output_spikes": spk_cnt,
                        "stdp_updated": updated,
                    }
                )
                key = f"{model_name} / {feat_name}"
                if key in agg_cms:
                    agg_cms[key] += np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])

        # Raw input baseline
        X_tr_raw = raw_bins(train_rates, bins=N_BINS)
        X_va_raw = raw_bins(val_rates, bins=N_BINS)
        X_te_raw = raw_bins(test_rates, bins=N_BINS)
        raw_metrics, raw_lr, raw_l2, _ = _select_and_evaluate(X_tr_raw, y_train, X_va_raw, y_val, X_te_raw, y_test)
        rows.append(
            {
                "seed": seed,
                "model": "raw",
                "feature": "raw_bins",
                "accuracy": raw_metrics["accuracy"],
                "balanced_accuracy": raw_metrics["balanced_accuracy"],
                "tp": raw_metrics["tp"],
                "fp": raw_metrics["fp"],
                "fn": raw_metrics["fn"],
                "tn": raw_metrics["tn"],
                "best_lr": raw_lr,
                "best_l2": raw_l2,
                "delta_w_in_norm": 0.0,
                "delta_w_rec_norm": 0.0,
                "train_output_spikes": 0,
                "stdp_updated": False,
            }
        )
        agg_cms["raw / raw_bins"] += np.array(
            [[raw_metrics["tn"], raw_metrics["fp"]], [raw_metrics["fn"], raw_metrics["tp"]]]
        )

    fieldnames = [
        "seed",
        "model",
        "feature",
        "accuracy",
        "balanced_accuracy",
        "tp",
        "fp",
        "fn",
        "tn",
        "best_lr",
        "best_l2",
        "delta_w_in_norm",
        "delta_w_rec_norm",
        "train_output_spikes",
        "stdp_updated",
    ]
    csv_path = artifact_dir / "temporal_classification_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate summary report
    report_path = artifact_dir / "temporal_classification_report.txt"
    train_n = int(0.70 * NUM_PER_CLASS)
    val_n = int(0.15 * NUM_PER_CLASS)
    test_n = NUM_PER_CLASS - train_n - val_n
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Temporal Pattern Classification Report\n")
        f.write(f"Seeds: {validated_seeds}\n")
        f.write("Chance baseline (balanced binary task): 0.5000\n\n")
        f.write("Provenance:\n")
        f.write(f"  Package version: neurochip-ar {_package_version()}\n")
        f.write(f"  Experiment seed set: {validated_seeds}\n")
        f.write(
            "  Dataset-generation strategy: independent numpy.random.SeedSequence streams "
            "keyed by (root_seed, namespace, [split/class, sample_idx])\n"
        )
        f.write(
            f"  Dataset size & class balance: {2 * NUM_PER_CLASS} total "
            f"({NUM_PER_CLASS} class 0, {NUM_PER_CLASS} class 1)\n"
        )
        f.write(
            f"  Split sizes (stratified 70/15/15): train={2 * train_n} ({train_n}/class), "
            f"val={2 * val_n} ({val_n}/class), test={2 * test_n} ({test_n}/class)\n"
        )
        f.write(f"  Encoder parameters: poisson_rate_encoder(dt={DT}s, duration={DURATION}s, r_max=100.0Hz)\n")
        f.write(f"  Neuron parameters: dt={NETWORK_DT_MS}ms, {NEURON_PARAMS}\n")
        f.write(f"  Feature representations: total, bins (n_bins={N_BINS}), latency, raw_bins (n_bins={N_BINS})\n")
        f.write(
            "  Network configurations: n_input=2, n_output=4 (sparse ring), "
            "fixed (STDP off), input_stdp (a_plus=0.02, a_minus=0.022, w=[0,1]), "
            "full_stdp (input+rec a_plus=0.02, a_minus=0.022, w=[0,1])\n"
        )
        f.write(
            "  Readout protocol: SoftmaxClassifier (400 epochs), grid search over "
            "lr in (0.05, 0.2, 0.5) and l2 in (0.0, 1e-3, 1e-2) on train/val "
            "(standardized on train), refit on train+val, single evaluation on test\n\n"
        )
        f.write("Aggregate Test Performance (mean ± sample std):\n")
        for m, feat in combos:
            sub = [r for r in rows if r["model"] == m and r["feature"] == feat]
            accs = np.array([float(r["accuracy"]) for r in sub])
            bals = np.array([float(r["balanced_accuracy"]) for r in sub])
            ddof = 1 if len(accs) > 1 else 0
            f.write(
                f"  {m:10s} | {feat:8s} | acc={accs.mean():.4f} ± {accs.std(ddof=ddof):.4f} "
                f"| bal_acc={bals.mean():.4f} ± {bals.std(ddof=ddof):.4f}\n"
            )
        f.write("\nSTDP & Spiking Diagnostics (per seed):\n")
        for r in rows:
            if r["feature"] in ("total", "raw_bins"):
                f.write(
                    f"  seed={r['seed']} {r['model']:10s} | spikes={r['train_output_spikes']} "
                    f"| dW_in={float(r['delta_w_in_norm']):.4f} | dW_rec={float(r['delta_w_rec_norm']):.4f} "
                    f"| updated={r['stdp_updated']}\n"
                )

    # 1. Genuine temporal feature comparison plot
    fig, ax = plt.subplots(figsize=(10, 5.5))
    snn_models = ["fixed", "input_stdp", "full_stdp"]
    features = ["total", "bins", "latency"]
    x = np.arange(len(snn_models))
    width = 0.25
    for idx, feat in enumerate(features):
        means = []
        stds = []
        for m in snn_models:
            vals = np.array([float(r["accuracy"]) for r in rows if r["model"] == m and r["feature"] == feat])
            means.append(float(vals.mean()))
            stds.append(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
        ax.bar(x + (idx - 1) * width, means, width, yerr=stds, capsize=4, label=feat)
    raw_vals = np.array([float(r["accuracy"]) for r in rows if r["model"] == "raw"])
    ax.axhline(float(raw_vals.mean()), color="black", linestyle="--", label=f"raw_bins ({raw_vals.mean():.3f})")
    ax.axhline(0.5, color="gray", linestyle=":", label="chance (0.500)")
    ax.set_xticks(x)
    ax.set_xticklabels(snn_models)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Temporal Feature Representation Comparison Across SNN Models")
    ax.legend()
    fig.tight_layout()
    fig.savefig(artifact_dir / "temporal_feature_comparison.png")
    plt.close(fig)

    # 2. Genuine weight evolution across training samples
    traj_arr = np.array(full_stdp_trajectories, dtype=np.float64)  # (n_seeds, n_train, 2)
    samples_axis = np.arange(1, traj_arr.shape[1] + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(samples_axis, traj_arr[:, :, 0].mean(axis=0), label=r"$\|\Delta W_{\mathrm{in}}\|_F$ (mean)")
    ax.plot(samples_axis, traj_arr[:, :, 1].mean(axis=0), label=r"$\|\Delta W_{\mathrm{rec}}\|_F$ (mean)")
    ax.set_xlabel("Training Sample Index")
    ax.set_ylabel("Frobenius Weight Change Norm")
    ax.set_title("Online STDP Weight Evolution During Training (Full STDP)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(artifact_dir / "weight_evolution.png")
    plt.close(fig)

    # 3. Confusion matrices aggregated across test predictions (all 10 model/feature combinations)
    fig, axes = plt.subplots(4, 3, figsize=(11, 13), constrained_layout=True)
    flat_axes = axes.ravel()
    im = None
    for idx, (label, cm) in enumerate(agg_cms.items()):
        ax = flat_axes[idx]
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(label)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        vmax = cm.max() if cm.size else 0
        for i in range(2):
            for j in range(2):
                color = "white" if vmax > 0 and cm[i, j] > 0.6 * vmax else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
    for idx in range(len(agg_cms), len(flat_axes)):
        flat_axes[idx].axis("off")
    if im is not None:
        fig.colorbar(im, ax=flat_axes[: len(agg_cms)].tolist(), shrink=0.6)
    fig.suptitle("Aggregated Test Confusion Matrices (All 10 Model / Feature Combinations)")
    fig.savefig(artifact_dir / "confusion_matrix.png")
    plt.close(fig)

    print("Experiment completed. Files written to", artifact_dir)
    return rows


if __name__ == "__main__":
    run_experiment(seeds=list(range(42, 52)))
