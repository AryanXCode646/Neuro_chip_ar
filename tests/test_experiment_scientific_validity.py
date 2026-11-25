# -*- coding: utf-8 -*-
"""Tests for scientific validity, randomization determinism, and artifacts of the temporal experiment."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

import examples.temporal_pattern_classification as exp
from neurochip_ar.encoding import poisson_rate_encoder


def test_build_dataset_training_shuffle_deterministic_and_mixed() -> None:
    (tr1, y1), (va1, yv1), (te1, yt1) = exp.build_dataset(20, exp.DURATION, exp.DT, seed=7)
    (tr2, y2), (va2, yv2), (te2, yt2) = exp.build_dataset(20, exp.DURATION, exp.DT, seed=7)

    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(yv1, yv2)
    np.testing.assert_array_equal(yt1, yt2)
    for a, b in zip(tr1, tr2):
        np.testing.assert_array_equal(a, b)

    # Ensure training labels are not a contiguous class-0 block followed by class-1 block.
    half = len(y1) // 2
    assert not (np.all(y1[:half] == 0) and np.all(y1[half:] == 1))


def test_experiment_produces_real_stdp_updates_and_artifacts(tmp_path: Path) -> None:
    rows = exp.run_experiment(seeds=[0], output_dir=tmp_path)

    expected_files = [
        "temporal_classification_results.csv",
        "temporal_classification_report.txt",
        "temporal_feature_comparison.png",
        "weight_evolution.png",
        "confusion_matrix.png",
    ]
    for name in expected_files:
        path = tmp_path / name
        assert path.exists(), f"Missing artifact: {name}"
        assert path.stat().st_size > 0, f"Empty artifact: {name}"

    combos = {(r["model"], r["feature"]) for r in rows}
    assert combos == {
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
    }

    in_row = next(r for r in rows if r["model"] == "input_stdp" and r["feature"] == "total")
    assert in_row["stdp_updated"] is True
    assert float(in_row["delta_w_in_norm"]) > 0.0
    assert float(in_row["delta_w_rec_norm"]) == 0.0
    assert int(in_row["train_output_spikes"]) > 0

    full_row = next(r for r in rows if r["model"] == "full_stdp" and r["feature"] == "total")
    assert full_row["stdp_updated"] is True
    assert float(full_row["delta_w_in_norm"]) > 0.0
    assert float(full_row["delta_w_rec_norm"]) > 0.0
    assert int(full_row["train_output_spikes"]) > 0

    with open(tmp_path / "temporal_classification_results.csv", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 10

    report_text = (tmp_path / "temporal_classification_report.txt").read_text(encoding="utf-8")
    assert "0.5000" in report_text
    assert "Provenance:" in report_text


def _dataset_fingerprints(seed: int, num_per_class: int = 40) -> tuple[set[bytes], set[bytes], set[bytes]]:
    (tr, _), (va, _), (te, _) = exp.build_dataset(num_per_class, exp.DURATION, exp.DT, seed=seed)
    return (
        {arr.tobytes() for arr in tr},
        {arr.tobytes() for arr in va},
        {arr.tobytes() for arr in te},
    )


def test_build_dataset_reproducible_for_same_seed() -> None:
    d1 = exp.build_dataset(30, exp.DURATION, exp.DT, seed=42)
    d2 = exp.build_dataset(30, exp.DURATION, exp.DT, seed=42)
    for (p1, y1), (p2, y2) in zip(d1, d2):
        np.testing.assert_array_equal(y1, y2)
        assert len(p1) == len(p2)
        for a, b in zip(p1, p2):
            np.testing.assert_array_equal(a, b)


def test_adjacent_seeds_generate_independent_disjoint_datasets() -> None:
    tr42, va42, te42 = _dataset_fingerprints(42, num_per_class=60)
    tr43, va43, te43 = _dataset_fingerprints(43, num_per_class=60)

    all42 = tr42 | va42 | te42
    all43 = tr43 | va43 | te43
    assert len(all42) == 120
    assert len(all43) == 120
    assert all42.isdisjoint(all43), "Adjacent seeds must not share underlying patterns"


def test_partitions_disjoint_within_experiment() -> None:
    tr, va, te = _dataset_fingerprints(42, num_per_class=60)
    assert tr.isdisjoint(va)
    assert tr.isdisjoint(te)
    assert va.isdisjoint(te)


def test_keyed_encoder_streams_deterministic_and_independent() -> None:
    pat = exp.generate_pattern(exp.DURATION, exp.DT, seed=exp._derive_seed_sequence(42, 1, 0, 0), first_channel=0)

    s42_a = poisson_rate_encoder(pat, dt=exp.DT, seed=exp._derive_seed_sequence(42, exp._NS_ENCODER, 0, 0))
    s42_b = poisson_rate_encoder(pat, dt=exp.DT, seed=exp._derive_seed_sequence(42, exp._NS_ENCODER, 0, 0))
    np.testing.assert_array_equal(s42_a, s42_b)

    s43 = poisson_rate_encoder(pat, dt=exp.DT, seed=exp._derive_seed_sequence(43, exp._NS_ENCODER, 0, 0))
    assert not np.array_equal(s42_a, s43)

    s42_val = poisson_rate_encoder(pat, dt=exp.DT, seed=exp._derive_seed_sequence(42, exp._NS_ENCODER, 1, 0))
    assert not np.array_equal(s42_a, s42_val)


@pytest.mark.parametrize(
    "bad_seeds,exc",
    [
        ([], ValueError),
        ((), ValueError),
        ([42, 42], ValueError),
        ([True], TypeError),
        ([42.5], TypeError),
        (["42"], TypeError),
        ([None], TypeError),
        ("42", TypeError),
        ([-1], ValueError),
    ],
)
def test_run_experiment_rejects_invalid_seeds(tmp_path: Path, bad_seeds, exc) -> None:
    with pytest.raises(exc):
        exp.run_experiment(seeds=bad_seeds, output_dir=tmp_path)


def test_run_experiment_rejects_invalid_output_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        exp.run_experiment(seeds=[0], output_dir="")
    with pytest.raises(ValueError):
        exp.run_experiment(seeds=[0], output_dir="   ")
    with pytest.raises(TypeError):
        exp.run_experiment(seeds=[0], output_dir=123)  # type: ignore[arg-type]

    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        exp.run_experiment(seeds=[0], output_dir=file_path)


@pytest.mark.parametrize(
    "kwargs,exc",
    [
        ({"num_per_class": 0, "duration": 0.1, "dt": 0.001, "seed": 0}, ValueError),
        ({"num_per_class": -5, "duration": 0.1, "dt": 0.001, "seed": 0}, ValueError),
        ({"num_per_class": True, "duration": 0.1, "dt": 0.001, "seed": 0}, TypeError),
        ({"num_per_class": 3, "duration": 0.1, "dt": 0.001, "seed": 0}, ValueError),
        ({"num_per_class": 20, "duration": 0.0, "dt": 0.001, "seed": 0}, ValueError),
        ({"num_per_class": 20, "duration": -0.1, "dt": 0.001, "seed": 0}, ValueError),
        ({"num_per_class": 20, "duration": float("nan"), "dt": 0.001, "seed": 0}, ValueError),
        ({"num_per_class": 20, "duration": float("inf"), "dt": 0.001, "seed": 0}, ValueError),
        ({"num_per_class": 20, "duration": 0.1, "dt": 0.0, "seed": 0}, ValueError),
        ({"num_per_class": 20, "duration": 0.1, "dt": float("nan"), "seed": 0}, ValueError),
        ({"num_per_class": 20, "duration": 0.1, "dt": 0.001, "seed": True}, TypeError),
        ({"num_per_class": 20, "duration": 0.1, "dt": 0.001, "seed": -1}, ValueError),
    ],
)
def test_build_dataset_rejects_invalid_inputs(kwargs, exc) -> None:
    with pytest.raises(exc):
        exp.build_dataset(**kwargs)
