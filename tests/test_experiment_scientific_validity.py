# -*- coding: utf-8 -*-
"""Tests for scientific validity, randomization determinism, and artifacts of the temporal experiment."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import examples.temporal_pattern_classification as exp


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
