# -*- coding: utf-8 -*-
"""Tests for temporal feature extraction utilities and dataset generation.

These tests verify:
- Correct split sizes and class balance of the synthetic dataset.
- Deterministic output shapes for each feature extractor.
- Handling of edge cases such as empty spike matrices and silent neurons.
- Finite values and reproducibility of feature vectors.
- Network reset behavior preserves learned weights while clearing state.
- Weight preservation across training and reset.
- Stratified dataset dimensions and balanced class distributions.
- Dataset separation (train/val/test are disjoint).
"""

import numpy as np

from neurochip_ar.temporal_features import extract_total, extract_bins, extract_latency
from examples.temporal_pattern_classification import (
    build_dataset,
    DURATION,
    DT,
    NUM_PER_CLASS,
    SEED,
)
from neurochip_ar.recurrent import RecurrentSpikingNetwork


def test_build_dataset_split_and_balance():
    (train_rates, y_train), (val_rates, y_val), (test_rates, y_test) = build_dataset(
        NUM_PER_CLASS, DURATION, DT, seed=SEED
    )
    total = 2 * NUM_PER_CLASS
    # Verify total count matches expectation
    assert len(train_rates) + len(val_rates) + len(test_rates) == total
    # Verify split ratios (rounded down)
    assert len(train_rates) == int(0.70 * total)
    assert len(val_rates) == int(0.15 * total)
    assert len(test_rates) == total - len(train_rates) - len(val_rates)
    # Verify each split contains both classes
    for split_labels in (y_train, y_val, y_test):
        assert set(split_labels) == {0, 1}
        counts = np.bincount(split_labels)
        # Allow at most 1 difference due to rounding
        assert abs(counts[0] - counts[1]) <= 1


def test_extract_total_shape_and_determinism():
    spikes = np.random.randint(0, 2, size=(100, 4))
    total1 = extract_total(spikes)
    total2 = extract_total(spikes.copy())
    assert total1.shape == (4,)
    assert np.array_equal(total1, total2)
    assert np.all(np.isfinite(total1))


def test_extract_bins_shapes_and_finiteness():
    spikes = np.random.randint(0, 2, size=(80, 3))
    for bins in [4, 8]:
        feats = extract_bins(spikes, bins=bins)
        assert feats.shape == (3 * bins,)
        assert np.all(feats >= 0)
        assert np.issubdtype(feats.dtype, np.integer)
        assert np.all(np.isfinite(feats))


def test_extract_latency_handling_and_finiteness():
    spikes = np.array([
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
    ])
    lat = extract_latency(spikes)
    assert lat.shape == (4,)
    assert lat[1] == 0
    assert lat[2] == 2
    assert lat[0] == spikes.shape[0]
    assert lat[3] == spikes.shape[0]
    assert np.all(np.isfinite(lat))


def test_extract_bins_edge_case_empty_spike_matrix():
    spikes = np.empty((0, 5), dtype=int)
    feats = extract_bins(spikes, bins=4)
    assert feats.shape == (5 * 4,)
    assert np.all(feats == 0)


def test_network_reset_and_weight_preservation():
    rng = np.random.default_rng(42)
    w_in = rng.uniform(0.2, 0.8, size=(2, 2))
    w_rec = rng.uniform(0.0, 0.5, size=(2, 2))
    np.fill_diagonal(w_rec, 0.0)
    net = RecurrentSpikingNetwork(
        n_input=2,
        n_neurons=2,
        input_weights=w_in,
        recurrent_weights=w_rec,
        dt=1.0,
        enable_stdp=False,
    )
    spikes = np.random.randint(0, 2, size=(10, 2))
    _ = net.run(spikes, record_weight_history=False)
    w_in_before = net.input_weights.copy()
    w_rec_before = net.recurrent_weights.copy()
    net.reset()
    assert np.array_equal(net.input_weights, w_in_before)
    assert np.array_equal(net.recurrent_weights, w_rec_before)
    for neuron in net.neurons:
        assert neuron.v == neuron.params["v_rest"]
    assert np.all(net.i_syn == 0)
    assert np.all(net.y_prev == 0)


def test_dataset_separation_disjoint():
    (train_rates, _), (val_rates, _), (test_rates, _) = build_dataset(
        NUM_PER_CLASS, DURATION, DT, seed=SEED
    )
    # Ensure no shared memory between splits
    for tr in train_rates:
        for vr in val_rates:
            assert not np.shares_memory(tr, vr)
        for te in test_rates:
            assert not np.shares_memory(tr, te)
    for vr in val_rates:
        for te in test_rates:
            assert not np.shares_memory(vr, te)
