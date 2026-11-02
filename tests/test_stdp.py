"""Tests for Pair-based STDP."""

import numpy as np
import pytest

from neurochip_ar.encoding import poisson_rate_encoder
from neurochip_ar.network import FeedForwardNetwork
from neurochip_ar.stdp import PairSTDP


def test_stdp_invalid_parameters():
    """Verify parameter validations for PairSTDP."""
    with pytest.raises(ValueError, match="a_plus must be positive"):
        PairSTDP(n_input=1, n_output=1, a_plus=-0.01)

    with pytest.raises(ValueError, match="a_minus must be positive"):
        PairSTDP(n_input=1, n_output=1, a_minus=0.0)

    with pytest.raises(ValueError, match="tau_plus must be positive"):
        PairSTDP(n_input=1, n_output=1, tau_plus=-10.0)

    with pytest.raises(ValueError, match="w_max .* strictly greater than w_min"):
        PairSTDP(n_input=1, n_output=1, w_min=0.5, w_max=0.2)


def test_stdp_pre_before_post_potentiation():
    """Verify that pre-before-post timing induces Long-Term Potentiation (LTP)."""
    # 1 input, 1 output
    stdp = PairSTDP(
        n_input=1,
        n_output=1,
        a_plus=0.05,
        a_minus=0.05,
        tau_plus=20.0,
        tau_minus=20.0,
        w_min=0.0,
        w_max=1.0,
        dt=1.0,
    )
    w_initial = np.array([[0.5]])

    # Pre spikes at t = 10 ms, Post spikes at t = 15 ms (delta_t = +5 ms)
    timesteps = 30
    pre_train = np.zeros((timesteps, 1), dtype=np.int32)
    post_train = np.zeros((timesteps, 1), dtype=np.int32)
    pre_train[10, 0] = 1
    post_train[15, 0] = 1

    w_final = stdp.apply_train(w_initial, pre_train, post_train)

    assert w_final[0, 0] > w_initial[0, 0], "Pre-before-post timing must potentiate synaptic weight"
    # Expected LTP increment: a_plus * exp(-5 / 20) = 0.05 * exp(-0.25)
    expected_dw = 0.05 * np.exp(-5.0 / 20.0)
    assert abs((w_final[0, 0] - w_initial[0, 0]) - expected_dw) < 1e-6


def test_stdp_post_before_pre_depression():
    """Verify that post-before-pre timing induces Long-Term Depression (LTD)."""
    stdp = PairSTDP(
        n_input=1,
        n_output=1,
        a_plus=0.05,
        a_minus=0.05,
        tau_plus=20.0,
        tau_minus=20.0,
        w_min=0.0,
        w_max=1.0,
        dt=1.0,
    )
    w_initial = np.array([[0.5]])

    # Post spikes at t = 10 ms, Pre spikes at t = 15 ms (post-before-pre, delta_t = -5 ms)
    timesteps = 30
    pre_train = np.zeros((timesteps, 1), dtype=np.int32)
    post_train = np.zeros((timesteps, 1), dtype=np.int32)
    post_train[10, 0] = 1
    pre_train[15, 0] = 1

    w_final = stdp.apply_train(w_initial, pre_train, post_train)

    assert w_final[0, 0] < w_initial[0, 0], "Post-before-pre timing must depress synaptic weight"
    # Expected LTD decrement: a_minus * exp(-5 / 20) = 0.05 * exp(-0.25)
    expected_dw = -0.05 * np.exp(-5.0 / 20.0)
    assert abs((w_final[0, 0] - w_initial[0, 0]) - expected_dw) < 1e-6


def test_stdp_weight_boundaries():
    """Verify that STDP strictly respects minimum and maximum weight bounds."""
    w_min = 0.1
    w_max = 0.9
    stdp = PairSTDP(n_input=1, n_output=1, a_plus=0.5, a_minus=0.5, w_min=w_min, w_max=w_max)

    # Test upper bound clamping
    w = np.array([[0.8]])
    # 5 successive potentiation pairings
    for _ in range(5):
        w = stdp.step(w, np.array([1]), np.array([0]))
        w = stdp.step(w, np.array([0]), np.array([1]))
    assert w[0, 0] == w_max

    # Test lower bound clamping
    stdp.reset()
    for _ in range(5):
        w = stdp.step(w, np.array([0]), np.array([1]))
        w = stdp.step(w, np.array([1]), np.array([0]))
    assert w[0, 0] == w_min


def test_stdp_simultaneous_spikes():
    """Verify isolated simultaneous spikes at time t with no prior history produce zero update."""
    stdp = PairSTDP(n_input=1, n_output=1, a_plus=0.05, a_minus=0.05)
    w_initial = np.array([[0.5]])

    # Both fire at step 0
    w_new = stdp.step(w_initial, np.array([1]), np.array([1]))
    assert w_new[0, 0] == w_initial[0, 0]


def test_end_to_end_snn_stdp_pipeline():
    """End-to-end integration test: encoding -> SNN simulation -> STDP update."""
    np.random.seed(42)
    n_in = 3
    n_out = 2
    duration_s = 0.2  # 200 ms
    dt_s = 0.001  # 1 ms

    # 1. Poisson rate encoding
    inputs = np.array([0.8, 0.4, 0.1])
    input_spikes = poisson_rate_encoder(inputs, duration=duration_s, dt=dt_s, seed=42)

    # 2. Feed-forward network simulation
    initial_weights = np.array([[20.0, 15.0, 5.0], [5.0, 10.0, 20.0]])
    net = FeedForwardNetwork(
        n_input=n_in,
        n_output=n_out,
        weight_matrix=initial_weights,
        tau_syn=5.0,
        dt=1.0,
        neuron_params={"v_rest": -65.0, "v_threshold": -50.0},
    )
    sim_results = net.run(input_spikes)
    out_spikes = sim_results["spikes"]

    # Verify simulation outputs
    assert out_spikes.shape == (200, n_out)
    assert sim_results["voltages"].shape == (200, n_out)

    # 3. Apply STDP learning
    stdp = PairSTDP(
        n_input=n_in,
        n_output=n_out,
        a_plus=0.01,
        a_minus=0.012,
        w_min=0.0,
        w_max=30.0,
    )
    updated_weights = stdp.apply_train(initial_weights, input_spikes, out_spikes)

    assert updated_weights.shape == initial_weights.shape
    assert np.all(updated_weights >= 0.0)
    assert np.all(updated_weights <= 30.0)
    assert not np.array_equal(updated_weights, initial_weights)
