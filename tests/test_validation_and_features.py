# -*- coding: utf-8 -*-
"""Regression tests for numerical input validation and temporal feature extraction safety."""

from __future__ import annotations

import numpy as np
import pytest

from neurochip_ar.encoding import poisson_rate_encoder
from neurochip_ar.network import FeedForwardNetwork
from neurochip_ar.neuron import LIFNeuron
from neurochip_ar.recurrent import RecurrentSpikingNetwork
from neurochip_ar.stdp import PairSTDP
from neurochip_ar.temporal_features import extract_bins, extract_latency, extract_total, raw_bins


@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
def test_lif_rejects_non_finite_parameters_and_current(bad_val: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        LIFNeuron(v_rest=bad_val)
    with pytest.raises(ValueError, match="finite"):
        LIFNeuron(tau_m=bad_val)

    neuron = LIFNeuron()
    with pytest.raises(ValueError, match="finite"):
        neuron.step(bad_val)


@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
def test_encoder_rejects_non_finite_and_empty_inputs(bad_val: float) -> None:
    with pytest.raises(ValueError):
        poisson_rate_encoder([0.5], duration=0.1, dt=bad_val)
    with pytest.raises(ValueError):
        poisson_rate_encoder([0.5], duration=0.1, r_max=bad_val)
    with pytest.raises(ValueError):
        poisson_rate_encoder([bad_val], duration=0.1)
    with pytest.raises(ValueError, match="empty"):
        poisson_rate_encoder(np.empty((0, 2)))


def test_feedforward_network_rejects_non_binary_and_non_finite_spikes() -> None:
    net = FeedForwardNetwork(n_input=2, n_output=2, weight_matrix=[[0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(ValueError, match="binary"):
        net.step([0.5, 1.0])
    with pytest.raises(ValueError, match="finite"):
        net.step([np.nan, 0.0])
    with pytest.raises(ValueError, match="binary"):
        net.run(np.array([[1.0, 2.0], [0.0, 1.0]]))

    with pytest.raises(ValueError, match="finite"):
        FeedForwardNetwork(n_input=2, n_output=2, weight_matrix=[[np.nan, 0.5], [0.5, 0.5]])


def test_stdp_rejects_non_finite_and_non_binary_inputs() -> None:
    stdp = PairSTDP(n_input=2, n_output=2)
    valid_w = np.full((2, 2), 0.5)
    with pytest.raises(ValueError, match="finite"):
        stdp.step(np.array([[np.nan, 0.5], [0.5, 0.5]]), [1, 0], [0, 1])
    with pytest.raises(ValueError, match="binary"):
        stdp.step(valid_w, [0.5, 0.0], [0, 1])
    with pytest.raises(ValueError, match="binary"):
        stdp.apply_train(valid_w, np.array([[1, 0], [2, 0]]), np.array([[0, 1], [0, 0]]))


def test_temporal_features_reject_stdp_enabled_network() -> None:
    net = RecurrentSpikingNetwork(
        n_input=2,
        n_neurons=2,
        input_weights=[[0.5, 0.5], [0.5, 0.5]],
        recurrent_weights=[[0.0, 0.2], [0.2, 0.0]],
        enable_stdp=True,
        stdp_input_params={"a_plus": 0.01, "a_minus": 0.01},
    )
    spikes = np.ones((10, 2), dtype=int)
    for fn in (extract_total, extract_bins, extract_latency):
        with pytest.raises(ValueError, match="enable_stdp=False"):
            fn(net, spikes)


def test_temporal_features_validation_edges() -> None:
    spikes = np.ones((6, 2), dtype=int)
    with pytest.raises(ValueError, match="positive integer"):
        extract_bins(spikes, bins=True)
    with pytest.raises(ValueError, match="positive integer"):
        raw_bins([np.ones((6, 2))], bins=True)
    with pytest.raises(ValueError, match="positive integer"):
        extract_bins(spikes, bins=0)
    with pytest.raises(ValueError, match="binary"):
        extract_total(np.full((4, 2), 0.5))
    with pytest.raises(ValueError, match="finite"):
        raw_bins([np.array([[1.0, np.nan], [0.0, 1.0]])], bins=2)

    lat_ms = extract_latency(np.array([[0, 1], [1, 0], [0, 0]]), dt_ms=2.0)
    np.testing.assert_allclose(lat_ms, np.array([2.0, 0.0]))
