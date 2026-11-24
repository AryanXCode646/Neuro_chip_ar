# -*- coding: utf-8 -*-
"""Tests for time‑unit consistency across NeuroChip AR modules.

The encoder operates with a timestep expressed in **seconds** (default
``ENCODER_DT_SECONDS`` = 0.001 s).  All simulation components use a timestep
in **milliseconds**.  The relationship must hold:

    NETWORK_DT_MS == ENCODER_DT_SECONDS * 1000

We verify that the defaults of ``FeedForwardNetwork`` and ``RecurrentSpikingNetwork``
match the derived constant.
"""

import numpy as np
import pytest

from neurochip_ar.constants import ENCODER_DT_SECONDS, NETWORK_DT_MS
from neurochip_ar.network import FeedForwardNetwork
from neurochip_ar.recurrent import RecurrentSpikingNetwork


def test_network_dt_consistency():
    # Create a minimal network and inspect its dt attribute
    net = FeedForwardNetwork(
        n_input=1,
        n_output=1,
        weight_matrix=[[1.0]],
    )
    assert net.dt == NETWORK_DT_MS
    # The decay factor should be computed with the same dt
    expected_decay = np.exp(-net.dt / net.tau_syn)
    assert net.decay_factor == pytest.approx(expected_decay)


def test_recurrent_dt_consistency():
    rec = RecurrentSpikingNetwork(
        n_input=1,
        n_neurons=1,
        input_weights=[[1.0]],
        recurrent_weights=[[0.0]],
    )
    assert rec.dt == NETWORK_DT_MS
    expected_decay = np.exp(-rec.dt / rec.tau_syn)
    assert rec.decay_factor == pytest.approx(expected_decay)


def test_encoder_network_dt_match():
    # Verify the conversion constant is correct
    assert NETWORK_DT_MS == pytest.approx(ENCODER_DT_SECONDS * 1000.0)
