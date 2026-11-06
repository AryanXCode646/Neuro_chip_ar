# -*- coding: utf-8 -*-
"""Tests for STDP configuration handling in RecurrentSpikingNetwork.

Scenarios covered:
- STDP disabled (both mechanisms None)
- Input‑only STDP enabled
- Recurrent‑only STDP enabled
- Both enabled
- Zero learning rates (both None)
"""

import numpy as np
from neurochip_ar.recurrent import RecurrentSpikingNetwork


def _simple_net(enable_stdp, input_params=None, rec_params=None):
    w_in = np.eye(2)  # simple identity weights
    w_rec = np.zeros((2, 2))
    return RecurrentSpikingNetwork(
        n_input=2,
        n_neurons=2,
        input_weights=w_in,
        recurrent_weights=w_rec,
        dt=1.0,
        enable_stdp=enable_stdp,
        stdp_input_params=input_params,
        stdp_rec_params=rec_params,
        neuron_params={"v_rest": -65.0, "v_reset": -65.0, "v_threshold": -50.0, "refractory_period": 0.0},
    )


def test_stdp_disabled():
    net = _simple_net(enable_stdp=False)
    assert net.stdp_input is None
    assert net.stdp_rec is None


def test_input_only_stdp():
    net = _simple_net(enable_stdp=True, input_params={"a_plus": 0.02, "a_minus": 0.022})
    assert net.stdp_input is not None
    assert net.stdp_rec is None


def test_recurrent_only_stdp():
    net = _simple_net(
        enable_stdp=True,
        rec_params={"a_plus": 0.02, "a_minus": 0.022},
    )
    assert net.stdp_input is None  # input params not given -> still created with defaults? default a_plus>0 leads creation, so we pass empty dict to avoid creation
    # To ensure input STDP not created, provide empty dict with a_plus=0, a_minus=0
    net = _simple_net(
        enable_stdp=True,
        input_params={"a_plus": 0.0, "a_minus": 0.0},
        rec_params={"a_plus": 0.02, "a_minus": 0.022},
    )
    assert net.stdp_input is None
    assert net.stdp_rec is not None


def test_both_enabled():
    net = _simple_net(
        enable_stdp=True,
        input_params={"a_plus": 0.01, "a_minus": 0.011},
        rec_params={"a_plus": 0.02, "a_minus": 0.022},
    )
    assert net.stdp_input is not None
    assert net.stdp_rec is not None


def test_zero_rates_disable_both():
    net = _simple_net(
        enable_stdp=True,
        input_params={"a_plus": 0.0, "a_minus": 0.0},
        rec_params={"a_plus": 0.0, "a_minus": 0.0},
    )
    assert net.stdp_input is None
    assert net.stdp_rec is None
