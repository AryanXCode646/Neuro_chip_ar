# Tests for STDP default bounds handling in RecurrentSpikingNetwork

import numpy as np
import pytest

from neurochip_ar.recurrent import RecurrentSpikingNetwork


def make_weights(shape, value):
    return np.full(shape, value, dtype=np.float64)


def test_initial_weights_within_default_bounds():
    # Default bounds are [0.0, 1.0]
    input_w = make_weights((2, 2), 0.5)
    rec_w = np.zeros((2, 2))
    # Should construct without error
    net = RecurrentSpikingNetwork(
        n_input=2,
        n_neurons=2,
        input_weights=input_w,
        recurrent_weights=rec_w,
        enable_stdp=True,
    )
    assert net.input_weights.shape == (2, 2)
    assert np.all(net.input_weights == 0.5)


def test_initial_weights_above_default_max():
    input_w = make_weights((2, 2), 1.5)  # above default max 1.0
    rec_w = np.zeros((2, 2))
    with pytest.raises(ValueError, match="outside configured STDP bounds"):
        RecurrentSpikingNetwork(
            n_input=2,
            n_neurons=2,
            input_weights=input_w,
            recurrent_weights=rec_w,
            enable_stdp=True,
        )


def test_initial_weights_below_default_min():
    input_w = make_weights((2, 2), -0.1)  # below default min 0.0
    rec_w = np.zeros((2, 2))
    with pytest.raises(ValueError, match="outside configured STDP bounds"):
        RecurrentSpikingNetwork(
            n_input=2,
            n_neurons=2,
            input_weights=input_w,
            recurrent_weights=rec_w,
            enable_stdp=True,
        )


def test_custom_bounds_allowed_range():
    input_w = make_weights((2, 2), 1.5)
    rec_w = make_weights((2, 2), 1.5)
    np.fill_diagonal(rec_w, 0.0)
    net = RecurrentSpikingNetwork(
        n_input=2,
        n_neurons=2,
        input_weights=input_w,
        recurrent_weights=rec_w,
        enable_stdp=True,
        stdp_input_params={"w_min": 0.0, "w_max": 2.0},
        stdp_rec_params={"w_min": 0.0, "w_max": 2.0},
    )
    assert np.all(net.input_weights == 1.5)
    assert np.all(net.recurrent_weights == rec_w)


def test_custom_bounds_out_of_range():
    input_w = make_weights((2, 2), 2.5)
    rec_w = np.zeros((2, 2))
    with pytest.raises(ValueError, match="outside configured STDP bounds"):
        RecurrentSpikingNetwork(
            n_input=2,
            n_neurons=2,
            input_weights=input_w,
            recurrent_weights=rec_w,
            enable_stdp=True,
            stdp_input_params={"w_min": 0.0, "w_max": 2.0},
        )


def test_stdp_disabled_allows_any_weights():
    # STDP disabled should not enforce bounds, but diagonal must be zero per network requirement
    input_w = make_weights((2, 2), 5.0)
    rec_w = make_weights((2, 2), 5.0)
    np.fill_diagonal(rec_w, 0.0)
    net = RecurrentSpikingNetwork(
        n_input=2,
        n_neurons=2,
        input_weights=input_w,
        recurrent_weights=rec_w,
        enable_stdp=False,
    )
    assert np.all(net.input_weights == 5.0)
    assert np.all(net.recurrent_weights == rec_w)
