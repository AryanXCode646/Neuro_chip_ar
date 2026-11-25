# -*- coding: utf-8 -*-
"""Test that STDP weight updates are preserved after reset and that transient state is cleared.

The test creates a simple recurrent network with STDP enabled, runs a deterministic input spike
pattern that triggers weight change, records the input weight matrix, calls ``reset`` and verifies
that the matrix is unchanged while internal state variables are zero. Then a second independent
sample is run and we assert that the network starts from a clean state (no residual currents or
spike history).
"""

import numpy as np

from neurochip_ar.recurrent import RecurrentSpikingNetwork


def _make_net(enable_stdp=True, enable_stdp_rec=False):
    """Construct a minimal recurrent network with optional STDP.

    Parameters
    ----------
    enable_stdp: bool
        Enable STDP on input synapses.
    enable_stdp_rec: bool
        Enable STDP on recurrent synapses.
    """
    # Simple 2‑input, 2‑neuron network with moderate input weights and zero recurrent weights.
    w_in = np.full((2, 2), 0.5)
    w_rec = np.zeros((2, 2))
    enable_any = enable_stdp or enable_stdp_rec
    stdp_input = {"a_plus": 1.0, "a_minus": 0.001, "w_max": 10.0} if enable_stdp else None
    stdp_rec = {"a_plus": 1.0, "a_minus": 0.001, "w_max": 10.0} if enable_stdp_rec else None
    return RecurrentSpikingNetwork(
        n_input=2,
        n_neurons=2,
        input_weights=w_in,
        recurrent_weights=w_rec,
        dt=1.0,
        enable_stdp=enable_any,
        stdp_input_params=stdp_input,
        stdp_rec_params=stdp_rec,
        neuron_params={"v_threshold": -30.0, "r_membrane": 100.0},
    )


def _spike_train(pattern: np.ndarray, repeats: int = 1) -> np.ndarray:
    """Return a binary spike train for *pattern* repeated ``repeats`` times.

    ``pattern`` is a 2‑D array of shape (T, n_input) with values in [0, 1].
    """
    binarized = (pattern > 0.5).astype(np.float64)
    return np.concatenate([binarized] * repeats, axis=0)


def _compare_state(net: RecurrentSpikingNetwork, fresh: RecurrentSpikingNetwork):
    """Assert that the mutable internal state of ``net`` matches a freshly built network.

    This includes synaptic currents, previous spike vector, neuron membrane potentials and
    both input and recurrent STDP traces (if present).
    """
    # Synaptic currents and previous spikes
    assert np.allclose(net.i_syn, fresh.i_syn), "i_syn mismatch after reset"
    assert np.allclose(net.y_prev, fresh.y_prev), "y_prev mismatch after reset"
    # Neuron membrane potentials should be at their reset values
    for n in net.neurons:
        assert np.isclose(n.v, n.v_rest), "Neuron membrane potential not reset"
    # Input STDP traces
    if net.stdp_input is not None:
        assert fresh.stdp_input is not None, "Fresh network missing input STDP"
        assert np.allclose(net.stdp_input.trace_pre, fresh.stdp_input.trace_pre), "Input STDP trace_pre not cleared"
        assert np.allclose(net.stdp_input.trace_post, fresh.stdp_input.trace_post), "Input STDP trace_post not cleared"
    else:
        assert fresh.stdp_input is None, "Fresh network should not have input STDP"
    # Recurrent STDP traces
    if net.stdp_rec is not None:
        assert fresh.stdp_rec is not None, "Fresh network missing recurrent STDP"
        assert np.allclose(net.stdp_rec.trace_pre, fresh.stdp_rec.trace_pre), "Recurrent STDP trace_pre not cleared"
        assert np.allclose(net.stdp_rec.trace_post, fresh.stdp_rec.trace_post), "Recurrent STDP trace_post not cleared"
    else:
        assert fresh.stdp_rec is None, "Fresh network should not have recurrent STDP"


def test_stdp_weight_preserved_and_state_cleared():
    net = _make_net(enable_stdp=True, enable_stdp_rec=False)
    w_initial = net.input_weights.copy()
    pattern = np.ones((200, 2))
    spikes = _spike_train(pattern)
    net.run(spikes, record_weight_history=False)
    assert not np.allclose(net.input_weights, w_initial), "STDP did not modify weights"
    w_after_training = net.input_weights.copy()
    net.reset()
    assert np.allclose(net.input_weights, w_after_training), "Weights changed on reset"
    fresh = _make_net(enable_stdp=True, enable_stdp_rec=False)
    fresh.input_weights = w_initial
    _compare_state(net, fresh)
    zero_spikes = np.zeros_like(spikes)
    net.run(zero_spikes, record_weight_history=False)
    assert np.allclose(net.i_syn, 0.0), "Synaptic currents not zero after zero‑input run"
    assert np.max(net.y_prev) == 0.0, "Residual spikes persisted after reset"


def test_recurrent_stdp_traces_cleared():
    net = _make_net(enable_stdp=False, enable_stdp_rec=True)
    pattern = np.ones((50, 2))
    spikes = _spike_train(pattern)
    net.run(spikes, record_weight_history=False)
    assert (
        net.stdp_rec.trace_pre.sum() != 0.0 or net.stdp_rec.trace_post.sum() != 0.0
    ), "Recurrent STDP traces did not update"
    net.reset()
    assert np.allclose(net.stdp_rec.trace_pre, 0.0), "Recurrent STDP trace_pre not cleared on reset"
    assert np.allclose(net.stdp_rec.trace_post, 0.0), "Recurrent STDP trace_post not cleared on reset"
