"""Tests for FeedForwardNetwork."""

import numpy as np
import pytest

from neurochip_ar.network import FeedForwardNetwork


def test_network_weight_dimensions_validation():
    """Verify shape mismatch checks between weight matrix and layer sizes."""
    with pytest.raises(ValueError, match="weight_matrix shape mismatch"):
        # n_output=2, n_input=3, but matrix is (3, 2)
        FeedForwardNetwork(n_input=3, n_output=2, weight_matrix=np.ones((3, 2)))

    with pytest.raises(ValueError, match="n_input must be positive"):
        FeedForwardNetwork(n_input=0, n_output=2, weight_matrix=np.ones((2, 0)))

    with pytest.raises(ValueError, match="tau_syn must be positive"):
        FeedForwardNetwork(n_input=2, n_output=2, weight_matrix=np.eye(2), tau_syn=-1.0)


def test_network_synaptic_current_decay():
    """Verify that a single input pulse causes a jump in synaptic current that decays exponentially."""
    n_in, n_out = 2, 2
    weights = np.array([[10.0, 0.0], [0.0, 5.0]])
    net = FeedForwardNetwork(n_input=n_in, n_output=n_out, weight_matrix=weights, tau_syn=10.0, dt=1.0)

    # Deliver one spike at input 0 at t=0
    spikes_t0 = np.array([1, 0])
    net.step(spikes_t0)
    assert net.i_syn[0] == 10.0
    assert net.i_syn[1] == 0.0

    # Next steps with no input should decay exponentially: I(t) = I(0) * exp(-1/10)
    decay = np.exp(-1.0 / 10.0)
    for step in range(1, 5):
        net.step(np.array([0, 0]))
        expected_current = 10.0 * (decay**step)
        assert abs(net.i_syn[0] - expected_current) < 1e-6


def test_network_spike_propagation():
    """Verify that strong synaptic weights cause output neurons to fire."""
    n_in, n_out = 2, 2
    # Strong weight to neuron 0, weak weight to neuron 1
    weights = np.array([[50.0, 50.0], [0.01, 0.01]])
    net = FeedForwardNetwork(
        n_input=n_in,
        n_output=n_out,
        weight_matrix=weights,
        tau_syn=5.0,
        dt=1.0,
        neuron_params={"v_rest": -65.0, "v_threshold": -50.0},
    )

    # Provide sustained spikes
    inputs = np.ones((10, 2), dtype=np.int32)
    res = net.run(inputs)

    assert np.sum(res["spikes"][:, 0]) > 0, "Output neuron 0 should fire with strong input"
    assert np.sum(res["spikes"][:, 1]) == 0, "Output neuron 1 should not fire with weak input"


def test_network_reset():
    """Verify that reset restores all output neurons and synaptic currents to zero."""
    weights = np.array([[20.0, 20.0]])
    net = FeedForwardNetwork(n_input=2, n_output=1, weight_matrix=weights)
    net.step(np.array([1, 1]))

    assert net.i_syn[0] > 0.0
    assert net.neurons[0].v != -65.0

    net.reset()
    assert net.i_syn[0] == 0.0
    assert net.neurons[0].v == -65.0


def test_network_deterministic_run():
    """Verify that two identical runs produce identical outputs."""
    weights = np.array([[12.0, 5.0], [3.0, 15.0]])
    inputs = np.zeros((30, 2), dtype=np.int32)
    inputs[5, 0] = 1
    inputs[10, 1] = 1
    inputs[15, :] = 1

    net1 = FeedForwardNetwork(n_input=2, n_output=2, weight_matrix=weights)
    res1 = net1.run(inputs)

    net2 = FeedForwardNetwork(n_input=2, n_output=2, weight_matrix=weights)
    res2 = net2.run(inputs)

    np.testing.assert_array_equal(res1["spikes"], res2["spikes"])
    np.testing.assert_allclose(res1["voltages"], res2["voltages"])
    np.testing.assert_allclose(res1["synaptic_currents"], res2["synaptic_currents"])
