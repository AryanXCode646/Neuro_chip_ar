"""Tests for RecurrentSpikingNetwork and online STDP."""

from __future__ import annotations

import numpy as np
import pytest

from neurochip_ar.recurrent import RecurrentSpikingNetwork


def test_recurrent_dimension_and_parameter_validation():
    """Ensure invalid dimensions and non-positive parameters are rejected."""
    # Mismatched input_weights shape
    with pytest.raises(ValueError, match="input_weights shape mismatch"):
        RecurrentSpikingNetwork(
            n_input=2,
            n_neurons=2,
            input_weights=np.zeros((1, 2)),
            recurrent_weights=np.zeros((2, 2)),
        )

    # Mismatched recurrent_weights shape
    with pytest.raises(ValueError, match="recurrent_weights shape mismatch"):
        RecurrentSpikingNetwork(
            n_input=2,
            n_neurons=2,
            input_weights=np.zeros((2, 2)),
            recurrent_weights=np.zeros((2, 3)),
        )

    # Invalid n_input or n_neurons
    with pytest.raises(ValueError, match="n_input must be positive"):
        RecurrentSpikingNetwork(
            n_input=0,
            n_neurons=2,
            input_weights=np.zeros((2, 0)),
            recurrent_weights=np.zeros((2, 2)),
        )
    with pytest.raises(ValueError, match="n_neurons must be positive"):
        RecurrentSpikingNetwork(
            n_input=2,
            n_neurons=0,
            input_weights=np.zeros((0, 2)),
            recurrent_weights=np.zeros((0, 0)),
        )

    # Invalid tau_syn or dt
    with pytest.raises(ValueError, match="tau_syn must be positive"):
        RecurrentSpikingNetwork(
            n_input=1,
            n_neurons=1,
            input_weights=np.zeros((1, 1)),
            recurrent_weights=np.zeros((1, 1)),
            tau_syn=-1.0,
        )
    with pytest.raises(ValueError, match="dt must be positive"):
        RecurrentSpikingNetwork(
            n_input=1,
            n_neurons=1,
            input_weights=np.zeros((1, 1)),
            recurrent_weights=np.zeros((1, 1)),
            dt=0.0,
        )


def test_recurrent_finite_and_self_connection_validation():
    """Verify NaN/Inf rejection and strict prohibition of self-connections and negative weights."""
    # NaN in input weights
    with pytest.raises(ValueError, match="finite"):
        RecurrentSpikingNetwork(
            n_input=1,
            n_neurons=1,
            input_weights=np.array([[np.nan]]),
            recurrent_weights=np.array([[0.0]]),
        )

    # Inf in recurrent weights
    with pytest.raises(ValueError, match="finite"):
        RecurrentSpikingNetwork(
            n_input=1,
            n_neurons=1,
            input_weights=np.array([[1.0]]),
            recurrent_weights=np.array([[np.inf]]),
        )

    # Non-zero diagonal (self-connections)
    with pytest.raises(ValueError, match="Self-connections are disallowed"):
        RecurrentSpikingNetwork(
            n_input=2,
            n_neurons=2,
            input_weights=np.zeros((2, 2)),
            recurrent_weights=np.array([[0.5, 0.0], [0.0, 0.0]]),
        )

    # Negative recurrent weight
    with pytest.raises(ValueError, match="Recurrent weights must be non-negative"):
        RecurrentSpikingNetwork(
            n_input=2,
            n_neurons=2,
            input_weights=np.zeros((2, 2)),
            recurrent_weights=np.array([[0.0, -0.5], [0.0, 0.0]]),
        )


def test_explicit_one_step_delay():
    """Verify that output spikes at time t affect recurrent inputs strictly at t+1, not t."""
    # 1 input, 2 recurrent neurons
    # Input drives only neuron 0 strongly at t=0
    # W_rec connects neuron 0 -> neuron 1
    w_in = np.array([[320.0], [0.0]])
    w_rec = np.array([[0.0, 0.0], [320.0, 0.0]])  # neuron 0 -> neuron 1

    net = RecurrentSpikingNetwork(
        n_input=1,
        n_neurons=2,
        input_weights=w_in,
        recurrent_weights=w_rec,
        tau_syn=1.0,
        dt=1.0,
        neuron_params={"v_rest": -65.0, "v_reset": -65.0, "v_threshold": -50.0, "refractory_period": 0.0},
    )

    # Step t=0: input spike to neuron 0
    spikes_t0, v_t0 = net.step(np.array([1.0]))
    # Neuron 0 spikes at t=0
    assert spikes_t0[0] == 1
    # Neuron 1 must NOT receive recurrent input at t=0 (delay is strictly 1 timestep)
    assert spikes_t0[1] == 0
    assert v_t0[1] == -65.0

    # Step t=1: no external input
    spikes_t1, v_t1 = net.step(np.array([0.0]))
    # Recurrent spike from neuron 0 now reaches neuron 1 at t=1, triggering a spike
    assert spikes_t1[0] == 0
    assert spikes_t1[1] == 1
    assert v_t1[1] == -65.0  # reset after spike


def test_recurrent_feedback_changes_network_activity():
    """Confirm that recurrent connections alter subsequent network trajectory compared to disconnected."""
    w_in = np.array([[320.0], [0.0]])
    w_rec_connected = np.array([[0.0, 0.0], [320.0, 0.0]])
    w_rec_disconnected = np.zeros((2, 2))

    net_conn = RecurrentSpikingNetwork(
        n_input=1,
        n_neurons=2,
        input_weights=w_in,
        recurrent_weights=w_rec_connected,
        tau_syn=1.0,
        dt=1.0,
        neuron_params={"refractory_period": 0.0},
    )
    net_disconn = RecurrentSpikingNetwork(
        n_input=1,
        n_neurons=2,
        input_weights=w_in,
        recurrent_weights=w_rec_disconnected,
        tau_syn=1.0,
        dt=1.0,
        neuron_params={"refractory_period": 0.0},
    )

    inputs = np.array([[1.0], [0.0], [0.0]])

    res_conn = net_conn.run(inputs)
    res_disconn = net_disconn.run(inputs)

    # In connected network, neuron 1 spikes at t=1 due to recurrent excitation from neuron 0
    assert res_conn["spikes"][1, 1] == 1
    # In disconnected network, neuron 1 never spikes
    assert res_disconn["spikes"][1, 1] == 0
    assert np.all(res_disconn["spikes"][:, 1] == 0)


def test_recurrent_reset_preserves_learned_weights():
    """Reset must clear neuron potentials, currents, and previous spikes but preserve weights."""
    w_in = np.array([[0.4], [0.6]])
    w_rec = np.array([[0.0, 0.5], [0.5, 0.0]])

    net = RecurrentSpikingNetwork(
        n_input=1,
        n_neurons=2,
        input_weights=w_in,
        recurrent_weights=w_rec,
        enable_stdp=True,
        stdp_input_params={"w_max": 2.0},
        stdp_rec_params={"w_max": 2.0},
    )

    # Modify weights to simulate learned state
    net.input_weights += 0.2
    net.recurrent_weights += 0.1
    np.fill_diagonal(net.recurrent_weights, 0.0)
    saved_w_in = np.copy(net.input_weights)
    saved_w_rec = np.copy(net.recurrent_weights)

    # Run one step to introduce state
    net.step(np.array([1.0]))
    assert np.any(net.i_syn != 0.0)

    # Reset
    net.reset()
    assert np.all(net.i_syn == 0.0)
    assert np.all(net.y_prev == 0.0)
    for neuron in net.neurons:
        assert neuron.v == neuron.v_rest

    # Learned weights must remain intact
    np.testing.assert_array_equal(net.input_weights, saved_w_in)
    np.testing.assert_array_equal(net.recurrent_weights, saved_w_rec)


def test_online_stdp_updates_and_affects_subsequent_steps():
    """Demonstrate that online STDP updates weights during simulation and affects later dynamics."""
    w_in = np.array([[350.0]])
    w_rec = np.array([[0.0]])

    # STDP enabled network
    net_plastic = RecurrentSpikingNetwork(
        n_input=1,
        n_neurons=1,
        input_weights=w_in,
        recurrent_weights=w_rec,
        tau_syn=1.0,
        dt=1.0,
        enable_stdp=True,
        stdp_input_params={"a_plus": 10.0, "a_minus": 5.0, "w_max": 500.0},
        neuron_params={"v_rest": -65.0, "v_reset": -65.0, "v_threshold": -50.0, "refractory_period": 0.0},
    )

    # STDP disabled network (control)
    net_static = RecurrentSpikingNetwork(
        n_input=1,
        n_neurons=1,
        input_weights=w_in,
        recurrent_weights=w_rec,
        tau_syn=1.0,
        dt=1.0,
        enable_stdp=False,
        neuron_params={"v_rest": -65.0, "v_reset": -65.0, "v_threshold": -50.0, "refractory_period": 0.0},
    )

    inputs = np.array([[1.0], [1.0], [1.0]])

    res_plastic = net_plastic.run(inputs, record_weight_history=True)
    res_static = net_static.run(inputs)

    # Weights must have changed in plastic network during simulation
    assert not np.allclose(net_plastic.input_weights, w_in)
    assert np.allclose(net_static.input_weights, w_in)

    # Check weight history was recorded
    assert "input_weights_history" in res_plastic
    assert res_plastic["input_weights_history"].shape == (3, 1, 1)

    # The updated weights must produce a difference in recorded synaptic currents
    assert not np.allclose(res_plastic["synaptic_currents"], res_static["synaptic_currents"])


def test_recurrent_online_stdp_pre_before_post_potentiation():
    """Verify that recurrent connections undergo potentiation when pre fires before post."""
    # 2 neurons: neuron 0 drives neuron 1 via recurrent connection
    w_in = np.array([[350.0], [0.0]])  # external input fires neuron 0
    w_rec = np.array([[0.0, 0.0], [1.0, 0.0]])  # neuron 0 -> neuron 1

    net = RecurrentSpikingNetwork(
        n_input=1,
        n_neurons=2,
        input_weights=w_in,
        recurrent_weights=w_rec,
        tau_syn=1.0,
        dt=1.0,
        enable_stdp=True,
        stdp_input_params={"w_max": 500.0},
        stdp_rec_params={"w_max": 10.0},
        neuron_params={"v_rest": -65.0, "v_reset": -65.0, "v_threshold": -50.0, "refractory_period": 0.0},
    )

    # t=0: input fires neuron 0 -> y[0] = [1, 0]
    net.step(np.array([1.0]))
    # t=1: delayed recurrent spike arrives (y_prev = [1, 0]), charging pre-trace
    net.step(np.array([0.0]))
    # t=2: neuron 1 fires (post-spike)
    net.neurons[1].v = -40.0
    initial_w_rec_01 = net.recurrent_weights[1, 0]
    net.step(np.array([0.0]))

    # Recurrent weight from 0 -> 1 should experience potentiation
    assert net.recurrent_weights[1, 0] > initial_w_rec_01
    # Diagonal must remain strictly zero
    assert net.recurrent_weights[0, 0] == 0.0
    assert net.recurrent_weights[1, 1] == 0.0


def test_recurrent_input_validation():
    """Test rejection of non-binary inputs and invalid dimensions."""
    net = RecurrentSpikingNetwork(
        n_input=2,
        n_neurons=2,
        input_weights=np.zeros((2, 2)),
        recurrent_weights=np.zeros((2, 2)),
    )

    # Non-binary step input
    with pytest.raises(ValueError, match="binary indicators"):
        net.step(np.array([0.5, 1.0]))

    # Wrong shape step input
    with pytest.raises(ValueError, match="input_spikes shape mismatch"):
        net.step(np.array([1.0]))

    # Non-binary run input
    with pytest.raises(ValueError, match="binary indicators"):
        net.run(np.array([[0.0, 2.0]]))

    # Wrong dimension run input
    with pytest.raises(ValueError, match="input_spikes must be 2D"):
        net.run(np.array([1.0, 0.0]))


def test_recurrent_network_determinism():
    """Identical inputs must yield bit-for-bit identical trajectories."""
    w_in = np.array([[0.5, 0.2], [0.1, 0.4]])
    w_rec = np.array([[0.0, 0.3], [0.3, 0.0]])

    net1 = RecurrentSpikingNetwork(2, 2, w_in, w_rec, enable_stdp=True)
    net2 = RecurrentSpikingNetwork(2, 2, w_in, w_rec, enable_stdp=True)

    inputs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])

    res1 = net1.run(inputs)
    res2 = net2.run(inputs)

    np.testing.assert_array_equal(res1["spikes"], res2["spikes"])
    np.testing.assert_allclose(res1["voltages"], res2["voltages"])
    np.testing.assert_allclose(net1.input_weights, net2.input_weights)
    np.testing.assert_allclose(net1.recurrent_weights, net2.recurrent_weights)
