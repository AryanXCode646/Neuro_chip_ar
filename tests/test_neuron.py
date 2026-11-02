"""Tests for LIF neuron implementation."""

import pytest

from neurochip_ar.neuron import LIFNeuron


def test_neuron_invalid_parameters():
    """Verify parameter validation for LIFNeuron."""
    with pytest.raises(ValueError, match="dt must be positive"):
        LIFNeuron(dt=0.0)

    with pytest.raises(ValueError, match="tau_m must be positive"):
        LIFNeuron(tau_m=-5.0)

    with pytest.raises(ValueError, match="r_membrane must be positive"):
        LIFNeuron(r_membrane=0.0)

    with pytest.raises(ValueError, match="refractory_period must be non-negative"):
        LIFNeuron(refractory_period=-1.0)

    with pytest.raises(ValueError, match="v_threshold .* strictly greater than v_reset"):
        LIFNeuron(v_reset=-50.0, v_threshold=-50.0)


def test_neuron_subthreshold_integration():
    """Verify subthreshold integration approaches steady state without spiking."""
    neuron = LIFNeuron(v_rest=-65.0, v_threshold=-50.0, r_membrane=1.0, tau_m=20.0, dt=1.0)
    # Inject current that brings V to -60 mV (subthreshold: -65 + 1.0 * 5.0 = -60.0)
    spiked = False
    for _ in range(100):
        if neuron.step(5.0):
            spiked = True

    assert not spiked
    # After 100 ms (5 * tau), V should be close to -60.0 mV
    assert abs(neuron.v - (-60.0)) < 0.1


def test_neuron_threshold_and_reset():
    """Verify that crossing threshold generates a spike and resets voltage."""
    neuron = LIFNeuron(
        v_rest=-65.0,
        v_reset=-70.0,
        v_threshold=-50.0,
        r_membrane=1.0,
        tau_m=10.0,
        dt=1.0,
        refractory_period=5.0,
    )

    # Inject strong current to trigger a spike
    spikes = []
    voltages = []
    for _ in range(25):
        s = neuron.step(25.0)
        spikes.append(s)
        voltages.append(neuron.v)

    assert any(spikes), "Neuron should have fired under strong injected current"
    first_spike_idx = spikes.index(True)
    # At the step where spike occurs, V is reset to v_reset
    assert voltages[first_spike_idx] == -70.0


def test_neuron_refractory_period():
    """Verify that during refractory period, neuron does not fire and voltage remains clamped."""
    refractory_ms = 4.0
    dt = 1.0
    neuron = LIFNeuron(
        v_rest=-65.0,
        v_reset=-70.0,
        v_threshold=-50.0,
        r_membrane=1.0,
        tau_m=5.0,
        dt=dt,
        refractory_period=refractory_ms,
    )

    # Force immediate spike with large current
    assert neuron.step(100.0) is True
    assert neuron.v == -70.0
    assert neuron.refractory_time_left == refractory_ms

    # During next steps of refractory period, even with huge current, no spike
    for _ in range(int(refractory_ms / dt)):
        spiked = neuron.step(100.0)
        assert not spiked, "Neuron must not fire during refractory period"
        assert neuron.v == -70.0, "Voltage must remain clamped at v_reset during refractory"

    assert neuron.refractory_time_left == 0.0


def test_neuron_reset():
    """Verify that reset restores initial resting state."""
    neuron = LIFNeuron(v_rest=-65.0, dt=1.0)
    neuron.step(10.0)
    assert neuron.v != -65.0

    neuron.reset()
    assert neuron.v == -65.0
    assert neuron.refractory_time_left == 0.0
