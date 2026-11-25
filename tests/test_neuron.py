"""Tests for LIF neuron implementation."""

import math

import numpy as np
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


def test_neuron_analytical_subthreshold_trajectory():
    """Compare simulated subthreshold voltage trajectory against the exact analytical solution."""
    v_rest = -65.0
    r_membrane = 1.5
    tau_m = 20.0
    i_inj = 6.0  # V_inf = -65 + 1.5 * 6 = -56.0 mV < -50.0 mV
    dt = 0.01
    neuron = LIFNeuron(
        v_rest=v_rest,
        v_reset=v_rest,
        v_threshold=-50.0,
        r_membrane=r_membrane,
        tau_m=tau_m,
        dt=dt,
        refractory_period=0.0,
    )
    v_inf = v_rest + r_membrane * i_inj
    n_steps = 2000  # 20.0 ms = 1 * tau_m
    times = np.arange(1, n_steps + 1) * dt
    sim_v = np.empty(n_steps, dtype=np.float64)
    for k in range(n_steps):
        spiked = neuron.step(i_inj)
        assert not spiked
        sim_v[k] = neuron.v

    exact_v = v_inf + (v_rest - v_inf) * np.exp(-times / tau_m)
    np.testing.assert_allclose(sim_v, exact_v, atol=5e-3)


def test_neuron_timestep_halving_convergence():
    """Verify forward Euler error decreases at first-order O(dt) under timestep halving."""
    v_rest = -65.0
    r_membrane = 1.0
    tau_m = 20.0
    i_inj = 10.0  # V_inf = -55.0 mV (subthreshold)
    t_end = 10.0
    v_exact = (v_rest + r_membrane * i_inj) + (-r_membrane * i_inj) * math.exp(-t_end / tau_m)

    errors = []
    for dt in (1.0, 0.5, 0.25, 0.125):
        neuron = LIFNeuron(v_rest=v_rest, r_membrane=r_membrane, tau_m=tau_m, dt=dt)
        n_steps = int(round(t_end / dt))
        for _ in range(n_steps):
            assert not neuron.step(i_inj)
        errors.append(abs(neuron.v - v_exact))

    for earlier, later in zip(errors, errors[1:]):
        assert later < earlier
        # First-order Euler: halving dt should roughly halve the global error.
        assert later / earlier < 0.55


def test_neuron_controlled_threshold_crossing_and_firing_rate():
    """Verify first-spike time and constant-current firing rate against analytical LIF theory."""
    v_rest = -65.0
    v_reset = -65.0
    v_threshold = -50.0
    r_membrane = 1.0
    tau_m = 20.0
    t_ref = 4.0
    i_inj = 20.0  # V_inf = -45.0 mV > -50.0 mV
    dt = 0.05

    neuron = LIFNeuron(
        v_rest=v_rest,
        v_reset=v_reset,
        v_threshold=v_threshold,
        r_membrane=r_membrane,
        tau_m=tau_m,
        dt=dt,
        refractory_period=t_ref,
    )

    v_inf = v_rest + r_membrane * i_inj
    t_cross_exact = -tau_m * math.log((v_inf - v_threshold) / (v_inf - v_rest))
    expected_isi_ms = t_ref + t_cross_exact
    expected_rate_hz = 1000.0 / expected_isi_ms

    duration_ms = 1000.0
    n_steps = int(round(duration_ms / dt))
    spike_times = []
    for step_idx in range(1, n_steps + 1):
        if neuron.step(i_inj):
            spike_times.append(step_idx * dt)

    assert len(spike_times) > 10
    assert abs(spike_times[0] - t_cross_exact) <= 2.0 * dt
    simulated_rate_hz = len(spike_times) / (duration_ms / 1000.0)
    assert abs(simulated_rate_hz - expected_rate_hz) / expected_rate_hz < 0.02
