# -*- coding: utf-8 -*-
"""Integration test for time‑unit consistency in the temporal pattern classification experiment.

The test ensures that the encoder uses ``ENCODER_DT_SECONDS`` and the recurrent network
uses ``NETWORK_DT_MS`` (milliseconds), and that the number of simulation steps matches the
specified ``DURATION``.
"""

import numpy as np
import pytest

import examples.temporal_pattern_classification as experiment
from neurochip_ar.constants import ENCODER_DT_SECONDS, NETWORK_DT_MS
from neurochip_ar.recurrent import RecurrentSpikingNetwork


# Helper to capture arguments
class CallRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        # Return a dummy value if needed; for encoder we need a spike array
        if self._original:
            return self._original(*args, **kwargs)
        return np.zeros((1, 1))

    def set_original(self, func):
        self._original = func


def test_experiment_time_units(monkeypatch, tmp_path):
    # Record constructor dt values
    init_dts = []
    orig_init = RecurrentSpikingNetwork.__init__

    def init_wrapper(self, *args, **kwargs):
        init_dts.append(kwargs.get("dt"))
        orig_init(self, *args, **kwargs)

    # Record run calls (network dt and number of steps)
    run_calls = []
    orig_run = RecurrentSpikingNetwork.run

    def run_wrapper(self, input_spikes, *args, **kwargs):
        run_calls.append((self.dt, input_spikes.shape[0]))
        return orig_run(self, input_spikes, *args, **kwargs)

    # Record encoder dt values
    encoder_dts = []
    orig_encoder = experiment.poisson_rate_encoder

    def encoder_wrapper(rate, dt, seed=None):
        encoder_dts.append(dt)
        return orig_encoder(rate, dt=dt, seed=seed)

    # Apply monkeypatches
    monkeypatch.setattr(RecurrentSpikingNetwork, "__init__", init_wrapper, raising=False)
    monkeypatch.setattr(RecurrentSpikingNetwork, "run", run_wrapper, raising=False)
    monkeypatch.setattr(experiment, "poisson_rate_encoder", encoder_wrapper, raising=False)

    # Run the experiment for a single seed to keep runtime small
    experiment.run_experiment(seeds=[0], output_dir=tmp_path)

    # Assertions
    assert init_dts, "Network constructor was not called"
    for dt in init_dts:
        assert dt == NETWORK_DT_MS, f"Network dt {dt} != NETWORK_DT_MS"

    assert encoder_dts, "Encoder was not called"
    for dt in encoder_dts:
        assert dt == ENCODER_DT_SECONDS, f"Encoder dt {dt} != ENCODER_DT_SECONDS"

    assert run_calls, "Network run was not called"
    for net_dt, steps in run_calls:
        # steps * net_dt (ms) -> duration in ms, compare to DURATION (seconds) * 1000
        expected_ms = experiment.DURATION * 1000.0
        actual_ms = steps * net_dt
        assert pytest.approx(actual_ms, rel=1e-6) == expected_ms
