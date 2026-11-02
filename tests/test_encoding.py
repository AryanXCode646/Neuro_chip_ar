"""Tests for Poisson rate encoding."""

import numpy as np
import pytest

from neurochip_ar.encoding import poisson_rate_encoder


def test_encoder_dimensions_1d():
    """Verify output shape for 1D input across duration."""
    inputs = np.array([0.2, 0.5, 0.8])
    duration = 0.1  # 100 ms
    dt = 0.001  # 1 ms
    spikes = poisson_rate_encoder(inputs, duration=duration, dt=dt, r_max=100.0, seed=42)

    assert spikes.shape == (100, 3)
    assert spikes.dtype == np.int32
    assert set(np.unique(spikes)).issubset({0, 1})


def test_encoder_dimensions_2d():
    """Verify output shape for 2D time-varying inputs."""
    inputs = np.zeros((50, 4))
    inputs[:, 0] = 0.5
    spikes = poisson_rate_encoder(inputs, dt=0.001, r_max=100.0, seed=123)

    assert spikes.shape == (50, 4)
    # Channel 1, 2, 3 have zero rate -> zero spikes
    assert np.all(spikes[:, 1:] == 0)


def test_encoder_reproducibility():
    """Verify that identical random seeds yield identical spike trains."""
    inputs = np.array([0.1, 0.4, 0.9])
    spikes1 = poisson_rate_encoder(inputs, duration=0.2, dt=0.001, r_max=100.0, seed=999)
    spikes2 = poisson_rate_encoder(inputs, duration=0.2, dt=0.001, r_max=100.0, seed=999)
    spikes3 = poisson_rate_encoder(inputs, duration=0.2, dt=0.001, r_max=100.0, seed=1000)

    assert np.array_equal(spikes1, spikes2)
    assert not np.array_equal(spikes1, spikes3)


def test_encoder_rate_monotonicity():
    """Verify that higher input values yield monotonically higher spike counts on average."""
    inputs = np.array([0.0, 0.2, 0.5, 1.0])
    # Simulate over a longer duration for stable Poisson statistics
    spikes = poisson_rate_encoder(inputs, duration=2.0, dt=0.001, r_max=100.0, seed=42)
    counts = np.sum(spikes, axis=0)

    assert counts[0] == 0, "Zero input should produce zero spikes"
    assert counts[1] < counts[2] < counts[3], "Spike count should increase with input rate"


def test_encoder_invalid_inputs():
    """Verify input validation handles invalid boundaries, NaNs, and negative parameters."""
    with pytest.raises(ValueError, match="Inputs must be normalized"):
        poisson_rate_encoder([-0.1, 0.5], duration=0.1)

    with pytest.raises(ValueError, match="Inputs must be normalized"):
        poisson_rate_encoder([0.5, 1.2], duration=0.1)

    with pytest.raises(ValueError, match="Inputs contains NaN"):
        poisson_rate_encoder([np.nan, 0.5], duration=0.1)

    with pytest.raises(ValueError, match="dt must be positive"):
        poisson_rate_encoder([0.5], duration=0.1, dt=0.0)

    with pytest.raises(ValueError, match="r_max must be positive"):
        poisson_rate_encoder([0.5], duration=0.1, r_max=-10.0)

    with pytest.raises(ValueError, match="duration must be a positive float"):
        poisson_rate_encoder([0.5], duration=None)
