"""Spike encoding utilities for neuromorphic simulations.

Implements Poisson rate encoding to convert normalized continuous inputs
into discrete binary spike trains.
"""

from __future__ import annotations

import numpy as np


def poisson_rate_encoder(
    inputs: np.ndarray | list[float] | float,
    duration: float | None = None,
    dt: float = 0.001,
    r_max: float = 100.0,
    seed: int | None = None,
) -> np.ndarray:
    """Convert normalized continuous inputs in [0, 1] into binary Poisson spike trains.

    Parameters
    ----------
    inputs : array_like
        Normalized numerical inputs in range [0.0, 1.0].
        Can be:
        - 1D array of shape (n_channels,): static rate per channel across time.
          Requires `duration` to be specified.
        - 2D array of shape (n_timesteps, n_channels): time-varying rates.
    duration : float, optional
        Simulation duration in seconds. Required if `inputs` is 1D. If `inputs`
        is 2D, `duration` is optional and validated against `inputs.shape[0] * dt`.
    dt : float, default 0.001
        Simulation time step in seconds (0.001 s = 1 ms). Must be positive.
    r_max : float, default 100.0
        Maximum firing rate in Hz (corresponding to normalized input = 1.0).
        Must be positive.
    seed : int, optional
        Random seed for deterministic, reproducible spike generation.

    Returns
    -------
    np.ndarray
        Binary spike matrix of shape (n_timesteps, n_channels) with dtype np.int32,
        where 1 indicates a spike event and 0 indicates absence of a spike.
    """
    if dt <= 0.0:
        raise ValueError(f"Simulation timestep dt must be positive, got {dt}")
    if r_max <= 0.0:
        raise ValueError(f"Maximum firing rate r_max must be positive, got {r_max}")

    arr = np.asarray(inputs, dtype=np.float64)

    if np.any(np.isnan(arr)):
        raise ValueError("Inputs contains NaN values")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("Inputs must be normalized in the range [0.0, 1.0]")

    if arr.ndim == 0:
        # Scalar single-channel input
        arr = arr.reshape(1)

    if arr.ndim == 1:
        if duration is None or duration <= 0.0:
            raise ValueError("duration must be a positive float when inputs is 1D")
        n_steps = int(round(duration / dt))
        if n_steps <= 0:
            raise ValueError(f"Calculated n_steps ({n_steps}) must be positive. Check duration and dt.")
        n_channels = arr.shape[0]
        # Tile rates across time steps: shape (n_steps, n_channels)
        rate_matrix = np.tile(arr, (n_steps, 1))
    elif arr.ndim == 2:
        rate_matrix = arr
        n_steps, n_channels = rate_matrix.shape
        if duration is not None:
            expected_steps = int(round(duration / dt))
            if expected_steps != n_steps:
                raise ValueError(
                    f"duration ({duration}s at dt={dt}s -> {expected_steps} steps) does not match "
                    f"inputs shape ({n_steps} steps)"
                )
    else:
        raise ValueError(f"inputs must be 1D or 2D, got shape {arr.shape}")

    # Exact Poisson event probability: p = 1 - exp(-rate * dt) where rate = normalized_input * r_max
    # Use numerically stable np.expm1 for -exp(-x)
    p_spike = -np.expm1(-rate_matrix * (r_max * dt))
    # Ensure probabilities are within [0, 1]
    np.clip(p_spike, 0.0, 1.0, out=p_spike)

    rng = np.random.default_rng(seed)
    random_draws = rng.random(size=p_spike.shape)
    spikes = (random_draws < p_spike).astype(np.int32)

    return spikes
