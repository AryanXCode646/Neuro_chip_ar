# -*- coding: utf-8 -*-
"""Utility functions for extracting temporal features from recurrent spiking networks.

The functions operate on a :class:`~neurochip_ar.recurrent.RecurrentSpikingNetwork` instance
and a binary spike-train matrix (``np.ndarray`` of shape ``(T, n_input)``) or a non-empty
list of equally shaped spike-train matrices. When a network is provided, STDP must be
disabled (``net.enable_stdp is False``), and the network's transient state is reset before
each sample while preserving learned synaptic weights.

Implemented features:
- ``extract_total`` – total spike count per neuron.
- ``extract_bins`` – concatenated spike counts over ``bins`` equal-duration windows.
- ``extract_latency`` – first-spike latency per neuron in timestep indices (or ms if ``dt_ms``
  is given), using ``T`` (or ``T * dt_ms``) as the silent-neuron sentinel.
- ``raw_bins`` – baseline binned sum of continuous rate matrices preserving channel identity.
"""

from __future__ import annotations

import math

import numpy as np

from neurochip_ar.recurrent import RecurrentSpikingNetwork

__all__ = ["extract_total", "extract_bins", "extract_latency", "raw_bins"]


def _validate_bins(bins: int) -> int:
    """Validate that ``bins`` is a positive integer and not a boolean."""
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or int(bins) <= 0:
        raise ValueError(f"bins must be a positive integer, got {bins!r}")
    return int(bins)


def _validate_spike_matrix(spikes: np.ndarray, index: int | None = None) -> np.ndarray:
    """Validate a single 2D finite binary spike matrix of shape ``(T, n_channels)``."""
    prefix = f"spike matrix at index {index}" if index is not None else "spikes"
    arr = np.asarray(spikes)
    if arr.ndim != 2:
        raise ValueError(f"{prefix} must be a 2D array of shape (n_timesteps, n_channels), got shape {arr.shape}")
    if arr.shape[1] == 0:
        raise ValueError(f"{prefix} must have at least one channel, got shape {arr.shape}")
    arr_f = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr_f)):
        raise ValueError(f"{prefix} must contain only finite numerical values")
    if arr_f.size > 0 and np.any((arr_f != 0.0) & (arr_f != 1.0)):
        raise ValueError(f"{prefix} must contain only binary spike indicators (0 or 1)")
    return arr.astype(np.int64, copy=False)


def _resolve_inputs(
    net_or_spikes: RecurrentSpikingNetwork | np.ndarray | list[np.ndarray],
    spikes: np.ndarray | list[np.ndarray] | None,
) -> tuple[RecurrentSpikingNetwork | None, list[np.ndarray], bool]:
    """Resolve ``(net, spikes_list, single_sample)`` and enforce STDP-disabled safety."""
    if isinstance(net_or_spikes, RecurrentSpikingNetwork):
        if spikes is None:
            raise ValueError("spikes must be provided when the first argument is a RecurrentSpikingNetwork")
        net: RecurrentSpikingNetwork | None = net_or_spikes
        if net.enable_stdp:
            raise ValueError(
                "Feature extraction requires STDP to be disabled (net.enable_stdp=False) "
                "so learned weights are not modified during evaluation."
            )
        raw_spikes = spikes
    else:
        if spikes is not None:
            raise TypeError("First argument must be a RecurrentSpikingNetwork when spikes is provided")
        net = None
        raw_spikes = net_or_spikes

    if isinstance(raw_spikes, list):
        if len(raw_spikes) == 0:
            raise ValueError("spikes list must be non-empty")
        single_sample = False
        spikes_list = [_validate_spike_matrix(item, index=idx) for idx, item in enumerate(raw_spikes)]
        first_shape = spikes_list[0].shape
        for idx, item in enumerate(spikes_list[1:], start=1):
            if item.shape != first_shape:
                raise ValueError(
                    f"All spike matrices must have the same shape; index 0 has {first_shape}, "
                    f"index {idx} has {item.shape}"
                )
    else:
        single_sample = True
        spikes_list = [_validate_spike_matrix(raw_spikes)]

    return net, spikes_list, single_sample


def _run_network(net: RecurrentSpikingNetwork, spikes: np.ndarray) -> np.ndarray:
    """Run ``net`` on ``spikes`` after resetting transient state."""
    if net.enable_stdp:
        raise ValueError(
            "Feature extraction requires STDP to be disabled (net.enable_stdp=False) "
            "so learned weights are not modified during evaluation."
        )
    net.reset()
    result = net.run(spikes, record_weight_history=False)
    return result["spikes"]


def extract_total(
    net_or_spikes: RecurrentSpikingNetwork | np.ndarray | list[np.ndarray],
    spikes: np.ndarray | list[np.ndarray] | None = None,
) -> np.ndarray:
    """Return total spike counts per neuron.

    Usage:
    - ``extract_total(spikes)`` – direct from binary spike matrix or list of matrices.
    - ``extract_total(net, spikes)`` – runs ``net`` (with ``enable_stdp=False``) on each spike train.
    """
    net, spikes_list, single_sample = _resolve_inputs(net_or_spikes, spikes)
    feats = []
    for s in spikes_list:
        out_spikes = _run_network(net, s) if net is not None else s
        feats.append(out_spikes.sum(axis=0))
    arr = np.stack(feats)
    return arr[0] if single_sample else arr


def raw_bins(rates: list[np.ndarray], bins: int = 4) -> np.ndarray:
    """Baseline: sum rates in equal-duration windows across all channels.

    Validation:
    - ``rates`` must be a non-empty list of finite 2D ``np.ndarray``
    - all arrays must have the same shape ``(time, channels)``
    - ``bins`` must be a positive integer (booleans rejected)
    - each array must have at least one timestep and one channel

    If ``bins`` exceeds the number of timesteps, empty bin slices evaluate to zero.
    Returns an array of shape ``(n_samples, n_channels * bins)``.
    """
    n_bins = _validate_bins(bins)
    if not isinstance(rates, list) or len(rates) == 0:
        raise ValueError("rates must be a non-empty list of 2D ndarrays")

    first_shape = None
    validated_rates: list[np.ndarray] = []
    for idx, r in enumerate(rates):
        if not isinstance(r, np.ndarray) or r.ndim != 2:
            raise ValueError(f"rate at index {idx} is not a 2D ndarray")
        if r.shape[0] == 0 or r.shape[1] == 0:
            raise ValueError("each rate must have at least one timestep and one channel")
        r_f = r.astype(np.float64, copy=False)
        if not np.all(np.isfinite(r_f)):
            raise ValueError(f"rate at index {idx} must contain only finite numerical values")
        if first_shape is None:
            first_shape = r_f.shape
        elif r_f.shape != first_shape:
            raise ValueError("all rate arrays must have the same shape")
        validated_rates.append(r_f)

    n_steps = validated_rates[0].shape[0]
    edges = np.linspace(0, n_steps, n_bins + 1, dtype=int)
    feats = []
    for r in validated_rates:
        per_bin = [r[edges[i] : edges[i + 1]].sum(axis=0) for i in range(n_bins)]
        feats.append(np.concatenate(per_bin).astype(np.float64))
    return np.stack(feats)


def extract_bins(
    net_or_spikes: RecurrentSpikingNetwork | np.ndarray | list[np.ndarray],
    spikes: np.ndarray | list[np.ndarray] | None = None,
    *,
    bins: int = 4,
) -> np.ndarray:
    """Extract binned spike count features over ``bins`` equal-duration windows.

    If ``bins`` exceeds the number of timesteps (including ``T == 0``), empty slices
    contribute zero spike counts.
    """
    n_bins = _validate_bins(bins)
    net, spikes_list, single_sample = _resolve_inputs(net_or_spikes, spikes)
    feats = []
    for s in spikes_list:
        out_spikes = _run_network(net, s) if net is not None else s
        n_steps = out_spikes.shape[0]
        edges = np.linspace(0, n_steps, n_bins + 1, dtype=int)
        bvec = [out_spikes[edges[i] : edges[i + 1]].sum(axis=0) for i in range(n_bins)]
        feats.append(np.concatenate(bvec))
    arr = np.stack(feats)
    return arr[0] if single_sample else arr


def extract_latency(
    net_or_spikes: RecurrentSpikingNetwork | np.ndarray | list[np.ndarray],
    spikes: np.ndarray | list[np.ndarray] | None = None,
    *,
    dt_ms: float | None = None,
) -> np.ndarray:
    """Return first-spike latency per neuron.

    Parameters
    ----------
    net_or_spikes : RecurrentSpikingNetwork or array-like
        Either a network (with ``spikes`` passed as the second argument) or spike matrix/list.
    spikes : array-like, optional
        Binary spike matrix or list of matrices when ``net_or_spikes`` is a network.
    dt_ms : float, optional
        If ``None`` (default), returns zero-based timestep indices with ``T`` as the
        silent-neuron sentinel. If provided, must be finite and positive, and latencies
        are returned in milliseconds (``index * dt_ms``) with ``T * dt_ms`` as sentinel.
    """
    if dt_ms is not None:
        if isinstance(dt_ms, bool) or not math.isfinite(float(dt_ms)) or float(dt_ms) <= 0.0:
            raise ValueError(f"dt_ms must be finite and positive when provided, got {dt_ms!r}")
        scale = float(dt_ms)
    else:
        scale = 1.0

    net, spikes_list, single_sample = _resolve_inputs(net_or_spikes, spikes)
    feats = []
    for s in spikes_list:
        out_spikes = _run_network(net, s) if net is not None else s
        n_steps = out_spikes.shape[0]
        if n_steps == 0:
            first = np.zeros(out_spikes.shape[1], dtype=np.float64)
        else:
            first = np.argmax(out_spikes, axis=0)
            silent = out_spikes.sum(axis=0) == 0
            first = np.where(silent, n_steps, first).astype(np.float64) * scale
        feats.append(first)
    arr = np.stack(feats)
    return arr[0] if single_sample else arr
