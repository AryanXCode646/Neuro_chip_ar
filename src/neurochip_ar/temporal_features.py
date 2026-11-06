# -*- coding: utf-8 -*-
"""Utility functions for extracting temporal features from recurrent spiking networks.

The functions operate on a :class:`~neurochip_ar.recurrent.RecurrentSpikingNetwork` instance
and a list of binary spike‑train inputs (``np.ndarray`` of shape ``(T, n_input)``).
All functions reset the network state before each simulation to ensure independent
feature vectors while preserving learned synaptic weights.

Implemented features:
- ``extract_total`` – total spike count per neuron.
- ``extract_bins`` – concatenated spike counts over ``bins`` equal‑duration windows.
- ``extract_latency`` – time of first spike per neuron (``T`` if silent).

All functions return a ``np.ndarray`` of shape ``(N_samples, n_neurons * bins)`` (or ``(N_samples, n_neurons)``
for total/latency). They are deterministic given the same random seeds used for the
input spike generation.
"""

from __future__ import annotations

import numpy as np
from neurochip_ar.recurrent import RecurrentSpikingNetwork

__all__ = ["extract_total", "extract_bins", "extract_latency"]


def _run_network(net: RecurrentSpikingNetwork, spikes: np.ndarray) -> np.ndarray:
    """Run ``net`` on ``spikes`` after resetting state.

    The network's plasticity (STDP) is *disabled* during feature extraction – callers
    should ensure ``net.enable_stdp`` is ``False`` before invoking this helper.
    """
    net.reset()
    result = net.run(spikes, record_weight_history=False)
    return result["spikes"]


def _ensure_list(spikes_input):
    """Accept either a single ndarray or a list of ndarrays and return a list.
    """
    if isinstance(spikes_input, list):
        return spikes_input
    return [spikes_input]


def extract_total(*args, **kwargs) -> np.ndarray:
    """Return total spike counts per neuron.

    Usage:
    - ``extract_total(spikes)`` – direct from binary spike matrix.
    - ``extract_total(net, spikes)`` – runs ``net`` on each spike train before counting.
    """
    if len(args) == 1:
        net = None
        spikes_input = args[0]
    elif len(args) == 2:
        net, spikes_input = args
    else:
        raise TypeError('extract_total expects 1 or 2 positional arguments')
    spikes_list = _ensure_list(spikes_input)
    feats = []
    for s in spikes_list:
        if net is not None:
            out_spikes = _run_network(net, s)
            feats.append(out_spikes.sum(axis=0))
        else:
            feats.append(s.sum(axis=0))
    arr = np.stack(feats)
    return arr[0] if arr.shape[0] == 1 else arr

    """Return total spike counts per neuron for each sample.

    If *net* is None, compute directly from the binary spike matrix.
    """
    spikes_list = _ensure_list(spikes_input)
    feats = []
    for s in spikes_list:
        if net is not None:
            out_spikes = _run_network(net, s)
            feats.append(out_spikes.sum(axis=0))
        else:
            feats.append(s.sum(axis=0))
    arr = np.stack(feats)
    return arr[0] if arr.shape[0] == 1 else arr

    """Return total spike counts per neuron for each sample.

    Parameters
    ----------
    net: RecurrentSpikingNetwork
        Network with learned weights already configured.
    spikes_input: np.ndarray or list of np.ndarray
        Each element is a binary spike train of shape ``(T, n_input)``.
    """
    spikes_list = _ensure_list(spikes_input)
    feats = []
    for s in spikes_list:
        out_spikes = _run_network(net, s)
        feats.append(out_spikes.sum(axis=0))
    arr = np.stack(feats)
    return arr[0] if arr.shape[0] == 1 else arr


def extract_bins(*args, bins: int = 4) -> np.ndarray:
    """Extract binned spike count features.

    Usage:
    - ``extract_bins(spikes, bins=4)`` – works directly on binary spike matrix.
    - ``extract_bins(net, spikes, bins=4)`` – runs the network first.
    """
    if len(args) == 1:
        net = None
        spikes_input = args[0]
    elif len(args) == 2:
        net, spikes_input = args
    else:
        raise TypeError('extract_bins expects 1 or 2 positional arguments')
    spikes_list = _ensure_list(spikes_input)
    feats = []
    for s in spikes_list:
        if net is not None:
            out_spikes = _run_network(net, s)
        else:
            out_spikes = s
        n_steps = out_spikes.shape[0]
        edges = np.linspace(0, n_steps, bins + 1, dtype=int)
        bvec = []
        for i in range(bins):
            bvec.append(out_spikes[edges[i]:edges[i + 1]].sum(axis=0))
        feats.append(np.concatenate(bvec))
    arr = np.stack(feats)
    return arr[0] if arr.shape[0] == 1 else arr

    """Extract binned spike count features.

    Returns a 1‑D array for a single input or a 2‑D array ``(N_samples, n_neurons * bins)``.
    """
    spikes_list = _ensure_list(spikes_input)
    feats = []
    for s in spikes_list:
        out_spikes = _run_network(net, s)
        n_steps = out_spikes.shape[0]
        edges = np.linspace(0, n_steps, bins + 1, dtype=int)
        bvec = []
        for i in range(bins):
            bvec.append(out_spikes[edges[i] : edges[i + 1]].sum(axis=0))
        feats.append(np.concatenate(bvec))
    arr = np.stack(feats)
    return arr[0] if arr.shape[0] == 1 else arr


def extract_latency(*args, **kwargs) -> np.ndarray:
    """Return first‑spike latency per neuron.

    Usage:
    - ``extract_latency(spikes)`` – direct from binary spike matrix.
    - ``extract_latency(net, spikes)`` – runs ``net`` on each spike train before extracting.
    """
    if len(args) == 1:
        net = None
        spikes_input = args[0]
    elif len(args) == 2:
        net, spikes_input = args
    else:
        raise TypeError('extract_latency expects 1 or 2 positional arguments')
    spikes_list = _ensure_list(spikes_input)
    feats = []
    for s in spikes_list:
        out_spikes = _run_network(net, s) if net is not None else s
        first = np.argmax(out_spikes, axis=0)
        silent = out_spikes.sum(axis=0) == 0
        first = np.where(silent, out_spikes.shape[0], first).astype(np.float64)
        feats.append(first)
    arr = np.stack(feats)
    return arr[0] if arr.shape[0] == 1 else arr
    """Return first‑spike latency per neuron.

    Silent neurons receive a latency equal to the total number of timesteps.
    """
    spikes_list = _ensure_list(spikes_input)
    feats = []
    for s in spikes_list:
        out_spikes = _run_network(net, s)
        first = np.argmax(out_spikes, axis=0)
        silent = out_spikes.sum(axis=0) == 0
        first = np.where(silent, out_spikes.shape[0], first).astype(np.float64)
        feats.append(first)
    arr = np.stack(feats)
    return arr[0] if arr.shape[0] == 1 else arr
