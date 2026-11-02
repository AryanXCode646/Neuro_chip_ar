"""Pair-based Spike-Timing-Dependent Plasticity (STDP).

Implements pair-based STDP using exponentially decaying presynaptic and postsynaptic
eligibility traces:
    dx/dt = -x / tau_plus  + s_pre
    dy/dt = -y / tau_minus + s_post

Synaptic update rules:
    - Pre-before-post (t_pre < t_post): Presynaptic spike leaves trace x > 0.
      A subsequent postsynaptic spike triggers Long-Term Potentiation (LTP):
          ΔW_ij = + A_plus * x_j
    - Post-before-pre (t_post < t_pre): Postsynaptic spike leaves trace y > 0.
      A subsequent presynaptic spike triggers Long-Term Depression (LTD):
          ΔW_ij = - A_minus * y_i

Simultaneous spikes:
    Updates are calculated from accumulated prior traces before registering
    coincident spikes at the current time step.
"""

from __future__ import annotations

import numpy as np


class PairSTDP:
    """Pair-based STDP learning rule with trace dynamics.

    Parameters
    ----------
    n_input : int
        Number of presynaptic input neurons.
    n_output : int
        Number of postsynaptic output neurons.
    a_plus : float, default 0.01
        Maximum potentiation step (learning rate for LTP). Must be positive.
    a_minus : float, default 0.0105
        Maximum depression step (learning rate for LTD). Must be positive.
    tau_plus : float, default 20.0
        Presynaptic trace decay time constant in ms. Must be positive.
    tau_minus : float, default 20.0
        Postsynaptic trace decay time constant in ms. Must be positive.
    w_min : float, default 0.0
        Minimum allowable synaptic weight.
    w_max : float, default 1.0
        Maximum allowable synaptic weight. Must be > w_min.
    dt : float, default 1.0
        Simulation time step in ms. Must be positive.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        a_plus: float = 0.01,
        a_minus: float = 0.0105,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        w_min: float = 0.0,
        w_max: float = 1.0,
        dt: float = 1.0,
    ) -> None:
        if n_input <= 0:
            raise ValueError(f"n_input must be positive, got {n_input}")
        if n_output <= 0:
            raise ValueError(f"n_output must be positive, got {n_output}")
        if a_plus <= 0.0:
            raise ValueError(f"a_plus must be positive, got {a_plus}")
        if a_minus <= 0.0:
            raise ValueError(f"a_minus must be positive, got {a_minus}")
        if tau_plus <= 0.0:
            raise ValueError(f"tau_plus must be positive, got {tau_plus}")
        if tau_minus <= 0.0:
            raise ValueError(f"tau_minus must be positive, got {tau_minus}")
        if w_max <= w_min:
            raise ValueError(f"w_max ({w_max}) must be strictly greater than w_min ({w_min})")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        self.n_input = int(n_input)
        self.n_output = int(n_output)
        self.a_plus = float(a_plus)
        self.a_minus = float(a_minus)
        self.tau_plus = float(tau_plus)
        self.tau_minus = float(tau_minus)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.dt = float(dt)

        self.decay_pre = float(np.exp(-self.dt / self.tau_plus))
        self.decay_post = float(np.exp(-self.dt / self.tau_minus))

        # Eligibility traces
        self.trace_pre = np.zeros(self.n_input, dtype=np.float64)
        self.trace_post = np.zeros(self.n_output, dtype=np.float64)

    def reset(self) -> None:
        """Reset presynaptic and postsynaptic eligibility traces to zero."""
        self.trace_pre.fill(0.0)
        self.trace_post.fill(0.0)

    def step(
        self,
        weights: np.ndarray,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
    ) -> np.ndarray:
        """Apply a single STDP step given spikes at the current timestep.

        Parameters
        ----------
        weights : np.ndarray, shape (n_output, n_input)
            Current synaptic weight matrix.
        pre_spikes : np.ndarray, shape (n_input,)
            Binary spike indicators for presynaptic neurons at time t.
        post_spikes : np.ndarray, shape (n_output,)
            Binary spike indicators for postsynaptic neurons at time t.

        Returns
        -------
        np.ndarray, shape (n_output, n_input)
            Updated synaptic weight matrix clamped to [w_min, w_max].
        """
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (self.n_output, self.n_input):
            raise ValueError(f"weights shape mismatch: expected ({self.n_output}, {self.n_input}), got {w.shape}")

        s_pre = np.asarray(pre_spikes, dtype=np.float64)
        s_post = np.asarray(post_spikes, dtype=np.float64)
        if s_pre.shape != (self.n_input,):
            raise ValueError(f"pre_spikes shape mismatch: expected ({self.n_input},), got {s_pre.shape}")
        if s_post.shape != (self.n_output,):
            raise ValueError(f"post_spikes shape mismatch: expected ({self.n_output},), got {s_post.shape}")

        # 1. Decay traces from previous step to current step t
        self.trace_pre *= self.decay_pre
        self.trace_post *= self.decay_post

        # 2. Compute weight update from existing traces
        # LTP: Postsynaptic spike pairs with presynaptic trace: outer(s_post, trace_pre)
        # LTD: Presynaptic spike pairs with postsynaptic trace: outer(trace_post, s_pre)
        dw_ltp = self.a_plus * np.outer(s_post, self.trace_pre)
        dw_ltd = self.a_minus * np.outer(self.trace_post, s_pre)
        w_updated = w + dw_ltp - dw_ltd

        # 3. Update traces with new spikes at time t
        self.trace_pre += s_pre
        self.trace_post += s_post

        # 4. Enforce hard boundary constraints
        np.clip(w_updated, self.w_min, self.w_max, out=w_updated)
        return w_updated

    def apply_train(
        self,
        weights: np.ndarray,
        pre_spike_train: np.ndarray,
        post_spike_train: np.ndarray,
    ) -> np.ndarray:
        """Apply STDP sequentially over complete spike trains.

        Parameters
        ----------
        weights : np.ndarray, shape (n_output, n_input)
            Initial synaptic weight matrix.
        pre_spike_train : np.ndarray, shape (n_timesteps, n_input)
            Presynaptic spike trains over time.
        post_spike_train : np.ndarray, shape (n_timesteps, n_output)
            Postsynaptic spike trains over time.

        Returns
        -------
        np.ndarray, shape (n_output, n_input)
            Final updated weight matrix.
        """
        pre_arr = np.asarray(pre_spike_train, dtype=np.float64)
        post_arr = np.asarray(post_spike_train, dtype=np.float64)

        if pre_arr.ndim != 2 or pre_arr.shape[1] != self.n_input:
            raise ValueError(f"pre_spike_train must be 2D of shape (n_timesteps, {self.n_input}), got {pre_arr.shape}")
        if post_arr.ndim != 2 or post_arr.shape[1] != self.n_output:
            raise ValueError(
                f"post_spike_train must be 2D of shape (n_timesteps, {self.n_output}), got {post_arr.shape}"
            )
        if pre_arr.shape[0] != post_arr.shape[0]:
            raise ValueError(f"Timestep count mismatch between pre ({pre_arr.shape[0]}) and post ({post_arr.shape[0]})")

        w = np.copy(weights)
        for t in range(pre_arr.shape[0]):
            w = self.step(w, pre_arr[t], post_arr[t])

        return w
