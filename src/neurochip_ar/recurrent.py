"""Recurrent spiking neural network with one-step delayed feedback and online STDP.

Extends the feedforward architecture to support recurrent synaptic connections
among output neurons with explicit one-timestep delay:
    I_syn[t] = alpha * I_syn[t-1] + W_in @ x[t] + W_rec @ y[t-1]

where:
    - x[t] is the external input spike vector at time t
    - y[t-1] is the output spike vector from the previous timestep (y[-1] = 0)
    - W_in is the input-to-neuron weight matrix (shape: (n_neurons, n_input))
    - W_rec is the recurrent weight matrix (shape: (n_neurons, n_neurons))
    - alpha = exp(-dt / tau_syn) is the synaptic current decay factor

Optionally applies online pair-based STDP to both input and recurrent synapses
at every simulation timestep.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from neurochip_ar.neuron import LIFNeuron
from neurochip_ar.stdp import PairSTDP


class RecurrentSpikingNetwork:
    """Recurrent spiking network with delayed feedback and online STDP.

    Parameters
    ----------
    n_input : int
        Number of presynaptic external input channels.
    n_neurons : int
        Number of recurrent LIF neurons in the network.
    input_weights : np.ndarray | list[list[float]]
        Synaptic weight matrix connecting external inputs to neurons.
        Shape must be (n_neurons, n_input).
    recurrent_weights : np.ndarray | list[list[float]]
        Synaptic weight matrix connecting recurrent neurons to each other.
        Shape must be (n_neurons, n_neurons). Diagonal must be zero (no self-connections).
    tau_syn : float, default 5.0
        Synaptic current decay time constant in ms.
    dt : float, default 1.0
        Simulation integration time step in ms.
    neuron_params : dict, optional
        Additional parameter dictionary forwarded to LIFNeuron constructors
        (e.g., v_rest, v_reset, v_threshold, tau_m, refractory_period).
    enable_stdp : bool, default False
        Whether to perform online STDP learning inside the simulation loop.
    stdp_input_params : dict, optional
        Parameters forwarded to PairSTDP for input synapses (a_plus, a_minus, etc.).
    stdp_rec_params : dict, optional
        Parameters forwarded to PairSTDP for recurrent synapses.
    """

    def __init__(
        self,
        n_input: int,
        n_neurons: int,
        input_weights: np.ndarray | list[list[float]],
        recurrent_weights: np.ndarray | list[list[float]],
        tau_syn: float = 5.0,
        dt: float = 1.0,
        neuron_params: dict[str, Any] | None = None,
        enable_stdp: bool = False,
        stdp_input_params: dict[str, Any] | None = None,
        stdp_rec_params: dict[str, Any] | None = None,
    ) -> None:
        if n_input <= 0:
            raise ValueError(f"n_input must be positive, got {n_input}")
        if n_neurons <= 0:
            raise ValueError(f"n_neurons must be positive, got {n_neurons}")
        if tau_syn <= 0.0:
            raise ValueError(f"tau_syn must be positive, got {tau_syn}")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        w_in = np.asarray(input_weights, dtype=np.float64)
        if w_in.shape != (n_neurons, n_input):
            raise ValueError(f"input_weights shape mismatch: expected ({n_neurons}, {n_input}), got {w_in.shape}")
        if not np.all(np.isfinite(w_in)):
            raise ValueError("input_weights must contain only finite numerical values")

        w_rec = np.asarray(recurrent_weights, dtype=np.float64)
        if w_rec.shape != (n_neurons, n_neurons):
            raise ValueError(
                f"recurrent_weights shape mismatch: expected ({n_neurons}, {n_neurons}), got {w_rec.shape}"
            )
        if not np.all(np.isfinite(w_rec)):
            raise ValueError("recurrent_weights must contain only finite numerical values")
        if np.any(np.diag(w_rec) != 0.0):
            raise ValueError("Self-connections are disallowed: recurrent_weights diagonal must be strictly zero")
        if np.any(w_rec < 0.0):
            raise ValueError("Recurrent weights must be non-negative in this milestone")

        # Validate STDP bounds if STDP is enabled
        if enable_stdp:
            in_cfg = stdp_input_params.copy() if stdp_input_params else {}
            rec_cfg = stdp_rec_params.copy() if stdp_rec_params else {}
            # Determine bounds, default to [0.0, 1.0] when not specified
            w_min_in = in_cfg.get("w_min", 0.0)
            w_max_in = in_cfg.get("w_max", 1.0)
            w_min_rec = rec_cfg.get("w_min", 0.0)
            w_max_rec = rec_cfg.get("w_max", 1.0)
            # Validate initial weight matrices against bounds
            if np.any(w_in < w_min_in) or np.any(w_in > w_max_in):
                raise ValueError(
                    f"Initial input_weights contain values outside configured STDP bounds [{w_min_in}, {w_max_in}]"
                )
            if np.any(w_rec < w_min_rec) or np.any(w_rec > w_max_rec):
                raise ValueError(
                    f"Initial recurrent_weights contain values outside configured STDP bounds "
                    f"[{w_min_rec}, {w_max_rec}]"
                )

        self.n_input = int(n_input)
        self.n_neurons = int(n_neurons)
        self.input_weights = np.copy(w_in)
        self.recurrent_weights = np.copy(w_rec)
        self.tau_syn = float(tau_syn)
        self.dt = float(dt)
        self.decay_factor = float(np.exp(-self.dt / self.tau_syn))

        params = neuron_params.copy() if neuron_params else {}
        params["dt"] = self.dt
        self.neurons = [LIFNeuron(**params) for _ in range(self.n_neurons)]

        self.i_syn = np.zeros(self.n_neurons, dtype=np.float64)
        self.y_prev = np.zeros(self.n_neurons, dtype=np.float64)

        self.enable_stdp = bool(enable_stdp)
        if self.enable_stdp:
            in_cfg = stdp_input_params.copy() if stdp_input_params else {}
            rec_cfg = stdp_rec_params.copy() if stdp_rec_params else {}
            in_cfg.setdefault("dt", self.dt)
            rec_cfg.setdefault("dt", self.dt)
            self.stdp_input = PairSTDP(n_input=self.n_input, n_output=self.n_neurons, **in_cfg)
            self.stdp_rec = PairSTDP(n_input=self.n_neurons, n_output=self.n_neurons, **rec_cfg)
        else:
            self.stdp_input = None
            self.stdp_rec = None

    def reset(self) -> None:
        """Reset output neurons, synaptic currents, previous spikes and STDP traces.

        Note: Learned synaptic weights are preserved across resets.
        """
        for neuron in self.neurons:
            neuron.reset()
        self.i_syn.fill(0.0)
        self.y_prev.fill(0.0)
        if self.stdp_input is not None:
            self.stdp_input.reset()
        if self.stdp_rec is not None:
            self.stdp_rec.reset()

    def step(self, input_spikes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Advance the recurrent network by one simulation time step dt.

        Parameters
        ----------
        input_spikes : np.ndarray, shape (n_input,)
            Binary spike vector from external inputs for the current time step.

        Returns
        -------
        output_spikes : np.ndarray, shape (n_neurons,)
            Binary output spike vector (1 = spike, 0 = no spike).
        voltages : np.ndarray, shape (n_neurons,)
            Membrane potential of each neuron in mV.
        """
        s_in = np.asarray(input_spikes, dtype=np.float64)
        if s_in.shape != (self.n_input,):
            raise ValueError(f"input_spikes shape mismatch: expected ({self.n_input},), got {s_in.shape}")
        if np.any((s_in != 0.0) & (s_in != 1.0)):
            raise ValueError("input_spikes must be binary indicators (0 or 1)")

        # 1. Update synaptic current using decayed current + input spikes + one-step delayed recurrent spikes
        self.i_syn = (
            (self.i_syn * self.decay_factor) + (self.input_weights @ s_in) + (self.recurrent_weights @ self.y_prev)
        )

        # 2. Advance LIF neurons
        out_spikes = np.zeros(self.n_neurons, dtype=np.int32)
        out_v = np.zeros(self.n_neurons, dtype=np.float64)

        for i, neuron in enumerate(self.neurons):
            spike = neuron.step(self.i_syn[i])
            out_spikes[i] = 1 if spike else 0
            out_v[i] = neuron.v

        s_out = out_spikes.astype(np.float64)

        # 3. Online STDP: update weights inside the timestep loop
        if self.enable_stdp and self.stdp_input is not None and self.stdp_rec is not None:
            self.input_weights = self.stdp_input.step(
                weights=self.input_weights,
                pre_spikes=s_in,
                post_spikes=s_out,
            )
            self.recurrent_weights = self.stdp_rec.step(
                weights=self.recurrent_weights,
                pre_spikes=self.y_prev,
                post_spikes=s_out,
            )
            # Enforce zero diagonal constraint after update
            np.fill_diagonal(self.recurrent_weights, 0.0)

        # 4. Save current spikes for next timestep's recurrent feedback
        self.y_prev = np.copy(s_out)

        return out_spikes, out_v

    def run(
        self,
        input_spikes: np.ndarray,
        record_weight_history: bool = False,
    ) -> dict[str, np.ndarray]:
        """Simulate the recurrent network across a full binary spike matrix.

        Parameters
        ----------
        input_spikes : np.ndarray, shape (n_timesteps, n_input)
            2D binary matrix of external input spike trains over time.
        record_weight_history : bool, default False
            If True, records input and recurrent weight matrices at every timestep.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing recorded simulation histories:
            - "spikes": np.ndarray of shape (n_timesteps, n_neurons), dtype int32
            - "voltages": np.ndarray of shape (n_timesteps, n_neurons), dtype float64
            - "synaptic_currents": np.ndarray of shape (n_timesteps, n_neurons), dtype float64
            - "input_weights_history": (optional) shape (n_timesteps, n_neurons, n_input)
            - "recurrent_weights_history": (optional) shape (n_timesteps, n_neurons, n_neurons)
        """
        arr = np.asarray(input_spikes, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != self.n_input:
            raise ValueError(f"input_spikes must be 2D of shape (n_timesteps, {self.n_input}), got {arr.shape}")
        if np.any((arr != 0.0) & (arr != 1.0)):
            raise ValueError("input_spikes must be binary indicators (0 or 1)")

        n_steps = arr.shape[0]
        recorded_spikes = np.zeros((n_steps, self.n_neurons), dtype=np.int32)
        recorded_v = np.zeros((n_steps, self.n_neurons), dtype=np.float64)
        recorded_i = np.zeros((n_steps, self.n_neurons), dtype=np.float64)

        w_in_hist = (
            np.zeros((n_steps, self.n_neurons, self.n_input), dtype=np.float64) if record_weight_history else None
        )
        w_rec_hist = (
            np.zeros((n_steps, self.n_neurons, self.n_neurons), dtype=np.float64) if record_weight_history else None
        )

        for t in range(n_steps):
            out_spikes, out_v = self.step(arr[t])
            recorded_spikes[t] = out_spikes
            recorded_v[t] = out_v
            recorded_i[t] = self.i_syn

            if record_weight_history:
                w_in_hist[t] = self.input_weights
                w_rec_hist[t] = self.recurrent_weights

        result: dict[str, np.ndarray] = {
            "spikes": recorded_spikes,
            "voltages": recorded_v,
            "synaptic_currents": recorded_i,
        }
        if record_weight_history:
            result["input_weights_history"] = w_in_hist
            result["recurrent_weights_history"] = w_rec_hist

        return result
