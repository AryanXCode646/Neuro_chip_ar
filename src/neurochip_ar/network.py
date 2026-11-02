"""Feed-forward spiking neural network.

Connects an input layer of spike trains to a layer of LIF output neurons
through a weighted synaptic matrix with exponential current decay.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from neurochip_ar.neuron import LIFNeuron


class FeedForwardNetwork:
    """Feed-forward network with exponentially decaying synaptic currents.

    Synaptic current dynamics:
        I_syn[t] = I_syn[t-1] * exp(-dt / tau_syn) + W @ s_in[t]

    Output neurons integrate I_syn[t] using LIF dynamics.

    Parameters
    ----------
    n_input : int
        Number of presynaptic input channels.
    n_output : int
        Number of postsynaptic output neurons.
    weight_matrix : np.ndarray, shape (n_output, n_input)
        Synaptic weights connecting inputs to outputs.
    tau_syn : float, default 5.0
        Synaptic current decay time constant in ms.
    dt : float, default 1.0
        Simulation integration time step in ms.
    neuron_params : dict, optional
        Additional parameter dictionary forwarded to LIFNeuron constructors
        (e.g., v_rest, v_reset, v_threshold, tau_m, refractory_period).
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        weight_matrix: np.ndarray | list[list[float]],
        tau_syn: float = 5.0,
        dt: float = 1.0,
        neuron_params: dict[str, Any] | None = None,
    ) -> None:
        if n_input <= 0:
            raise ValueError(f"n_input must be positive, got {n_input}")
        if n_output <= 0:
            raise ValueError(f"n_output must be positive, got {n_output}")
        if tau_syn <= 0.0:
            raise ValueError(f"tau_syn must be positive, got {tau_syn}")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        weights = np.asarray(weight_matrix, dtype=np.float64)
        if weights.shape != (n_output, n_input):
            raise ValueError(f"weight_matrix shape mismatch: expected ({n_output}, {n_input}), got {weights.shape}")

        self.n_input = int(n_input)
        self.n_output = int(n_output)
        self.weights = np.copy(weights)
        self.tau_syn = float(tau_syn)
        self.dt = float(dt)
        self.decay_factor = float(np.exp(-self.dt / self.tau_syn))

        params = neuron_params.copy() if neuron_params else {}
        params["dt"] = self.dt
        self.neurons = [LIFNeuron(**params) for _ in range(self.n_output)]

        self.i_syn = np.zeros(self.n_output, dtype=np.float64)

    def reset(self) -> None:
        """Reset all output neurons and synaptic currents to resting/zero state."""
        for neuron in self.neurons:
            neuron.reset()
        self.i_syn.fill(0.0)

    def step(self, input_spikes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Advance the network by one simulation time step dt.

        Parameters
        ----------
        input_spikes : np.ndarray, shape (n_input,)
            Binary spike vector for current time step.

        Returns
        -------
        output_spikes : np.ndarray, shape (n_output,)
            Binary output spike vector (1 = spike, 0 = no spike).
        voltages : np.ndarray, shape (n_output,)
            Membrane potential of each output neuron in mV.
        """
        s_in = np.asarray(input_spikes, dtype=np.float64)
        if s_in.shape != (self.n_input,):
            raise ValueError(f"input_spikes shape mismatch: expected ({self.n_input},), got {s_in.shape}")

        # Decay previous synaptic current and add incoming spikes weighted sum
        self.i_syn = (self.i_syn * self.decay_factor) + (self.weights @ s_in)

        out_spikes = np.zeros(self.n_output, dtype=np.int32)
        out_v = np.zeros(self.n_output, dtype=np.float64)

        for i, neuron in enumerate(self.neurons):
            spike = neuron.step(self.i_syn[i])
            out_spikes[i] = 1 if spike else 0
            out_v[i] = neuron.v

        return out_spikes, out_v

    def run(self, input_spikes: np.ndarray) -> dict[str, np.ndarray]:
        """Simulate the network across a full binary spike matrix.

        Parameters
        ----------
        input_spikes : np.ndarray, shape (n_timesteps, n_input)
            2D binary matrix of input spike trains over time.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing recorded simulation histories:
            - "spikes": np.ndarray of shape (n_timesteps, n_output), dtype int32
            - "voltages": np.ndarray of shape (n_timesteps, n_output), dtype float64
            - "synaptic_currents": np.ndarray of shape (n_timesteps, n_output), dtype float64
        """
        arr = np.asarray(input_spikes, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != self.n_input:
            raise ValueError(f"input_spikes must be 2D of shape (n_timesteps, {self.n_input}), got {arr.shape}")

        n_steps = arr.shape[0]
        recorded_spikes = np.zeros((n_steps, self.n_output), dtype=np.int32)
        recorded_v = np.zeros((n_steps, self.n_output), dtype=np.float64)
        recorded_i = np.zeros((n_steps, self.n_output), dtype=np.float64)

        for t in range(n_steps):
            out_spikes, out_v = self.step(arr[t])
            recorded_spikes[t] = out_spikes
            recorded_v[t] = out_v
            recorded_i[t] = self.i_syn

        return {
            "spikes": recorded_spikes,
            "voltages": recorded_v,
            "synaptic_currents": recorded_i,
        }
