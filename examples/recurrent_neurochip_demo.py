"""Demonstration of Recurrent Spiking Neural Network with Online STDP.

Demonstrates:
1. External input spike trains driving recurrent LIF neurons.
2. One-step delayed recurrent synaptic feedback between output neurons.
3. Online pair-based STDP adapting input and recurrent weights at every timestep.
4. Generates raster plots, voltage traces, and weight evolution saved to artifacts/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neurochip_ar.encoding import poisson_rate_encoder
from neurochip_ar.recurrent import RecurrentSpikingNetwork


def run_demo() -> None:
    print("=" * 60)
    print("NeuroChip AR v0.2 — Recurrent SNN with Online STDP")
    print("=" * 60)

    duration_ms = 250.0
    dt = 1.0
    n_timesteps = int(duration_ms / dt)
    n_input = 2
    n_neurons = 3

    # Generate deterministic Poisson spike trains for external inputs
    # Channel 0: active burst in first half
    rates_ch0 = np.full(n_timesteps, 0.2)
    rates_ch0[20:100] = 0.85
    # Channel 1: steady moderate rate
    rates_ch1 = np.full(n_timesteps, 0.35)
    rates_2d = np.column_stack([rates_ch0, rates_ch1])

    input_spikes = poisson_rate_encoder(rates_2d, dt=dt * 1e-3, r_max=120.0, seed=42)

    # Initial input weights: input 0 -> neuron 0, input 1 -> neuron 1
    w_in_initial = np.array(
        [
            [22.0, 4.0],
            [4.0, 22.0],
            [2.0, 2.0],
        ]
    )

    # Initial recurrent weights: ring topology (0 -> 1 -> 2 -> 0), strictly 0 on diagonal
    w_rec_initial = np.array(
        [
            [0.0, 0.0, 8.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
        ]
    )

    # Build Recurrent Network with online STDP enabled
    net = RecurrentSpikingNetwork(
        n_input=n_input,
        n_neurons=n_neurons,
        input_weights=w_in_initial,
        recurrent_weights=w_rec_initial,
        tau_syn=6.0,
        dt=dt,
        neuron_params={
            "v_rest": -65.0,
            "v_reset": -65.0,
            "v_threshold": -50.0,
            "tau_m": 20.0,
            "r_membrane": 1.0,
            "refractory_period": 3.0,
        },
        enable_stdp=True,
        stdp_input_params={"a_plus": 0.02, "a_minus": 0.015, "w_min": 0.0, "w_max": 35.0},
        stdp_rec_params={"a_plus": 0.03, "a_minus": 0.02, "w_min": 0.0, "w_max": 20.0},
    )

    print(f"Simulation duration: {duration_ms} ms | Timestep: {dt} ms")
    print(f"Architecture: {n_input} external inputs -> {n_neurons} recurrent LIF neurons")
    print(f"Input spikes generated: Ch0 = {int(np.sum(input_spikes[:, 0]))}, Ch1 = {int(np.sum(input_spikes[:, 1]))}\n")

    print("Initial Input Weights (W_in):\n", w_in_initial)
    print("Initial Recurrent Weights (W_rec):\n", w_rec_initial)

    # Run simulation with weight history recording
    sim_data = net.run(input_spikes, record_weight_history=True)
    out_spikes = sim_data["spikes"]
    voltages = sim_data["voltages"]
    w_in_hist = sim_data["input_weights_history"]
    w_rec_hist = sim_data["recurrent_weights_history"]

    w_in_final = net.input_weights
    w_rec_final = net.recurrent_weights

    print("\nSimulation complete.")
    for i in range(n_neurons):
        print(f"Neuron {i} total spikes emitted: {int(np.sum(out_spikes[:, i]))}")

    print("\nFinal Input Weights after Online STDP:\n", np.round(w_in_final, 3))
    print("Final Recurrent Weights after Online STDP:\n", np.round(w_rec_final, 3))

    delta_w_in = w_in_final - w_in_initial
    delta_w_rec = w_rec_final - w_rec_initial
    print("\nInput Weight Deltas (W_in_final - W_in_initial):\n", np.round(delta_w_in, 3))
    print("Recurrent Weight Deltas (W_rec_final - W_rec_initial):\n", np.round(delta_w_rec, 3))

    # Visualization
    time_axis = np.arange(n_timesteps) * dt
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, constrained_layout=True)

    # 1. Input Spike Raster
    ax0 = axes[0]
    for ch in range(n_input):
        spike_times = time_axis[input_spikes[:, ch] == 1]
        ax0.scatter(spike_times, np.full_like(spike_times, ch), marker="|", s=70, color="navy")
    ax0.set_yticks([0, 1])
    ax0.set_yticklabels(["Input 0", "Input 1"])
    ax0.set_ylabel("Input Ch")
    ax0.set_title("A. External Poisson Input Spike Trains")
    ax0.grid(True, linestyle="--", alpha=0.5)

    # 2. Output Spike Raster
    ax1 = axes[1]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    for i in range(n_neurons):
        spike_times = time_axis[out_spikes[:, i] == 1]
        ax1.scatter(spike_times, np.full_like(spike_times, i), marker="|", s=90, color=colors[i], label=f"Neuron {i}")
    ax1.set_yticks(range(n_neurons))
    ax1.set_yticklabels([f"Neuron {i}" for i in range(n_neurons)])
    ax1.set_ylabel("Neuron")
    ax1.set_title("B. Recurrent Output Spikes (Delayed Feedback Dynamics)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 3. Membrane Potentials
    ax2 = axes[2]
    for i in range(n_neurons):
        ax2.plot(time_axis, voltages[:, i], label=f"V_m Neuron {i}", color=colors[i], alpha=0.85, linewidth=1.2)
    ax2.axhline(-50.0, color="gray", linestyle=":", linewidth=1.0, label="Threshold (-50 mV)")
    ax2.axhline(-65.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6, label="Rest / Reset (-65 mV)")
    ax2.set_ylabel("Potential (mV)")
    ax2.set_title("C. Membrane Potential Integration")
    ax2.legend(loc="upper right", ncol=4, fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 4. Weight Evolution History
    ax3 = axes[3]
    ax3.plot(time_axis, w_in_hist[:, 0, 0], color="blue", label="W_in[0, 0] (In0 -> N0)", linewidth=1.5)
    ax3.plot(time_axis, w_in_hist[:, 1, 1], color="darkcyan", label="W_in[1, 1] (In1 -> N1)", linewidth=1.5)
    ax3.plot(time_axis, w_rec_hist[:, 1, 0], color="purple", label="W_rec[1, 0] (N0 -> N1)", linewidth=1.5)
    ax3.plot(time_axis, w_rec_hist[:, 2, 1], color="crimson", label="W_rec[2, 1] (N1 -> N2)", linewidth=1.5)
    ax3.plot(time_axis, w_rec_hist[:, 0, 2], color="darkorange", label="W_rec[0, 2] (N2 -> N0)", linewidth=1.5)
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("Synaptic Weight")
    ax3.set_title("D. Online STDP Synaptic Weight Trajectories")
    ax3.legend(loc="upper left", ncol=3, fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.5)

    out_dir = Path(__file__).resolve().parent.parent / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "recurrent_neurochip_demo.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(f"\nVisualization saved successfully to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
