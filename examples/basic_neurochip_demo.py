"""NeuroChip AR v0.1: Basic Neuromorphic Computing Demonstration.

Executes a complete software-first neuromorphic pipeline:
1. Seeded Poisson rate encoding of continuous input stimuli.
2. Feed-forward synaptic current integration and LIF membrane potential dynamics.
3. Postsynaptic spike generation and refractory period behavior.
4. Pair-based STDP learning showing both potentiation (LTP) and depression (LTD).
5. Comprehensive multi-panel visualization saved to artifacts/basic_neurochip_demo.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neurochip_ar.encoding import poisson_rate_encoder
from neurochip_ar.network import FeedForwardNetwork
from neurochip_ar.stdp import PairSTDP


def run_demonstration() -> Path:
    # ---------------------------------------------------------
    # 1. Simulation setup & Input Poisson Rate Encoding
    # ---------------------------------------------------------
    dt_ms = 1.0  # 1 ms time step
    dt_s = dt_ms / 1000.0
    duration_s = 0.25  # 250 ms simulation
    n_timesteps = int(duration_s / dt_s)
    time_axis = np.arange(n_timesteps) * dt_ms

    n_inputs = 2
    n_outputs = 2

    # Two input channels with distinct stimulus rates
    stimulus_rates = np.array([0.75, 0.35])  # normalized in [0, 1]
    input_spikes = poisson_rate_encoder(
        stimulus_rates,
        duration=duration_s,
        dt=dt_s,
        r_max=80.0,
        seed=42,
    )

    # ---------------------------------------------------------
    # 2. Feed-forward network configuration & execution
    # ---------------------------------------------------------
    # Initial synaptic weight matrix (n_output x n_input)
    # Channel 0 drives Output 0; Channel 1 drives Output 1
    w_initial = np.array(
        [
            [18.0, 4.0],
            [4.0, 18.0],
        ],
        dtype=np.float64,
    )

    network = FeedForwardNetwork(
        n_input=n_inputs,
        n_output=n_outputs,
        weight_matrix=w_initial,
        tau_syn=6.0,
        dt=dt_ms,
        neuron_params={
            "v_rest": -65.0,
            "v_reset": -65.0,
            "v_threshold": -50.0,
            "r_membrane": 1.0,
            "tau_m": 15.0,
            "refractory_period": 4.0,
        },
    )

    sim_results = network.run(input_spikes)
    out_spikes = sim_results["spikes"]
    out_voltages = sim_results["voltages"]
    out_currents = sim_results["synaptic_currents"]

    # ---------------------------------------------------------
    # 3. STDP Learning
    # ---------------------------------------------------------
    stdp = PairSTDP(
        n_input=n_inputs,
        n_output=n_outputs,
        a_plus=0.08,
        a_minus=0.085,
        tau_plus=15.0,
        tau_minus=15.0,
        w_min=0.0,
        w_max=30.0,
        dt=dt_ms,
    )

    w_final = stdp.apply_train(w_initial, input_spikes, out_spikes)

    # ---------------------------------------------------------
    # 4. Controlled STDP Verification Experiment (LTP & LTD)
    # ---------------------------------------------------------
    delta_t_range = np.arange(-40, 41, 1)  # delta_t = t_post - t_pre
    dw_measured = []

    for dt_pair in delta_t_range:
        stdp_test = PairSTDP(
            n_input=1,
            n_output=1,
            a_plus=0.05,
            a_minus=0.055,
            tau_plus=15.0,
            tau_minus=15.0,
            w_min=0.0,
            w_max=2.0,
            dt=1.0,
        )
        w_baseline = np.array([[1.0]])
        test_len = 100
        s_pre = np.zeros((test_len, 1), dtype=np.int32)
        s_post = np.zeros((test_len, 1), dtype=np.int32)

        t_pre = 50
        t_post = 50 + dt_pair
        if 0 <= t_pre < test_len and 0 <= t_post < test_len:
            s_pre[t_pre, 0] = 1
            s_post[t_post, 0] = 1
            w_res = stdp_test.apply_train(w_baseline, s_pre, s_post)
            dw_measured.append(w_res[0, 0] - w_baseline[0, 0])
        else:
            dw_measured.append(0.0)

    # ---------------------------------------------------------
    # 5. Console Reporting
    # ---------------------------------------------------------
    print("=" * 60)
    print("NeuroChip AR v0.1 — Demonstration")
    print("=" * 60)
    print(f"Simulation duration: {duration_s * 1000:.1f} ms | Timestep: {dt_ms:.1f} ms")
    print(f"Input spikes generated: Ch0 = {np.sum(input_spikes[:, 0])}, Ch1 = {np.sum(input_spikes[:, 1])}")
    print(f"Output spikes emitted:  Neuron0 = {np.sum(out_spikes[:, 0])}, Neuron1 = {np.sum(out_spikes[:, 1])}")
    print("\nInitial Synaptic Weights (W_initial):")
    print(np.round(w_initial, 3))
    print("\nFinal Synaptic Weights after STDP (W_final):")
    print(np.round(w_final, 3))
    print("\nWeight delta (W_final - W_initial):")
    print(np.round(w_final - w_initial, 3))

    # ---------------------------------------------------------
    # 6. Multi-Panel Visualization
    # ---------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=False, constrained_layout=True)

    # Panel 1: Presynaptic Input Spikes (Raster)
    ax0 = axes[0]
    for ch in range(n_inputs):
        spike_times = time_axis[input_spikes[:, ch] == 1]
        ax0.vlines(
            spike_times,
            ch - 0.35,
            ch + 0.35,
            color=f"C{ch}",
            linewidth=1.5,
            label=f"Input Ch {ch}",
        )
    ax0.set_yticks([0, 1])
    ax0.set_yticklabels(["Input 0", "Input 1"])
    ax0.set_xlim(0, time_axis[-1])
    ax0.set_title(
        "A. Presynaptic Poisson Spike Trains (Input Stimulus)",
        loc="left",
        fontweight="bold",
    )
    ax0.set_ylabel("Channel")
    ax0.grid(True, linestyle=":", alpha=0.5)

    # Panel 2: Synaptic Currents
    ax1 = axes[1]
    ax1.plot(time_axis, out_currents[:, 0], label="I_syn Output 0", color="royalblue")
    ax1.plot(
        time_axis,
        out_currents[:, 1],
        label="I_syn Output 1",
        color="darkorange",
        linestyle="--",
    )
    ax1.set_xlim(0, time_axis[-1])
    ax1.set_title(
        "B. Postsynaptic Synaptic Current Dynamics (Exponential Decay)",
        loc="left",
        fontweight="bold",
    )
    ax1.set_ylabel("Current (nA)")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Panel 3: LIF Membrane Voltages & Output Spikes
    ax2 = axes[2]
    ax2.plot(
        time_axis,
        out_voltages[:, 0],
        label="V_m Output 0",
        color="royalblue",
        alpha=0.9,
    )
    ax2.plot(
        time_axis,
        out_voltages[:, 1],
        label="V_m Output 1",
        color="darkorange",
        alpha=0.8,
    )
    ax2.axhline(-50.0, color="crimson", linestyle=":", label="Threshold (-50 mV)")
    ax2.axhline(-65.0, color="gray", linestyle="-.", alpha=0.6, label="Rest / Reset (-65 mV)")

    # Overlay output spike markers
    spikes_0 = time_axis[out_spikes[:, 0] == 1]
    spikes_1 = time_axis[out_spikes[:, 1] == 1]
    if len(spikes_0) > 0:
        ax2.scatter(
            spikes_0,
            np.full_like(spikes_0, -48.0),
            marker="v",
            color="royalblue",
            s=30,
            zorder=5,
        )
    if len(spikes_1) > 0:
        ax2.scatter(
            spikes_1,
            np.full_like(spikes_1, -48.0),
            marker="v",
            color="darkorange",
            s=30,
            zorder=5,
        )

    ax2.set_xlim(0, time_axis[-1])
    ax2.set_title(
        "C. Postsynaptic LIF Membrane Potentials & Spikes",
        loc="left",
        fontweight="bold",
    )
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Membrane (mV)")
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle=":", alpha=0.5)

    # Panel 4: STDP Learning Window (LTP & LTD) & Synaptic Weight Adaptation
    ax3 = axes[3]
    # Subdivide panel 4 conceptually into two subplots or plot both side by side
    # We will use ax3 for STDP empirical pairing curve
    ax3.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax3.axvline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax3.plot(
        delta_t_range,
        dw_measured,
        color="purple",
        marker="o",
        markersize=3,
        label="Empirical STDP (ΔW)",
    )
    ax3.fill_between(
        delta_t_range[delta_t_range > 0],
        0,
        np.array(dw_measured)[delta_t_range > 0],
        color="forestgreen",
        alpha=0.25,
        label="LTP (Pre before Post)",
    )
    ax3.fill_between(
        delta_t_range[delta_t_range < 0],
        0,
        np.array(dw_measured)[delta_t_range < 0],
        color="crimson",
        alpha=0.25,
        label="LTD (Post before Pre)",
    )
    ax3.set_title(
        "D. STDP Plasticity Window: Controlled Potentiation vs. Depression",
        loc="left",
        fontweight="bold",
    )
    ax3.set_xlabel("Timing difference Δt = t_post - t_pre (ms)")
    ax3.set_ylabel("Weight Change ΔW")
    ax3.legend(loc="upper right")
    ax3.grid(True, linestyle=":", alpha=0.5)

    # Save to artifacts directory
    output_dir = Path(__file__).resolve().parent.parent / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "basic_neurochip_demo.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)

    print(f"\nVisualization saved successfully to: {out_file}")
    return out_file


if __name__ == "__main__":
    run_demonstration()
