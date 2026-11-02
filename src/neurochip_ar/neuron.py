"""Leaky Integrate-and-Fire (LIF) neuron model.

Implements standard LIF membrane dynamics using forward Euler numerical integration:
    dV/dt = (V_rest - V + R * I) / tau

Units:
- Potential (V, V_rest, V_reset, V_threshold): mV
- Input resistance (R): MΩ
- Current (I): nA
- Time constant (tau), time step (dt), refractory period: ms
Note: R * I yields (MΩ * nA) = (10^6 Ω * 10^-9 A) = 10^-3 V = 1 mV, maintaining unit consistency.
"""

from __future__ import annotations


class LIFNeuron:
    """Leaky Integrate-and-Fire neuron with absolute refractory period.

    Parameters
    ----------
    v_rest : float, default -65.0
        Resting membrane potential in mV.
    v_reset : float, default -65.0
        Reset potential in mV applied immediately after a spike.
    v_threshold : float, default -50.0
        Spike generation threshold in mV.
    r_membrane : float, default 1.0
        Membrane input resistance in MΩ.
    tau_m : float, default 20.0
        Membrane time constant in ms.
    dt : float, default 1.0
        Simulation integration time step in ms.
    refractory_period : float, default 5.0
        Absolute refractory duration in ms during which the membrane
        is clamped at v_reset and cannot spike.
    """

    def __init__(
        self,
        v_rest: float = -65.0,
        v_reset: float = -65.0,
        v_threshold: float = -50.0,
        r_membrane: float = 1.0,
        tau_m: float = 20.0,
        dt: float = 1.0,
        refractory_period: float = 5.0,
    ) -> None:
        if dt <= 0.0:
            raise ValueError(f"Simulation timestep dt must be positive, got {dt}")
        if tau_m <= 0.0:
            raise ValueError(f"Membrane time constant tau_m must be positive, got {tau_m}")
        if r_membrane <= 0.0:
            raise ValueError(f"Membrane resistance r_membrane must be positive, got {r_membrane}")
        if refractory_period < 0.0:
            raise ValueError(f"refractory_period must be non-negative, got {refractory_period}")
        if v_threshold <= v_reset:
            raise ValueError(f"v_threshold ({v_threshold}) must be strictly greater than v_reset ({v_reset})")

        self.v_rest = float(v_rest)
        self.v_reset = float(v_reset)
        self.v_threshold = float(v_threshold)
        self.r_membrane = float(r_membrane)
        self.tau_m = float(tau_m)
        self.dt = float(dt)
        self.refractory_period = float(refractory_period)

        self.v = self.v_rest
        self.refractory_time_left = 0.0

    def reset(self) -> None:
        """Reset the neuron to its resting state."""
        self.v = self.v_rest
        self.refractory_time_left = 0.0

    def step(self, i_inj: float) -> bool:
        """Advance neuron state by one timestep dt with injected current i_inj.

        Parameters
        ----------
        i_inj : float
            Injected/synaptic current in nA.

        Returns
        -------
        bool
            True if a spike was generated at this step, False otherwise.
        """
        if self.refractory_time_left > 0.0:
            self.refractory_time_left = max(0.0, self.refractory_time_left - self.dt)
            self.v = self.v_reset
            return False

        # Forward Euler numerical integration:
        # dV = dt/tau * (V_rest - V + R * I)
        dv = (self.dt / self.tau_m) * (self.v_rest - self.v + self.r_membrane * float(i_inj))
        self.v += dv

        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self.refractory_time_left = self.refractory_period
            return True

        return False

    def __repr__(self) -> str:
        return (
            f"LIFNeuron(v={self.v:.2f} mV, v_rest={self.v_rest:.1f} mV, "
            f"v_thresh={self.v_threshold:.1f} mV, refractory_left={self.refractory_time_left:.2f} ms)"
        )
