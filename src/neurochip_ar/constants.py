# -*- coding: utf-8 -*-
"""Common constants for NeuroChip AR.

All time‑step values are expressed in **milliseconds** for the simulation core
(LIFNeuron, FeedForwardNetwork, etc.) while the encoder operates on **seconds**.
Defining the conversion here makes the relationship explicit and prevents unit
mismatches.
"""

# Encoder time step (seconds)
ENCODER_DT_SECONDS: float = 0.001  # 1 ms

# Network simulation time step (milliseconds)
NETWORK_DT_MS: float = ENCODER_DT_SECONDS * 1000.0


# Helper conversion functions
def seconds_to_ms(seconds: float) -> float:
    """Convert seconds to milliseconds.

    Parameters
    ----------
    seconds: float
        Time in seconds.

    Returns
    -------
    float
        Time in milliseconds.
    """
    return seconds * 1000.0


def ms_to_seconds(ms: float) -> float:
    """Convert milliseconds to seconds.

    Parameters
    ----------
    ms: float
        Time in milliseconds.

    Returns
    -------
    float
        Time in seconds.
    """
    return ms / 1000.0
