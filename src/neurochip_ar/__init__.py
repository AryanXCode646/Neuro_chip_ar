"""NeuroChip AR: Software-first brain-inspired AI and neuromorphic computing prototype."""

from neurochip_ar.encoding import poisson_rate_encoder
from neurochip_ar.network import FeedForwardNetwork
from neurochip_ar.neuron import LIFNeuron
from neurochip_ar.stdp import PairSTDP

__all__ = [
    "LIFNeuron",
    "poisson_rate_encoder",
    "FeedForwardNetwork",
    "PairSTDP",
]
