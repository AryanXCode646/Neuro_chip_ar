# -*- coding: utf-8 -*-
"""Test that raw_bins preserves channel and bin identity."""

import numpy as np

from neurochip_ar.temporal_features import raw_bins


def test_raw_bins_channel_preservation():
    # Two channels, four timesteps per bin (total 8 timesteps, 2 bins)
    # Create rates where channel 0 is high in first half, channel 1 high in second half
    rates = []
    # channel 0 high first 4 steps, low next 4
    rates.append(np.concatenate([np.ones((4, 1)), np.zeros((4, 1))], axis=0))
    # channel 1 high first 4 steps, low next 4 (inverse)
    rates.append(np.concatenate([np.zeros((4, 1)), np.ones((4, 1))], axis=0))
    # Stack to shape (time, 2)
    rates = [np.hstack([rates[0], rates[1]])]  # single sample with both channels
    # Compute features with 2 bins
    feats = raw_bins(rates, bins=2)
    # Expected: first bin sums first 4 steps -> [4., 0.], second bin sums next 4 steps -> [0., 4.]
    expected = np.array([[4.0, 0.0, 0.0, 4.0]])
    np.testing.assert_array_equal(feats, expected)
