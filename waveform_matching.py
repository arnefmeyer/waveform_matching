#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
Spike cluster waveform matching using the algorithm described in

Tolias, A. S.; Ecker, A. S.; Siapas, A. G.; Hoenselaar, A.; Keliris, G. A. & Logothetis, N. K.
Recording chronically from the same neurons in awake, behaving primates.
Journal of neurophysiology, 2007, 98, 3780-3790
"""

from __future__ import print_function

import numpy as np


def vec_norm(v):
    return np.sqrt(np.sum(v ** 2))


def compute_waveform_distance(x, y):
    """compute waveform distances d_1 (shape) and d_2 (scaling)

    Parameters
    ----------
    x, y: array-like
        The average waveforms of the first (x) and second (y)
        spike cluster. Each column contains a single electrode
        channel and x and y must have the same number of columns.
    """

    # check input arrays
    x = np.asarray(x)
    y = np.asarray(y)

    assert x.shape == y.shape, "arrays x and y must have the same shape"

    if x.ndim == 1:
        # this assumes that x and y contain a single channel
        x = np.atleast_2d(x).T
        y = np.atleast_2d(y).T

    n_channels = x.shape[1]

    # compute scaling factors
    alphas = np.zeros((n_channels,))
    for i in range(n_channels):
        # p. 3784
        # For each channel, for a pair of average waveforms x and y, we
        # first scale x by alpha to minimize the sum of squared
        # differences between x and y. We refer to the scaling factor
        # of x and y as alpha(x, y).
        alphas[i] = np.dot(x[:, i], y[:, i]) / np.dot(x[:, i], x[:, i])

    # We then compute two different distance measures d_1 and d_2.
    # d_1 is a normalized Euclidean distance between the scaled
    # waveforms where the sum is over the four channels i. This solely
    # captures the difference in shape because the x_i values have been
    # scaled to match y_i and both have further been scaled by vecnorm(y_i).
    d_1 = 0.
    for i in range(n_channels):
        d_1 += vec_norm(alphas[i] * x[:, i] - y[:, i]) / vec_norm(y[:, i])

    # d_2 captures the difference in amplitudes across the four channels.
    d_2a = np.max(np.abs(np.log(alphas)))
    d_2b = []
    for i in range(n_channels):
        for j in range(n_channels):
            d_2b.append(np.abs(np.log(alphas[i]) - np.log(alphas[j])))
    d_2 = d_2a + max(d_2b)

    return np.array([d_1, d_2])
