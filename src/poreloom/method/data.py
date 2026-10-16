"""Dataset layout for the PoreLoom method.

A dataset root holds one prefix cache per split (poreloom.prefix_data layout):

    <root>/train/{adc.npy, metadata.npz, manifest.json}
    <root>/dev/...
    <root>/test/...

adc.npy is int16 ADC [N, P] from the read start (P >= 9000, zero past the read end); metadata.npz holds read_id,
is_target (True = keep / target, False = host), raw_length (full native length), offset, range, digitisation (per-read
calibration), role; optionally repeat (TEST report subgroups). DEV is split by sha256(read_id) parity: half A (parity 0)
for early stopping and head fitting, half B (parity 1) for head selection.
"""
import hashlib
from pathlib import Path

import numpy as np

from ..prefix_data import PrefixSet, load_prefix_cache


def dev_half(read_id):
    return np.array([int(hashlib.sha256(str(r).encode()).hexdigest()[:8], 16) % 2 for r in read_id])


def subset(data, rows):
    rows = np.asarray(rows)
    return PrefixSet(np.asarray(data.adc[rows]), data.is_target[rows], data.raw_length[rows], data.calibration[rows],
                     data.read_id[rows], data.role[rows])


def load_split(root, split, verify=False):
    return load_prefix_cache(Path(root)/split, verify=verify)


def dev_halves(root):
    dev = load_split(root, 'dev')
    half = dev_half(dev.read_id)
    a, b = subset(dev, np.flatnonzero(half == 0)), subset(dev, np.flatnonzero(half == 1))
    for d in (a, b):
        if set(d.is_target.tolist()) != {False, True}: raise ValueError('each DEV half needs both classes')
    return a, b


def mad_normalize(x, threshold=3.5):
    """Per-row (x - median) / (1.4826 MAD); each |z| > threshold index is then replaced, in increasing index order, by
    its neighbours' mean (first / last index: its single neighbour). Returns (float32 z, valid = MAD > 1e-8)."""
    x = np.asarray(x, np.float64)
    median = np.median(x, axis=1)
    dev = x - median[:, None]
    mad = np.median(np.abs(dev), axis=1)
    valid = np.isfinite(mad) & (mad > 1e-8)
    z = np.zeros_like(x)
    z[valid] = dev[valid] / (1.4826 * mad[valid, None])
    width = x.shape[1]
    for row in np.flatnonzero(valid):
        for i in np.flatnonzero(np.abs(z[row]) > threshold):
            z[row, i] = z[row, 1] if i == 0 else z[row, width - 2] if i == width - 1 else (z[row, i - 1] + z[row, i + 1]) / 2
    return z.astype(np.float32), valid
