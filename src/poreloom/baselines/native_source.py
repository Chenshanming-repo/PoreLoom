"""Read a native window export (``native_source`` / ``native_development/source``).

Layout, one row per parent read:

- ``metadata.npz``: ``labels`` (microbe 0 / host 1), ``raw_length``, ``split`` or
  ``role``, ``eligible_<key>`` masks, and for TRAIN parents ``cache_offset`` /
  ``cache_length`` into the full-read caches.
- ``adc_1000_4000.npy`` / ``adc_1500_4500.npy``: int16 ADC windows [N, 3000].
- ``pa_1500_4500.npy``: float32 calibrated pA windows [N, 3000].
- ``train_{microbe,host}_adc.npy`` (training export only): concatenated full
  int16 ADC of every TRAIN parent, used for ReadCurrent/CAFES tiling.

Rows whose read is too short for a window are zero and not eligible.
"""
from pathlib import Path

import numpy as np

WINDOWS = {'adc_1000_4000': (np.int16, 4000), 'adc_1500_4500': (np.int16, 4500),
           'pa_1500_4500': (np.float32, 4500)}


class NativeSource:
    def __init__(self, root):
        self.root = Path(root)
        with np.load(self.root / 'metadata.npz', allow_pickle=False) as archive:
            self.meta = {key: archive[key] for key in archive.files}
        self.n = len(self.meta['labels'])
        if any(value.shape[:1] != (self.n,) for value in self.meta.values()):
            raise ValueError(f'{self.root}: metadata arrays differ in length')
        if not np.isin(self.meta['labels'], (0, 1)).all():
            raise ValueError(f'{self.root}: labels must be microbe 0 / host 1')

    @property
    def labels(self):
        return self.meta['labels']

    def windows(self, key):
        dtype, minimum = WINDOWS[key]
        array = np.load(self.root / f'{key}.npy', mmap_mode='r', allow_pickle=False)
        if array.dtype != np.dtype(dtype) or array.shape != (self.n, 3000):
            raise ValueError(f'{self.root / key}: expected {np.dtype(dtype)} [{self.n}, 3000]')
        eligible = self.eligible(key)
        if not np.array_equal(eligible, self.meta['raw_length'] >= minimum):
            raise ValueError(f'{self.root}: eligible_{key} disagrees with raw_length')
        return array

    def eligible(self, key):
        return self.meta[f'eligible_{key}'].astype(bool)

    def rows(self, split):
        """Row indices of one split (``train``/``dev``) or role (e.g. ``C_screen``)."""
        if 'split' in self.meta and split in ('train', 'dev'):
            return np.flatnonzero(self.meta['split'] == split)
        return np.flatnonzero(self.meta['role'] == split)

    def train_cache(self, label):
        name = 'microbe' if label == 0 else 'host'
        cache = np.load(self.root / f'train_{name}_adc.npy', mmap_mode='r', allow_pickle=False)
        if cache.ndim != 1 or cache.dtype != np.int16:
            raise ValueError(f'{self.root}: train_{name}_adc.npy must be 1-D int16')
        return cache

    def full_read(self, row, cache):
        offset, length = int(self.meta['cache_offset'][row]), int(self.meta['cache_length'][row])
        if offset < 0:
            raise ValueError(f'row {row} has no cached full read (only TRAIN parents are cached)')
        return cache[offset:offset + length]
