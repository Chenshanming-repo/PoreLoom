"""Raw ADC prefix cache for the streaming encoder: int16 [N, L] cut from the original BLOW5, verified per read.

Building needs pyslow5 (.venv-raw) and no torch. Membership and row order come from an existing metadata.npz with
read_id / source_chunk / raw_length / labels (1 = host), e.g. poreloom_20261009/data/<split> or
native_development/source. Every read is checked against its chunk's truth.jsonl (raw_sha256, length, class,
calibration) before its prefix is stored.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import file_sha256

PREFIX = 9000
FILES = ('adc.npy', 'metadata.npz')


def build_prefix_cache(metadata_path, dest, length=PREFIX, log=print):
    dest = Path(dest)
    if (dest/'manifest.json').exists(): return dest
    import pyslow5
    dest.mkdir(parents=True, exist_ok=True)
    with np.load(metadata_path, allow_pickle=False) as z:
        meta = {k: z[k].copy() for k in z.files}
    n = len(meta['read_id']); row_of = {str(r): i for i, r in enumerate(meta['read_id'])}
    adc = np.lib.format.open_memmap(dest/'adc.npy', mode='w+', dtype=np.int16, shape=(n, length)); adc[:] = 0
    calibration = np.zeros((n, 3)); seen = np.zeros(n, bool)
    for chunk in sorted(set(meta['source_chunk'].tolist())):
        chunk = Path(chunk)
        truth = {t['parent_id']: t for t in map(json.loads, (chunk/'truth.jsonl').read_text().splitlines())}
        reader = pyslow5.Open(str(chunk/'signals.blow5'), 'r')
        try:
            for rec in reader.seq_reads(aux='all'):
                parent = rec['read_id'].split('!')[1]
                if parent not in row_of: continue
                row, t = row_of[parent], truth[parent]
                raw = np.asarray(rec['signal']).astype('<i2', copy=False)
                if seen[row] or float(rec['sampling_rate']) != 4000: raise ValueError(f'{parent}: duplicate or not 4 kHz')
                if hashlib.sha256(raw.tobytes()).hexdigest() != t['raw_sha256']: raise ValueError(f'{parent}: raw_sha256')
                if len(raw) != meta['raw_length'][row]: raise ValueError(f'{parent}: raw_length')
                if (t['class'] == 'host') != bool(meta['labels'][row] == 1): raise ValueError(f'{parent}: class')
                cal = (float(rec['offset']), float(rec['range']), float(rec['digitisation']))
                if not np.allclose(cal, (t['offset'], t['range'], t['digitisation'])): raise ValueError(f'{parent}: calibration')
                adc[row, :min(length, len(raw))] = raw[:length]; calibration[row] = cal; seen[row] = True
        finally:
            reader.close()
        log('prefix', dest.name, int(seen.sum()), n)
    if not seen.all(): raise ValueError(f'{int((~seen).sum())} reads not found in their source chunks')
    adc.flush(); del adc
    extra = {k: meta[k] for k in ('role', 'reference_id') if k in meta}
    np.savez_compressed(dest/'metadata.npz', read_id=meta['read_id'], is_target=meta['labels'] == 0,
                        raw_length=meta['raw_length'], source_chunk=meta['source_chunk'], offset=calibration[:, 0],
                        range=calibration[:, 1], digitisation=calibration[:, 2], **extra)
    (dest/'manifest.json').write_text(json.dumps(dict(
        status='completed', reads=n, prefix_samples=length, units='int16 ADC from read start, zero past raw_length',
        class_counts=dict(target=int((meta['labels'] == 0).sum()), host=int((meta['labels'] == 1).sum())),
        metadata_source=str(Path(metadata_path).resolve()), metadata_sha256=file_sha256(metadata_path),
        raw_check='per-read raw_sha256, raw_length, class and calibration against truth.jsonl',
        files={name: file_sha256(dest/name) for name in FILES}), indent=2) + '\n')
    return dest


@dataclass
class PrefixSet:
    adc: np.ndarray
    is_target: np.ndarray
    raw_length: np.ndarray
    calibration: np.ndarray
    read_id: np.ndarray
    role: np.ndarray

    def __len__(self): return len(self.read_id)

    def pa(self, rows, start=0, stop=None):
        """pA float32 [len(rows), stop-start] for samples [start, stop) of each read (zeros past raw_length)."""
        from .streaming import to_pa  # torch-side only; building the cache must not need torch
        rows = np.asarray(rows); stop = self.adc.shape[1] if stop is None else stop
        offset, range_, digitisation = self.calibration[rows].T
        out = to_pa(self.adc[rows, start:stop], offset[:, None], range_[:, None], digitisation[:, None])
        out[np.arange(start, stop) >= self.raw_length[rows, None]] = 0
        return out


def load_prefix_cache(path, verify=True):
    path = Path(path); manifest = json.loads((path/'manifest.json').read_text())
    if verify:
        for name, expected in manifest['files'].items():
            if file_sha256(path/name) != expected: raise ValueError(f'prefix cache hash mismatch: {path/name}')
    with np.load(path/'metadata.npz', allow_pickle=False) as z:
        meta = {k: z[k] for k in z.files}
    role = meta['role'] if 'role' in meta else np.full(len(meta['read_id']), path.name)
    return PrefixSet(np.load(path/'adc.npy', mmap_mode='r'), meta['is_target'], meta['raw_length'],
                     np.stack([meta['offset'], meta['range'], meta['digitisation']], 1), meta['read_id'], role)
