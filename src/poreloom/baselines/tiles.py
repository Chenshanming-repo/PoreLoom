"""ReadCurrent / CAFES tile datasets in the patched ReadCurrent trainer layout.

TRAIN uses upstream ReadCurrent tiling of each full read: three offset passes
(raw starts 1500, 2500, 3500), consecutive 3000-sample tiles to the read's end,
offset-major order. DEV uses only each read's first native window raw[1500:4500].

Output, for ``name`` in cafes/readcurrent::

    <output>/<name>/data/{pos,neg}/{train,valid}_preprocessed.npy
    <output>/<name>/data/{pos,neg}/{train,valid}_metadata.npz   row_index, raw_start, raw_end
    <output>/<name>/data/read_metadata.npz
    <output>/<name>/protocol.json

``pos`` is microbe (network label 1), ``neg`` is host (network label 0). CAFES
tiles are raw int16 ADC (features are computed per batch at fit time);
ReadCurrent tiles are normalized float32, and tiles with MAD <= 1e-8 are dropped.
"""
import json
from pathlib import Path

import numpy as np

from .native_source import NativeSource
from .preprocessing import readcurrent_inputs

GROUPS = ((0, 'pos'), (1, 'neg'))
ROLES = (('train', 'train'), ('dev', 'valid'))
BATCH = 128


def train_spans(length):
    for offset in range(3):
        for start in range(1500 + offset * 1000, int(length) - 2999, 3000):
            yield start, start + 3000


def tile_count(source, label, split):
    rows = np.flatnonzero((source.labels == label) & (source.meta['split'] == split))
    if split == 'dev':
        return int(source.eligible('adc_1500_4500')[rows].sum())
    return sum(len(range(1500 + o * 1000, int(source.meta['raw_length'][r]) - 2999, 3000))
               for o in range(3) for r in rows)


def tiles(source, label, split):
    """Yield ``(row, raw_start, raw_end, int16 window)`` in the published order."""
    rows = np.flatnonzero((source.labels == label) & (source.meta['split'] == split))
    if split == 'dev':
        first, eligible = source.windows('adc_1500_4500'), source.eligible('adc_1500_4500')
        for row in rows:
            if eligible[row]:
                yield int(row), 1500, 4500, np.asarray(first[row])
        return
    cache = source.train_cache(label)
    for offset in range(3):
        for row in rows:
            read = source.full_read(row, cache)
            for start in range(1500 + offset * 1000, len(read) - 2999, 3000):
                yield int(row), start, start + 3000, np.asarray(read[start:start + 3000])


def _batches(iterator):
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == BATCH:
            yield batch
            batch = []
    if batch:
        yield batch


def _write_group(source, label, split, stem, folders):
    total = tile_count(source, label, split)
    if not total:
        raise ValueError(f'no tiles for label {label} {split}')
    arrays = {'cafes': np.lib.format.open_memmap(folders['cafes'] / f'{stem}_preprocessed.npy', mode='w+',
                                                 dtype=np.int16, shape=(total, 3000)),
              'readcurrent': np.lib.format.open_memmap(folders['readcurrent'] / f'{stem}_preprocessed.npy',
                                                       mode='w+', dtype=np.float32, shape=(total, 3000))}
    spans = np.empty((total, 3), np.int64)
    valid = np.zeros(total, bool)
    at = 0
    for batch in _batches(tiles(source, label, split)):
        raw = np.stack([item[3] for item in batch])
        normalized, ok = readcurrent_inputs(raw)
        end = at + len(batch)
        arrays['cafes'][at:end] = raw
        arrays['readcurrent'][at:end] = normalized
        spans[at:end] = [item[:3] for item in batch]
        valid[at:end] = ok
        at = end
    if at != total:
        raise RuntimeError(f'tile count mismatch: wrote {at}, expected {total}')
    for array in arrays.values():
        array.flush()
    del arrays
    if not valid.all():
        # Keep only valid ReadCurrent tiles; rewrite the (rare) shortened array.
        path = folders['readcurrent'] / f'{stem}_preprocessed.npy'
        temporary = path.with_suffix('.tmp.npy')
        full = np.load(path, mmap_mode='r')
        kept = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=(int(valid.sum()), 3000))
        for start in range(0, total, 65536):
            chunk = valid[start:start + 65536]
            first = int(valid[:start].sum())
            kept[first:first + int(chunk.sum())] = full[start:start + 65536][chunk]
        kept.flush()
        del full, kept
        temporary.replace(path)
    for name, keep in (('cafes', np.ones(total, bool)), ('readcurrent', valid)):
        np.savez(folders[name] / f'{stem}_metadata.npz', row_index=spans[keep, 0], raw_start=spans[keep, 1],
                 raw_end=spans[keep, 2], eligible=np.ones(int(keep.sum()), bool))
    counts = np.bincount(spans[:, 0], minlength=source.n), np.bincount(spans[valid, 0], minlength=source.n)
    return dict(candidate_tiles=total, readcurrent_valid=int(valid.sum()), invalid_MAD=int((~valid).sum())), counts


def prepare_tiles(native, output):
    source = NativeSource(native)
    output = Path(output)
    for name in ('cafes', 'readcurrent'):
        if (output / name).exists():
            raise FileExistsError(output / name)
    summary = {}
    candidates = np.zeros(source.n, np.int64)
    valid_counts = np.zeros(source.n, np.int64)
    for label, group in GROUPS:
        folders = {name: output / name / 'data' / group for name in ('cafes', 'readcurrent')}
        for folder in folders.values():
            folder.mkdir(parents=True, exist_ok=True)
        for split, stem in ROLES:
            summary[f'{group}/{stem}'], (all_tiles, kept) = _write_group(source, label, split, stem, folders)
            candidates += all_tiles
            valid_counts += kept
    for name in ('cafes', 'readcurrent'):
        meta = dict(source.meta, network_labels=1 - source.labels, candidate_tile_count=candidates)
        if name == 'readcurrent':
            meta.update(valid_tile_count=valid_counts, invalid_MAD_tile_count=candidates - valid_counts)
        np.savez(output / name / 'data' / 'read_metadata.npz', **meta)
        protocol = dict(architecture='CAFES' if name == 'cafes' else 'ReadCurrent', native_source=str(Path(native).resolve()),
                        cut=1500, length=3000, tiling_fold=3, signal_units='raw_ADC',
                        normalization='six CAFES features at fit time' if name == 'cafes' else
                        'median / (1.4826*MAD); original |z|>3.5 indices replaced in ascending order',
                        training_scope='TRAIN: full-read three-offset tiling; DEV: first window raw[1500:4500]',
                        label_map={'microbe': {'project': 0, 'network': 1, 'folder': 'pos'},
                                   'host': {'project': 1, 'network': 0, 'folder': 'neg'}},
                        summary=summary)
        (output / name / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    return summary
