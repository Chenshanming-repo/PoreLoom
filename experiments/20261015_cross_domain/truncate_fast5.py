#!/usr/bin/env python3
"""Copy the wanted reads of multi-read FAST5 files with Raw/Signal cut to the first N samples (uncompressed), so a
species-labelling basecall only decodes ~N/10 bases per read (zymo_species.py). Run with
HDF5_PLUGIN_PATH=<dir holding libvbz_hdf_plugin.so> to read VBZ-compressed signal.

    truncate_fast5.py --src <fast5 dir> --ids <ids.txt> --dst <dir> [--samples 40000]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py


def one(args):
    src, dst, ids, n = args
    k = 0
    with h5py.File(src, 'r') as f, h5py.File(dst, 'w') as g:
        for key, val in f.attrs.items(): g.attrs[key] = val
        for name in f:
            if name[5:] not in ids: continue
            r, w = f[name], g.create_group(name)
            for key, val in r.attrs.items(): w.attrs[key] = val
            for child in r:
                if child != 'Raw': f.copy(r[child], w, name=child)
            raw = w.create_group('Raw')
            for key, val in r['Raw'].attrs.items(): raw.attrs[key] = val
            ds = raw.create_dataset('Signal', data=r['Raw']['Signal'][:n])   # reads only the leading chunks
            for key, val in r['Raw']['Signal'].attrs.items(): ds.attrs[key] = val
            raw.attrs['duration'] = ds.shape[0]; k += 1
    return src.name, k


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, required=True); ap.add_argument('--ids', type=Path, required=True)
    ap.add_argument('--dst', type=Path, required=True); ap.add_argument('--samples', type=int, default=40000)
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    ids = set(a.ids.read_text().split()); a.dst.mkdir(parents=True, exist_ok=True)
    jobs = [(p, a.dst/p.name, ids, a.samples) for p in sorted(a.src.glob('*.fast5'))]
    with ProcessPoolExecutor(a.workers) as pool:
        for name, k in pool.map(one, jobs): print(name, k, flush=True)


if __name__ == '__main__':
    main()
