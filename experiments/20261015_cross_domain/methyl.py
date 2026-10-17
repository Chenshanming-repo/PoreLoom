#!/usr/bin/env python3
"""Library-chemistry check behind the asymmetric cross-dataset result: the R10 mix is
native DNA (SQK-LSK114) and the sputum run is PCR-amplified (SQK-RPB114.24), so CpG 5mC should survive only in the R10
reads. 2000 TEST reads per group (hg002 / zymo from r10_run01, sputum_human / sputum_microbe from sputum_g1), first
40000 samples, Guppy 6.5.7 5mC-CpG hac models (4 kHz: dna_r10.4.1_e8.2_400bps_modbases_5mc_cg_hac_prom; 5 kHz:
dna_r10.4.1_e8.2_400bps_5khz_modbases_5mc_cg_hac), then the share of CpG calls with P(5mC) >= 0.5 per group.

    methyl.py dump   --ids hg002_ids.txt --out hg002.npz          (.venv-raw: pyslow5; HG002 BLOW5 -> npz)
    methyl.py fast5  --npz hg002.npz --template <zymo fast5> --out <dir>/hg002.fast5    (h5py)
    methyl.py pod5   --ids sputum_human_ids.txt --out <dir>                            (data-reader env: pod5 subset)
    methyl.py parse  --bam <dir> [<dir> ...] --md methyl_zh.md                          (pysam)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

N = 40000
BLOW5 = '/path/to/poreloom_work/data/repeated_benchmark_20260929/human/hg2_subsample.blow5'
HG002_RUN = 'dc60b20f5078b3546ded810fb828b49c438fbd89'
SPLITS = Path('/path/to/poreloom_work/results/stream_sputum_20261013/groups/g1/dataset/splits.csv')
POD5 = Path('/path/to/poreloom_work/data/sputum_20261013/pod5')
READER_SITE = '/path/to/data-reader-env/lib/python3.10/site-packages'


def dump(a):
    import pyslow5
    ids = a.ids.read_text().split(); r = pyslow5.Open(BLOW5, 'r')
    sig, meta = [], []
    for rid in ids:
        x = r.get_read(rid, pA=False, aux='all')
        s = np.asarray(x['signal'], np.int16)[:N]; sig.append(s)
        meta.append((rid, int(x['channel_number']), float(x['digitisation']), float(x['offset']), float(x['range']),
                     float(x['sampling_rate']), int(x['start_time']), int(x['len_raw_signal'])))
    np.savez(a.out, signal=np.array(sig, dtype=object), meta=np.array(meta, dtype=object), allow_pickle=True)
    print('dumped', len(ids))


def fast5(a):
    import h5py
    z = np.load(a.npz, allow_pickle=True)
    with h5py.File(a.template, 'r') as t, h5py.File(a.out, 'w') as g:
        for k, v in t.attrs.items(): g.attrs[k] = v
        tpl = t[next(iter(t))]
        for s, (rid, ch, dig, off, rng, rate, start, n) in zip(z['signal'], z['meta']):
            w = g.create_group(f'read_{rid}')
            w.attrs['run_id'] = HG002_RUN.encode(); w.attrs['pore_type'] = b'not_set'
            for sub in ('context_tags', 'tracking_id'):
                t.copy(tpl[sub], w, name=sub)
            w['tracking_id'].attrs['run_id'] = HG002_RUN.encode()
            c = w.create_group('channel_id')
            c.attrs.update(dict(channel_number=str(ch).encode(), digitisation=dig, offset=off, range=rng, sampling_rate=rate))
            raw = w.create_group('Raw')
            raw.attrs.update(dict(duration=len(s), read_id=rid.encode(), read_number=0, start_mux=1, start_time=start,
                                  median_before=200., end_reason=1))
            raw.create_dataset('Signal', data=np.asarray(s, np.int16))
    print('wrote', len(z['meta']))


def pod5_subset(a):
    sys.path.append(READER_SITE)
    import pod5
    ids = set(a.ids.read_text().split())
    files = sorted({r['pod5'] for r in csv.DictReader(open(SPLITS)) if r['read_id'] in ids})
    a.out.mkdir(parents=True, exist_ok=True)
    with pod5.Writer(a.out/'subset.pod5') as w:
        for f in files:
            with pod5.Reader(POD5/f) as r:
                for rec in r.reads(selection=[i for i in ids], missing_ok=True):
                    w.add_read(rec.to_read())
    print('pod5 files', len(files))


def parse(a):
    import pysam
    rows = {}
    for d in a.bam:
        meth = tot = 0; per = []
        for b in sorted(Path(d).rglob('*.bam')):
            with pysam.AlignmentFile(str(b), check_sq=False) as f:
                for r in f:
                    mb = r.modified_bases
                    if not mb: continue
                    calls = [q for k, v in mb.items() if k[2] == 'm' for _, q in v]
                    if not calls: continue
                    q = np.asarray(calls)/255.; m = int((q >= .5).sum())
                    meth += m; tot += len(q); per.append(m/len(q))
        rows[Path(d).name] = dict(reads=len(per), cpg_calls=tot, frac_5mc=meth/max(tot, 1),
                                  median_read_frac=float(np.median(per)) if per else float('nan'))
    L = ['# CpG 5mC 检测：R10 混合集（天然 DNA，LSK114）vs 痰液（PCR，RPB114）', '',
         'Guppy 6.5.7 5mC-CpG hac 模型，每组 2000 条 TEST read 的前 40000 个样本；甲基化比例 = P(5mC) ≥ 0.5 的 CpG 调用占比。', '',
         '| 组 | read 数 | CpG 调用数 | 5mC 比例 % | 每条 read 中位数 % |', '|---|---:|---:|---:|---:|']
    for k, v in rows.items():
        L.append(f"| {k} | {v['reads']} | {v['cpg_calls']} | {100*v['frac_5mc']:.1f} | {100*v['median_read_frac']:.1f} |")
    text = '\n'.join(L) + '\n'; print(text)
    if a.md: a.md.write_text(text); a.md.with_suffix('.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('dump', 'fast5', 'pod5', 'parse'))
    ap.add_argument('--ids', type=Path); ap.add_argument('--out', type=Path); ap.add_argument('--npz', type=Path)
    ap.add_argument('--template', type=Path); ap.add_argument('--bam', type=Path, nargs='+'); ap.add_argument('--md', type=Path)
    a = ap.parse_args()
    dict(dump=dump, fast5=fast5, pod5=pod5_subset, parse=parse)[a.stage](a)
