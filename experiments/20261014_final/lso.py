#!/usr/bin/env python3
"""Leave-species-out (LSO) on sputum group g1: four held-out taxa, every model retrained from scratch per fold.

Scope: sputum g1 first, then g2 and g3 as replicate groups (same four folds); methods PoreLoom (poreloom.method) and the single-window baselines
SquiggleNet, NanoDeep, ReadCurrent (poreloom.method.baselines); no CAFES.
Folds fixed a priori (same design as the stopped 2026-10-14 sputum LOSO, experiments/20261013_stream_sputum/loso.py):
Prevotella, Streptococcus, Fusobacterium (genera) and Actinomycetota (phylum: Rothia, Schaalia, Actinomyces,
Lancefieldella). Per fold: every read of the held-out taxon is removed from TRAIN and DEV; in each split and barcode the
same number of host reads (sha256 order) is removed to keep host = target per barcode; TEST is g1's TEST unchanged.
Reference: the main-comparison g1 models of each method (same TEST reads, same DEV-B max-ACC rule, 0.4 s).
Primary: held-out taxon TEST target command FRR, fold minus main, paired bootstrap (5000, seed 20261010);
secondary: host command reject change.

    lso.py make   [--group g1] [--splits <groups>/gN/dataset/splits.csv] [--prefix <gN prefix root>] [--data <out root>]
                  fold prefix caches in <data>/sputum_<group>_lso_<fold>/{train,dev} (test -> ../sputum_<group>/test)
    lso.py report --runs <runs root holding main/ and lso/> [--group g1 ...]
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

SE = Path('/path/to/poreloom_work')
G1 = SE/'results/stream_sputum_20261013/groups/g1'
DATA = SE/'results/final_20261014/data'
FOLDS = {'Prevotella': ['Prevotella'], 'Streptococcus': ['Streptococcus'], 'Fusobacterium': ['Fusobacterium'],
         'Actinomycetota': ['Rothia', 'Schaalia', 'Actinomyces', 'Lancefieldella']}
METHODS = ('poreloom', 'readcurrent', 'nanodeep_upstream', 'squigglenet', 'nanodeep')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 23), b''): h.update(blk)
    return h.hexdigest()


def h8(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def genus(origin):
    return str(origin).split('_')[0]


def make(group='g1', splits_csv=None, prefix=None, data=DATA):
    import pandas as pd
    G = G1.parent/group
    splits_csv = Path(splits_csv or G/'dataset/splits.csv'); prefix = Path(prefix or G/'stream/prefix'); data = Path(data)
    data.mkdir(parents=True, exist_ok=True)
    if not (data/f'sputum_{group}').exists(): (data/f'sputum_{group}').symlink_to(prefix)
    splits = pd.read_csv(splits_csv, dtype={'channel': str})
    splits['genus'] = splits.origin.map(genus)
    for fold, taxa in FOLDS.items():
        root = data/f'sputum_{group}_lso_{fold}'
        if (root/'plan.json').exists(): continue
        removed, keep = {}, {}
        for split in ('train', 'dev'):
            part = splits[splits.split == split]
            drop_t = (part.label == 'target') & part.genus.isin(taxa)
            out = [part[~drop_t & (part.label == 'target')]]
            for b, n in part[drop_t].groupby('barcode').size().items(): removed[f'{split}/{b}'] = int(n)
            for b, g in part[part.label == 'host'].groupby('barcode'):
                order = sorted(g.read_id, key=lambda r: h8('loso|' + r))
                out.append(g[g.read_id.isin(set(order[removed.get(f'{split}/{b}', 0):]))])
            keep[split] = pd.concat(out)
        for split in ('train', 'dev'):
            dest, src = root/split, prefix/split
            dest.mkdir(parents=True, exist_ok=True)
            with np.load(src/'metadata.npz', allow_pickle=False) as z: meta = {k: z[k] for k in z.files}
            rows = np.flatnonzero(np.isin(meta['read_id'], keep[split].read_id.values))
            assert len(rows) == len(keep[split])
            adc = np.load(src/'adc.npy', mmap_mode='r')
            out = np.lib.format.open_memmap(dest/'adc.npy', mode='w+', dtype=np.int16, shape=(len(rows), adc.shape[1]))
            for s in range(0, len(rows), 8192): out[s:s + 8192] = adc[rows[s:s + 8192]]
            out.flush(); del out
            np.savez(dest/'metadata.npz', **{k: v[rows] for k, v in meta.items()})
            manifest = json.loads((src/'manifest.json').read_text())
            manifest.update(reads=len(rows), lso_fold=fold, parent=str(src), files={n: sha(dest/n) for n in ('adc.npy', 'metadata.npz')})
            (dest/'manifest.json').write_text(json.dumps(manifest, indent=2))
        if not (root/'test').exists(): (root/'test').symlink_to(f'../sputum_{group}/test')
        counts = {f'{s}/{"target" if t else "host"}': int(n) for s, d in keep.items() for t, n in (d.label == 'target').value_counts().items()}
        (root/'plan.json').write_text(json.dumps(dict(fold=fold, taxa=taxa, removed_targets_by_split_barcode=removed, counts=counts,
                                                      group=group, source=str(splits_csv), source_sha256=sha(splits_csv),
                                                      written=time.strftime('%Y-%m-%d %H:%M:%S')), indent=2))
        log(fold, counts, 'removed targets', sum(removed.values()))


def outcomes(run, method):
    return np.load(run/'test'/('outcomes_0.4_nominal.npz' if method == 'poreloom' else 'outcomes_max_acc_0.4_measured.npz'))


def report(runs, groups=('g1',)):
    import pandas as pd
    rows = []
    for group, fold in ((gr, f) for gr in groups for f in FOLDS):
        taxa = FOLDS[fold]
        splits = pd.read_csv(G1.parent/group/'dataset/splits.csv'); g = splits.set_index('read_id').origin.map(genus)
        rng = np.random.default_rng(20261010)
        for m in METHODS:
            if not (runs/'lso'/f'sputum_{group}_lso_{fold}'/f'{m}_r0'/'JOB_DONE').exists(): log('skip', group, fold, m, '(not finished)'); continue
            a = outcomes(runs/'lso'/f'sputum_{group}_lso_{fold}'/f'{m}_r0', m); b = outcomes(runs/'main'/f'sputum_{group}'/f'{m}_r0', m)
            assert np.array_equal(a['read_id'], b['read_id'])
            y = a['is_target']; held = y & g.reindex(a['read_id']).isin(taxa).values; host = ~y
            d = a['command_reject'][held].astype(float) - b['command_reject'][held].astype(float)
            ci = np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)], [2.5, 97.5])
            dh = a['command_reject'][host].astype(float) - b['command_reject'][host].astype(float)
            cih = np.percentile([dh[rng.integers(0, len(dh), len(dh))].mean() for _ in range(2000)], [2.5, 97.5])
            rows.append(dict(group=group, fold=fold, method=m, n_held=int(held.sum()), frr_main=float(b['command_reject'][held].mean()),
                             frr_fold=float(a['command_reject'][held].mean()), dfrr=float(d.mean()), dfrr_ci=ci.tolist(),
                             host_main=float(b['command_reject'][host].mean()), host_fold=float(a['command_reject'][host].mean()),
                             dhost=float(dh.mean()), dhost_ci=cih.tolist()))
    (runs/'lso'/'analysis.json').write_text(json.dumps(rows, indent=2))
    name = dict(nanodeep='NanoDeep 项目配方', nanodeep_upstream='NanoDeep 原作者配方', readcurrent='ReadCurrent', squigglenet='SquiggleNet', poreloom='PoreLoom')
    lines = [f"# LSO：痰液 {'/'.join(groups)}，4 个留出类群，全部从零训练", '',
             'fold 模型 − 主比较（全类群）模型，同一批 TEST read 配对；各自 DEV-B 最大 ACC 工作点，0.4 s。', '',
             '| 组 | 留出 | 方法 | n | 误拒率 主 → fold % | Δ误拒率 pp [95% CI] | host 拒绝率 主 → fold % | Δhost pp [95% CI] |',
             '|---|---|---|---:|---|---:|---|---:|']
    for r in rows:
        lines.append(f"| {r['group']} | {r['fold']} | {name[r['method']]} | {r['n_held']} | {100*r['frr_main']:.1f} → {100*r['frr_fold']:.1f} | "
                     f"{100*r['dfrr']:+.1f} [{100*r['dfrr_ci'][0]:+.1f}, {100*r['dfrr_ci'][1]:+.1f}] | "
                     f"{100*r['host_main']:.1f} → {100*r['host_fold']:.1f} | {100*r['dhost']:+.1f} [{100*r['dhost_ci'][0]:+.1f}, {100*r['dhost_ci'][1]:+.1f}] |")
    (runs/'lso'/'summary_zh.md').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('make', 'report'))
    ap.add_argument('--runs', type=Path); ap.add_argument('--group', nargs='+', default=['g1'])
    ap.add_argument('--splits', type=Path); ap.add_argument('--prefix', type=Path); ap.add_argument('--data', type=Path, default=DATA)
    a = ap.parse_args()
    if a.stage == 'make':
        for g in a.group: make(g, a.splits, a.prefix, a.data)
    else: report(a.runs, a.group)
