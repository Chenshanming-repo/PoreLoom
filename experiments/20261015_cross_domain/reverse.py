#!/usr/bin/env python3
"""Reverse cross-dataset test: R10 HG002 + Zymo models on the R10 sputum TEST, with
a per-species breakdown of target false rejects. Transfer, labelled as such; nothing is refit on the test dataset.

Source models: the main-comparison models of r10_run01 / 02 / 03 (replicate 0; same five methods as cross_domain.py).
Test sets: the TEST split of sputum_g1 / g2 / g3 (GridION 5 kHz). R10 is 4 kHz, so the primary test resamples each
sputum TEST read to 4 kHz (scipy resample_poly 4/5 on the first min(raw_length, 11250) native samples read again from
POD5, so the 4 kHz prefix keeps 9000 samples like the R10 caches; rounded to int16; raw_length x 4/5; calibration
unchanged; replay at fs 4000). Sensitivity: the native 5 kHz caches (replay at fs 5000). References on the same TEST
reads: the sputum group's own main model (in-domain), and the source model on its own R10 TEST (self).

    reverse.py make    --data <data>        (local; needs the sputum POD5)  <data>/<sputum>_at4k/test
    reverse.py run     --main <runs>/main --data <data> --out <runs>/cross_domain_rev --device cuda:0
    reverse.py report  --main <runs>/main --out <runs>/cross_domain_rev --splits <dir with g1..g3 splits.csv> [--md ..]
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_domain import METHODS, NAMES, REPO, correct, log, point, sha  # noqa: E402

SOURCES = ('r10_run01', 'r10_run02', 'r10_run03')
TARGETS = ('sputum_g1', 'sputum_g2', 'sputum_g3')
VARIANTS = dict(at4k=4000., native=5000.)
UP, DOWN, P4 = 4, 5, 9000
POD5 = Path('/path/to/poreloom_work/data/sputum_20261013/pod5')
READER_SITE = '/path/to/data-reader-env/lib/python3.10/site-packages'
SPLITS = Path('/path/to/poreloom_work/results/stream_sputum_20261013/groups')


def _read(args):
    sys.path.append(READER_SITE)
    import pod5
    name, wanted, need = args
    out = {}
    with pod5.Reader(POD5/name) as reader:
        for rec in reader.reads(selection=list(wanted)):
            out[str(rec.read_id)] = (rec.signal[:need].copy(), len(rec.signal))
    return out


def make(a):
    from concurrent.futures import ProcessPoolExecutor
    from scipy.signal import resample_poly
    need = P4*DOWN//UP                                        # 11250 native samples fill 9000 output samples
    for t in TARGETS:
        src, dst = a.data/t/'test', a.data/f'{t}_at4k'/'test'
        if (dst/'manifest.json').exists(): continue
        dst.mkdir(parents=True, exist_ok=True)
        with np.load(src/'metadata.npz', allow_pickle=False) as z: meta = {k: z[k] for k in z.files}
        pod = {r['read_id']: r['pod5'] for r in csv.DictReader(open(SPLITS/t.replace('sputum_', '')/'dataset'/'splits.csv'))}
        pos = {r: i for i, r in enumerate(meta['read_id'])}
        by = defaultdict(list)
        for r in meta['read_id']: by[pod[r]].append(r)
        out = np.zeros((len(pos), P4), np.int16); seen = np.zeros(len(pos), bool)
        with ProcessPoolExecutor(a.workers) as pool:
            for got in pool.map(_read, [(k, v, need) for k, v in sorted(by.items())]):
                for rid, (x, n) in got.items():
                    i = pos[rid]; assert not seen[i] and n == int(meta['raw_length'][i]); seen[i] = True
                    y = resample_poly(x.astype(np.float64), UP, DOWN)[:min(P4, int(np.ceil(len(x)*UP/DOWN)))]
                    out[i, :len(y)] = np.clip(np.rint(y), -32768, 32767)
        assert seen.all(), f'{t}: {int((~seen).sum())} reads not found in POD5'
        meta['raw_length'] = np.floor(meta['raw_length']*UP/DOWN).astype(meta['raw_length'].dtype)
        np.save(dst/'adc.npy', out); np.savez(dst/'metadata.npz', **meta)
        m = json.loads((src/'manifest.json').read_text())
        m.update(sampling_rate_hz=4000, prefix_samples=P4,
                 resampled=dict(source=str(POD5), up=UP, down=DOWN, method='scipy.signal.resample_poly',
                                note=f'first min(raw_length, {need}) native 5 kHz POD5 samples; raw_length x 4/5'),
                 files={'adc.npy': sha(dst/'adc.npy'), 'metadata.npz': sha(dst/'metadata.npz')},
                 written=time.strftime('%Y-%m-%d %H:%M:%S'))
        (dst/'manifest.json').write_text(json.dumps(m, indent=2))
        log('resampled', t, out.shape)


def run(a):
    from poreloom.method import baselines as B, pipeline as P
    P._setup(a.device)
    for v, fs in VARIANTS.items():
        for s in SOURCES:
            for t in TARGETS:
                for m in METHODS:
                    model = a.main/s/f'{m}_r0'
                    out = a.out/v/f'{s}_on_{t}'/f'{m}_r0'
                    if (out/'test'/'metrics.json').exists(): continue
                    out.mkdir(parents=True, exist_ok=True)
                    for name in (('stream', 'head') if m == 'poreloom' else ('fit', 'selection.json')) + ('run.json',):
                        if not (out/name).exists(): os.symlink(model/name, out/name)
                    info = json.loads((model/'run.json').read_text())
                    data = a.data/(f'{t}_at4k' if v == 'at4k' else t)
                    log('test', v, m, s, '->', t)
                    (P.test if m == 'poreloom' else B.test)(SimpleNamespace(data=data, out=out, fs=fs, device=a.device, tool=info.get('tool')))
    (a.out/'JOB_DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))


def species(splits, t):
    """read_id -> (genus, species) for target reads of sputum TEST group t (labels from the group's splits.csv)."""
    out = {}
    for r in csv.DictReader(open(splits/t.replace('sputum_', '')/'dataset'/'splits.csv')):
        if r['split'] == 'test' and r['label'] == 'target': out[r['read_id']] = (r['origin'].split('_')[0], r['origin'])
    return out


def report(a):
    rng = np.random.default_rng(20261015)
    cells, rows = [], []
    for v in VARIANTS:
        for s in SOURCES:
            for t in TARGETS:
                sp = species(a.splits, t)
                for m in METHODS:
                    d = a.out/v/f'{s}_on_{t}'/f'{m}_r0'
                    if not (d/'test'/'metrics.json').exists(): continue
                    c = dict(variant=v, src=s, tgt=t, method=m, **point(d, m),
                             self=point(a.main/s/f'{m}_r0', m), indomain=point(a.main/t/f'{m}_r0', m))
                    ids, x, _, host = correct(d, m); ids0, x0, _, _ = correct(a.main/t/f'{m}_r0', m)
                    assert np.array_equal(ids, ids0)
                    diff = x - x0; idx = rng.integers(0, len(diff), (2000, len(diff)))
                    c.update(d_indomain=float(diff.mean()), d_indomain_ci=np.percentile(diff[idx].mean(1), [2.5, 97.5]).tolist())
                    cells.append(c)
                    g = np.array([sp.get(r, ('', ''))[0] for r in ids])
                    for k in np.unique(g[~host]):
                        sel = (g == k) & ~host
                        rows.append(dict(variant=v, src=s, tgt=t, method=m, genus=str(k), n=int(sel.sum()),
                                         frr=float(1 - x[sel].mean()), frr_indomain=float(1 - x0[sel].mean())))
    f = lambda x: f'{100*x:.2f}'
    L = ['# 反方向跨数据集测试：R10 HG002 + Zymo 模型 → R10 痰液（迁移，不在测试集上重新拟合）', '',
         '源模型 = r10_run01/02/03 的主实验模型（权重、β / 阈值原样）；测试集 = sputum_g1/g2/g3 的 TEST（GridION 5 kHz）。'
         '主结果把痰液 read 从 POD5 重新读取后重采样到 4 kHz（与 R10 一致）；"原始 5 kHz" 为敏感性分析。0.4 s，max-ACC 规则。'
         '每格为 3 个源 × 3 个测试集 = 9 格的均值。', '',
         '"自测" = 源模型在自己的 R10 TEST 上；"痰液域内" = 测试集自己的痰液模型在同一批 read 上（5 kHz）。'
         'Δ域内 = 跨数据集 − 痰液域内（同一批 read 配对）。', '']
    for v, title in (('at4k', '主结果：痰液重采样到 4 kHz'), ('native', '敏感性：痰液原始 5 kHz')):
        L += [f'## {title}', '', '| 方法 | 自测 ACC % | 跨数据集 ACC % | 下降（vs 自测）pp | 痰液域内 ACC % | Δ域内 pp | 跨数据集 target 误拒 % | host 排出 % | host 占孔 s |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for m in METHODS:
            cs = [c for c in cells if c['variant'] == v and c['method'] == m]
            if not cs: continue
            mean = lambda g: float(np.mean([g(c) for c in cs]))
            L.append(f"| {NAMES[m]} (n={len(cs)}) | {f(mean(lambda c: c['self']['acc']))} | {f(mean(lambda c: c['acc']))} | "
                     f"{100*mean(lambda c: c['acc'] - c['self']['acc']):+.2f} | {f(mean(lambda c: c['indomain']['acc']))} | "
                     f"{100*mean(lambda c: c['d_indomain']):+.2f} | {f(mean(lambda c: c['target_frr']))} | {f(mean(lambda c: c['host_reject']))} | "
                     f"{mean(lambda c: c['host_pore']):.2f} |")
        L.append('')
    L += ['## 逐属 target 误拒 %（主结果 4 kHz；9 格合并；括号 = 痰液域内模型在同一批 read 上）', '',
          '| 属 | read 数 | ' + ' | '.join(NAMES[m] for m in METHODS) + ' |', '|---|---:|' + '---:|'*len(METHODS)]
    R = [r for r in rows if r['variant'] == 'at4k']
    tot = defaultdict(int)
    for r in R:
        if r['method'] == 'poreloom': tot[r['genus']] += r['n']
    for k in sorted(tot, key=lambda k: -tot[k]):
        if tot[k] < 90: continue
        cell = []
        for m in METHODS:
            rs = [r for r in R if r['genus'] == k and r['method'] == m]; n = sum(r['n'] for r in rs)
            cell.append(f"{100*sum(r['frr']*r['n'] for r in rs)/n:.1f} ({100*sum(r['frr_indomain']*r['n'] for r in rs)/n:.1f})" if n else '–')
        L.append(f'| {k} | {tot[k]//3} | ' + ' | '.join(cell) + ' |')
    L += ['', 'read 数 = 3 个测试组合计（每条 read 被 3 个源模型各测一次，表中按 1 次计）；只列合计 ≥ 30 条的属。', '']
    text = '\n'.join(L) + '\n'; print(text)
    (a.out/'report.json').write_text(json.dumps(dict(cells=cells, genus=rows), indent=2))
    if a.md: a.md.write_text(text)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('make', 'run', 'report'))
    ap.add_argument('--main', type=Path); ap.add_argument('--data', type=Path); ap.add_argument('--out', type=Path)
    ap.add_argument('--splits', type=Path, default=SPLITS); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--md', type=Path); ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    globals()[a.stage](a)
