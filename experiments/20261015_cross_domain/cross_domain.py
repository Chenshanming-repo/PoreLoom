#!/usr/bin/env python3
"""Cross-dataset test: R10 sputum models on the R10 HG002 + Zymo mix. Transfer, labelled as such; nothing is refit on the test dataset.

Source models: the main-comparison models of sputum_g1 / g2 / g3 (replicate 0; PoreLoom and the NanoDeep / NanoDeep
upstream / ReadCurrent / SquiggleNet baselines; weights, C-p beta or DEV-B threshold and measured service all from the
sputum group). Test sets: the TEST split of r10_run01 / 02 / 03 (20000 reads each, host HG002 vs target Zymo D6322,
PromethION 4 kHz). Sputum is 5 kHz, so the primary test resamples each R10 TEST read to 5 kHz (scipy resample_poly
5/4 on the first min(raw_length, 7200) ADC samples, rounded to int16; raw_length x 5/4; calibration unchanged; replay
at fs 5000) so a sample spans the same time as in training. Sensitivity: the native 4 kHz reads (replay at fs 4000).
References on the same TEST reads: the R10 cohort's own main model (in-domain), and the source model on its own sputum
TEST (self). Primary point: 0.4 s, PoreLoom nominal service / baselines measured service, max-ACC rule.

    cross_domain.py make   --data <data>                      <data>/<r10>_at5k/test (resampled cache)
    cross_domain.py run    --main <runs>/main --data <data> --out <runs>/cross_domain --device cuda:0
    cross_domain.py report --main <runs>/main --out <runs>/cross_domain [--md summary_zh.md]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'src'))

SOURCES = ('sputum_g1', 'sputum_g2', 'sputum_g3')
TARGETS = ('r10_run01', 'r10_run02', 'r10_run03')
METHODS = ('poreloom', 'nanodeep_upstream', 'readcurrent', 'squigglenet', 'nanodeep')
VARIANTS = dict(at5k=5000., native=4000.)
NAMES = dict(poreloom='PoreLoom', nanodeep='NanoDeep（项目配方）', nanodeep_upstream='NanoDeep（作者配方）',
             readcurrent='ReadCurrent', squigglenet='SquiggleNet')
UP, DOWN = 5, 4


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''): h.update(block)
    return h.hexdigest()


def make(a):
    from scipy.signal import resample_poly
    for t in TARGETS:
        src, dst = a.data/t/'test', a.data/f'{t}_at5k'/'test'
        if (dst/'manifest.json').exists(): continue
        dst.mkdir(parents=True, exist_ok=True)
        adc = np.load(src/'adc.npy', mmap_mode='r')
        with np.load(src/'metadata.npz', allow_pickle=False) as z: meta = {k: z[k] for k in z.files}
        n, P = adc.shape
        need = P*DOWN//UP                                     # input samples that fill P output samples
        out = np.zeros((n, P), np.int16)
        for i in range(n):
            k = int(min(meta['raw_length'][i], need))
            y = resample_poly(adc[i, :k].astype(np.float64), UP, DOWN)[:min(P, int(np.ceil(k*UP/DOWN)))]
            out[i, :len(y)] = np.clip(np.rint(y), -32768, 32767)
        meta['raw_length'] = np.floor(meta['raw_length']*UP/DOWN).astype(meta['raw_length'].dtype)
        np.save(dst/'adc.npy', out); np.savez(dst/'metadata.npz', **meta)
        m = json.loads((src/'manifest.json').read_text())
        m.update(sampling_rate_hz=5000, resampled=dict(source=str(src), up=UP, down=DOWN, method='scipy.signal.resample_poly',
                                                       note='first min(raw_length, P*4/5) native samples; raw_length x 5/4'),
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
                    data = a.data/(f'{t}_at5k' if v == 'at5k' else t)
                    log('test', v, m, s, '->', t)
                    (P.test if m == 'poreloom' else B.test)(SimpleNamespace(data=data, out=out, fs=fs, device=a.device, tool=info.get('tool')))
    (a.out/'JOB_DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))


def outcome_file(m):
    return 'outcomes_0.4_nominal.npz' if m == 'poreloom' else 'outcomes_max_acc_0.4_measured.npz'


def point(d, m):
    rows = json.loads((d/'test'/'metrics.json').read_text())['rows']
    if m == 'poreloom':
        r = next(r for r in rows if r['interval'] == .4 and r['service'] == 'nominal' and r['group'] == 'pooled'); hr = r['host_command_reject']
    else:
        r = next(r for r in rows if r['rule'] == 'max_acc' and r['interval'] == .4 and r['service'] == 'measured' and r['group'] == 'pooled')
        hr = r['host_command_reject_rate']
    return dict(acc=r['acc'], host_reject=hr, target_frr=r['target_command_frr'], host_pore=r['host_mean_pore_seconds'])


def correct(d, m):
    with np.load(d/'test'/outcome_file(m)) as z:
        y = z['is_target'].astype(bool)
        return z['read_id'].astype(str), np.where(y, ~z['command_reject'], z['command_reject']).astype(float), z['pore_seconds'], ~y


def report(a):
    rng = np.random.default_rng(20261015)
    cells = []
    for v in VARIANTS:
        for s in SOURCES:
            for t in TARGETS:
                for m in METHODS:
                    d = a.out/v/f'{s}_on_{t}'/f'{m}_r0'
                    if not (d/'test'/'metrics.json').exists(): continue
                    c = dict(variant=v, src=s, tgt=t, method=m, **point(d, m),
                             self=point(a.main/s/f'{m}_r0', m), indomain=point(a.main/t/f'{m}_r0', m))
                    ids, x, _, _ = correct(d, m); ids0, x0, _, _ = correct(a.main/t/f'{m}_r0', m)
                    assert np.array_equal(ids, ids0)
                    diff = x - x0; idx = rng.integers(0, len(diff), (2000, len(diff)))
                    c.update(d_indomain=float(diff.mean()), d_indomain_ci=np.percentile(diff[idx].mean(1), [2.5, 97.5]).tolist())
                    cells.append(c)
    vs = {}
    for v in VARIANTS:
        for s in SOURCES:
            for t in TARGETS:
                dp = a.out/v/f'{s}_on_{t}'/'poreloom_r0'
                for m in METHODS[1:]:
                    dm = a.out/v/f'{s}_on_{t}'/f'{m}_r0'
                    if not ((dp/'test'/'metrics.json').exists() and (dm/'test'/'metrics.json').exists()): continue
                    ids, xp, pp, host = correct(dp, 'poreloom'); ids2, xm, pm, _ = correct(dm, m); assert np.array_equal(ids, ids2)
                    diff = xp - xm; idx = rng.integers(0, len(diff), (2000, len(diff)))
                    vs[(v, s, t, m)] = dict(dacc=float(diff.mean()), ci=np.percentile(diff[idx].mean(1), [2.5, 97.5]).tolist(),
                                            dpore=float((pp - pm)[host].mean()))
    f = lambda x: f'{100*x:.2f}'
    L = ['# 跨数据集测试：R10 痰液模型 → R10 HG002 + Zymo 拼接集（迁移，不在测试集上重新拟合）', '',
         '源模型 = sputum_g1/g2/g3 的主实验模型（权重、β / 阈值原样）；测试集 = r10_run01/02/03 的 TEST（各 20000 条，host = HG002，'
         'target = Zymo D6322，PromethION 4 kHz）。主结果把 R10 read 重采样到 5 kHz（与痰液一致）；"原始 4 kHz" 为敏感性分析。'
         '0.4 s，max-ACC 规则。每格为 3 个源 × 3 个测试集 = 9 格的均值。', '',
         '"自测" = 源模型在自己的痰液 TEST 上；"R10 域内" = 测试集自己的 R10 模型在同一批 read 上（4 kHz）。'
         'Δ域内 = 跨数据集 − R10 域内（同一批 read 配对）。', '']
    for v, title in (('at5k', '主结果：R10 重采样到 5 kHz'), ('native', '敏感性：R10 原始 4 kHz')):
        L += [f'## {title}', '', '| 方法 | 自测 ACC % | 跨数据集 ACC % | 下降（vs 自测）pp | R10 域内 ACC % | Δ域内 pp | 跨数据集 target 误拒 % | host 排出 % | host 占孔 s |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for m in METHODS:
            cs = [c for c in cells if c['variant'] == v and c['method'] == m]
            if not cs: continue
            mean = lambda g: float(np.mean([g(c) for c in cs]))
            L.append(f"| {NAMES[m]} (n={len(cs)}) | {f(mean(lambda c: c['self']['acc']))} | {f(mean(lambda c: c['acc']))} | "
                     f"{100*mean(lambda c: c['acc'] - c['self']['acc']):+.2f} | {f(mean(lambda c: c['indomain']['acc']))} | "
                     f"{100*mean(lambda c: c['d_indomain']):+.2f} | {f(mean(lambda c: c['target_frr']))} | {f(mean(lambda c: c['host_reject']))} | "
                     f"{mean(lambda c: c['host_pore']):.2f} |")
        L += ['', f'PoreLoom − 工具（{title}，同一格同一批 read 配对）：', '',
              '| 工具 | ΔACC pp | PoreLoom 显著更好 / 显著更差的格数 | Δhost 占孔 s |', '|---|---:|---:|---:|']
        for m in METHODS[1:]:
            xs = [x for (vv, s, t, mm), x in vs.items() if vv == v and mm == m]
            if not xs: continue
            L.append(f"| {NAMES[m]} | {100*np.mean([x['dacc'] for x in xs]):+.2f} | {sum(x['ci'][0] > 0 for x in xs)} / "
                     f"{sum(x['ci'][1] < 0 for x in xs)} (共 {len(xs)}) | {np.mean([x['dpore'] for x in xs]):+.2f} |")
        L.append('')
    L += ['## 逐格 ACC %（主结果 5 kHz；行 = 源痰液组，列 = R10 测试集）', '']
    for m in METHODS:
        L += [f'**{NAMES[m]}**', '', '| 源 ↓ / 测试 → | ' + ' | '.join(t.replace('r10_', '') for t in TARGETS) + ' |', '|---|' + '---:|'*len(TARGETS)]
        for s in SOURCES:
            row = [next((f(c['acc']) for c in cells if (c['variant'], c['src'], c['tgt'], c['method']) == ('at5k', s, t, m)), '–') for t in TARGETS]
            L.append(f"| {s.replace('sputum_', '')} | " + ' | '.join(row) + ' |')
        L.append('')
    text = '\n'.join(L) + '\n'; print(text)
    (a.out/'report.json').write_text(json.dumps(dict(cells=cells, vs=[dict(variant=k[0], src=k[1], tgt=k[2], method=k[3], **x) for k, x in vs.items()]), indent=2))
    if a.md: a.md.write_text(text)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('make', 'run', 'report'))
    ap.add_argument('--main', type=Path); ap.add_argument('--data', type=Path); ap.add_argument('--out', type=Path)
    ap.add_argument('--md', type=Path); ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    globals()[a.stage](a)
