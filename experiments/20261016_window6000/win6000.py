#!/usr/bin/env python3
"""Main-experiment baseline check: does PoreLoom win only because it may look at more signal?

Replaces the E2 cascade (20261014_final/cascade.py: 3000-sample models re-scored on shifted windows) with baselines
TRAINED and TESTED on one 6000-sample window raw[1500:7500] = PoreLoom's whole signal (1500 skip + 6000, budget end
7500). Every tool (ReadCurrent, NanoDeep, SquiggleNet) uses the same window start 1500 (NanoDeep's native start is
1000; 1500 keeps the input identical to PoreLoom's). Everything else is the main-run baseline unchanged
(poreloom.method.baselines): native architecture from the seeded random init, the tool's own window normalization
over the 6000 samples, ReadCurrent / SquiggleNet project recipe, NanoDeep authors' recipe ('nanodeep_upstream', as in
the main table), DEV-B max-ACC threshold, one decision at the first 0.4 s chunk arrival holding raw 7500 + measured
single-window service; reads shorter than 7500 get no command (kept).

    win6000.py run    --tool T [--recipe upstream] --data <data>/<ds> --fs F --rep r --out <runs>/win6000/<ds>/<T>_r<r>
    win6000.py report --root <runs>/win6000 --main <runs>/main [--e2 <runs>/e2] [--md summary_zh.md]

Report: per unit, PoreLoom (main locked point, 0.4 s nominal) vs the tool at 3000 (main) / cascade (E2) / 6000 (this):
ACC, host reject, target FRR, host pore time; paired bootstrap over TEST reads (2000, seed 20261016) of
PoreLoom - W6000 for ACC and host pore time.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'src'))

WINDOW, START = 6000, 1500
TOOLS = ('readcurrent', 'nanodeep_upstream', 'squigglenet')
NAMES = dict(readcurrent='ReadCurrent', nanodeep_upstream='NanoDeep', squigglenet='SquiggleNet')


def run(a):
    from poreloom.baselines import preprocessing as pp
    from poreloom.method import baselines
    pp.WINDOW = WINDOW
    for c in pp.CONTRACTS.values():       # shared dict objects: native / native_policy / baselines see the change
        c.update(raw_start=START, raw_end=START + WINDOW)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/'window.json').write_text(json.dumps(dict(window=WINDOW, raw_start=START, raw_end=START + WINDOW)))
    baselines.main(['--tool', a.tool, '--recipe', a.recipe, '--data', str(a.data), '--out', str(a.out), '--fs', str(a.fs),
                    '--replicate', str(a.rep), '--device', a.device])


def pooled(path, rule=None, service='nominal'):
    rows = json.loads(path.read_text())['rows']
    return next(r for r in rows if r['interval'] == .4 and r['service'] == service and r['group'] == 'pooled'
                and (rule is None or r['rule'] == rule))


def outcome(path):
    z = np.load(path)
    return z['read_id'], (z['command_reject'] != z['is_target']), z['pore_seconds'], ~z['is_target']


def boot(pl, tw, n=2000, seed=20261016):
    (ida, ca, pa, host), (idb, cb, pb, _) = pl, tw
    order = {r: i for i, r in enumerate(idb)}; j = np.array([order[r] for r in ida])
    cb, pb = cb[j], pb[j]
    rng = np.random.default_rng(seed); N = len(ida); h = np.flatnonzero(host)
    dacc = ca.astype(float) - cb; dpore = pa - pb
    acc = [dacc[rng.integers(0, N, N)].mean() for _ in range(n)]
    pore = [dpore[h[rng.integers(0, len(h), len(h))]].mean() for _ in range(n)]
    return dict(dacc=float(dacc.mean()), dacc_ci=list(np.percentile(acc, [2.5, 97.5])),
                dpore=float(dpore[h].mean()), dpore_ci=list(np.percentile(pore, [2.5, 97.5])))


def report(a):
    fmt = lambda r: f"{100*r['acc']:.1f} / {r['host_mean_pore_seconds']:.2f}" if r else '–'
    lines = ['# 6000 单窗口 baseline vs PoreLoom（TEST，0.4 s，DEV-B 最大 ACC）', '',
             '每格 ACC % / host 平均占孔 s。W3000 = 主实验单窗口；级联 = E2；W6000 = 本实验（raw[1500:7500] 训练+测试，'
             '与 PoreLoom 看到的信号相同）。Δ = PoreLoom − W6000，配对 bootstrap 95% CI（2000 次）。', '',
             '| 数据 | 组 | PoreLoom | 工具 | W3000 | 级联 | W6000 | W6000 host 拒绝 / 目标 FRR % | ΔACC pp | Δhost 占孔 s |',
             '|---|---|---|---|---|---|---|---|---|---|']
    out, wins = [], {}
    for ds in sorted(p for p in a.root.iterdir() if p.is_dir()):
        for rep in range(3):
            plp = a.main/ds.name/f'poreloom_r{rep}'/'test'
            if not (plp/'metrics.json').exists(): continue
            pl = pooled(plp/'metrics.json'); plo = outcome(plp/'outcomes_0.4_nominal.npz')
            for tool in TOOLS:
                wp = ds/f'{tool}_r{rep}'/'test'
                if not (wp/'metrics.json').exists(): continue
                w = pooled(wp/'metrics.json', 'max_acc', 'measured')
                b = boot(plo, outcome(wp/'outcomes_max_acc_0.4_measured.npz'))
                m3p = a.main/ds.name/f'{tool}_r{rep}'/'test'/'metrics.json'
                m3 = pooled(m3p, 'max_acc', 'measured') if m3p.exists() else None
                cp = a.e2/ds.name/f'{tool}_r{rep}'/'metrics.json' if a.e2 else None
                cas = json.loads(cp.read_text())['cascade'] if cp and cp.exists() else None
                out.append(dict(dataset=ds.name, rep=rep, tool=tool, poreloom=pl, w3000=m3, cascade=cas, w6000=w, paired=b))
                k = wins.setdefault(tool, [0, 0, 0, 0])
                k[0] += 1; k[1] += b['dacc_ci'][0] > 0; k[2] += b['dacc_ci'][1] < 0; k[3] += b['dpore_ci'][1] < 0
                lines.append(f"| {ds.name} | r{rep} | {fmt(pl)} | {NAMES[tool]} | {fmt(m3)} | {fmt(cas)} | {fmt(w)} | "
                             f"{100*w['host_command_reject_rate']:.1f} / {100*w['target_command_frr']:.1f} | "
                             f"{100*b['dacc']:+.1f} [{100*b['dacc_ci'][0]:+.1f}, {100*b['dacc_ci'][1]:+.1f}] | "
                             f"{b['dpore']:+.2f} [{b['dpore_ci'][0]:+.2f}, {b['dpore_ci'][1]:+.2f}] |")
    lines += ['', '| 工具 | 单元数 | PoreLoom ACC 显著更高 | 显著更低 | PoreLoom host 占孔显著更短 |', '|---|---|---|---|---|']
    lines += [f'| {NAMES[t]} | {v[0]} | {v[1]} | {v[2]} | {v[3]} |' for t, v in wins.items()]
    text = '\n'.join(lines) + '\n'; print(text)
    if a.md: a.md.write_text(text); a.md.with_suffix('.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('run', 'report'))
    ap.add_argument('--tool', choices=('readcurrent', 'nanodeep', 'squigglenet'))
    ap.add_argument('--recipe', choices=('project', 'upstream'), default='project')
    ap.add_argument('--data', type=Path); ap.add_argument('--out', type=Path); ap.add_argument('--fs', type=float)
    ap.add_argument('--rep', type=int, default=0); ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--root', type=Path); ap.add_argument('--main', type=Path); ap.add_argument('--e2', type=Path)
    ap.add_argument('--md', type=Path)
    a = ap.parse_args()
    run(a) if a.stage == 'run' else report(a)
