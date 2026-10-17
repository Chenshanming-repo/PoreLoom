#!/usr/bin/env python3
"""PoreLoom with a reduced decision budget vs the single-window baselines. Post hoc, no new training: the locked TEST outcomes (0.4 s) of the ablation budget variants
(runs/ablation/<ds>/r<r>/budget_B, B = absolute samples from the read start; the first 1500 are skipped, so B = 3000 uses
1500 samples of signal) and of the main baselines (runs/main/<ds>/<tool>_r<r>, max-ACC rule, measured service; windows
NanoDeep raw[1000:4000], ReadCurrent / SquiggleNet raw[1500:4500]). Paired bootstrap over TEST reads (2000, seed
20261015) of ACC(PoreLoom budget B) - ACC(tool), and of host pore time.

    budget_vs_baselines.py --ablation <runs>/ablation --main <runs>/main [--md summary_zh.md]
"""
import argparse
import json
from pathlib import Path

import numpy as np

UNITS = ['r10_run01/r0', 'r10_run02/r0', 'r10_run03/r0', 'r9/r0', 'r9/r1', 'r9/r2', 'sim_g1/r0', 'sim_g2/r0', 'sim_g3/r0',
         'sputum_g1/r0', 'sputum_g2/r0', 'sputum_g3/r0']
FAMILY = (('r10', 'R10'), ('r9', 'R9'), ('sim', 'Sim'), ('sputum', 'Sputum'))
PL = (('budget_3000', 'PoreLoom 预算 3000（用 1500 点）'), ('budget_4500', 'PoreLoom 预算 4500（用 3000 点）'),
      ('budget_6000', 'PoreLoom 预算 6000（用 4500 点）'), ('main', 'PoreLoom 完整（预算 7500，用 6000 点）'))
TOOLS = (('nanodeep_upstream', 'NanoDeep（作者配方，raw[1000:4000]）'), ('readcurrent', 'ReadCurrent（raw[1500:4500]）'),
         ('squigglenet', 'SquiggleNet（raw[1500:4500]）'), ('nanodeep', 'NanoDeep（项目配方）'))


def load(path):
    with np.load(path) as z:
        y = z['is_target'].astype(bool)
        return dict(id=z['read_id'].astype(str), y=y, ok=np.where(y, ~z['command_reject'], z['command_reject']).astype(float),
                    rej=z['command_reject'].astype(bool), pore=z['pore_seconds'].astype(float))


def stats(o):
    y = o['y']
    return dict(acc=o['ok'].mean(), frr=o['rej'][y].mean(), host_rej=o['rej'][~y].mean(), host_pore=o['pore'][~y].mean())


def main(a):
    rng = np.random.default_rng(20261015)
    rows = []
    for u in UNITS:
        ds, rep = u.split('/')
        pl = {v: load(a.ablation/u/v/'test_outcomes_0.4.npz') for v, _ in PL}
        tl = {t: load(a.main/ds/f'{t}_{rep}'/'test'/'outcomes_max_acc_0.4_measured.npz') for t, _ in TOOLS}
        ref = pl['main']['id']
        for o in list(pl.values()) + list(tl.values()): assert np.array_equal(o['id'], ref), u
        host = ~pl['main']['y']
        idx = rng.integers(0, len(ref), (2000, len(ref)))
        for v, _ in PL:
            r = dict(unit=u, method=v, **stats(pl[v]))
            for t, _ in TOOLS:
                d = pl[v]['ok'] - tl[t]['ok']; ci = np.percentile(d[idx].mean(1), [2.5, 97.5])
                r[f'vs_{t}'] = dict(dacc=float(d.mean()), ci=ci.tolist(), dpore=float((pl[v]['pore'] - tl[t]['pore'])[host].mean()))
            rows.append(r)
        for t, _ in TOOLS: rows.append(dict(unit=u, method=t, **stats(tl[t])))
    fam = lambda u: next(l for p, l in FAMILY if u.startswith(p))
    pct = lambda x: f'{100*x:.1f}'
    L = ['# PoreLoom 限制决策预算 vs 单窗口 baseline（TEST，0.4 s，事后分析，无新训练）', '',
         '预算 = 从 read 起点算的总采样点数（前 1500 点跳过）。每格 = 各数据集族 3 个单元的均值。', '',
         '## 绝对性能（ACC % / target 误拒 % / host 占孔 s）', '', '| 方法 | ' + ' | '.join(l for _, l in FAMILY) + ' |', '|---|' + '---|'*len(FAMILY)]
    for m, name in PL + TOOLS:
        cells = []
        for _, l in FAMILY:
            rs = [r for r in rows if r['method'] == m and fam(r['unit']) == l]
            cells.append(f"{pct(np.mean([r['acc'] for r in rs]))} / {pct(np.mean([r['frr'] for r in rs]))} / {np.mean([r['host_pore'] for r in rs]):.2f}")
        L.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    for t, tname in TOOLS:
        L += ['', f'## PoreLoom − {tname}（配对；ΔACC pp [显著更好/显著更差 单元数] / Δhost 占孔 s）', '',
              '| PoreLoom 版本 | ' + ' | '.join(l for _, l in FAMILY) + ' |', '|---|' + '---|'*len(FAMILY)]
        for v, vname in PL:
            cells = []
            for _, l in FAMILY:
                xs = [r[f'vs_{t}'] for r in rows if r['method'] == v and fam(r['unit']) == l]
                cells.append(f"{100*np.mean([x['dacc'] for x in xs]):+.1f} [{sum(x['ci'][0] > 0 for x in xs)}/{sum(x['ci'][1] < 0 for x in xs)}] / "
                             f"{np.mean([x['dpore'] for x in xs]):+.2f}")
            L.append(f'| {vname} | ' + ' | '.join(cells) + ' |')
    text = '\n'.join(L) + '\n'; print(text)
    if a.md:
        Path(a.md).write_text(text); Path(a.md).with_suffix('.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ablation', type=Path, required=True); ap.add_argument('--main', type=Path, required=True)
    ap.add_argument('--md', type=Path)
    main(ap.parse_args())
