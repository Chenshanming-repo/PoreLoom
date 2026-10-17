#!/usr/bin/env python3
"""E0: signal consumed per read at the decision (TEST, 0.4 s chunks, max-ACC operating point) for PoreLoom vs the
single-window baselines, from the existing outcome files (no new inference).

A command at time d after service s was issued on the arrival holding n = floor((d - s)*fs) raw samples.
  raw_at_decision  n (samples sequenced before the command, including the skipped read start)
  model_input      samples the model actually used: PoreLoom min(n, 7500) - 1500; baselines their native window (3000)
PoreLoom reads are split at n <= 4500 (no more model input than ReadCurrent / SquiggleNet) vs n > 4500, and every
method's command ACC is reported on the same read subsets (reads paired by read_id). Reads with no command (the read
ended first) count as accepted, as in command ACC.

    signal_budget.py <runs root>/main [--json out.json] [--md out.md]
"""
import argparse
import json
from pathlib import Path

import numpy as np

PORELOOM_SERVICE = 0.0158467092551291     # poreloom.method.head HEAD['service'] (nominal replay)
SKIP, BUDGET, WINDOW_END = 1500, 7500, 4500
BASELINES = ('readcurrent', 'nanodeep_upstream', 'squigglenet')
GROUPS = dict(r10=('r10_run01', 'r10_run02', 'r10_run03'), sputum=('sputum_g1', 'sputum_g2', 'sputum_g3'),
              r9=('r9',), sim=('sim_g1', 'sim_g2', 'sim_g3'))


def load(run, name, service, fs):
    z = np.load(run/'test'/name)
    n = np.where(z['command'], np.floor((z['decision_seconds'] - service)*fs + 1e-6), np.nan)
    correct = np.where(z['is_target'], ~z['command_reject'], z['command_reject'])
    return dict(read_id=z['read_id'], is_target=z['is_target'].astype(bool), command=z['command'].astype(bool), n=n, correct=correct)


def dist(x):
    x = x[np.isfinite(x)]
    return dict(mean=float(x.mean()), median=float(np.median(x)), p90=float(np.percentile(x, 90)), max=float(x.max())) if len(x) else None


def one(ds_dir, rep):
    pl_dir = ds_dir/f'poreloom_r{rep}'
    if not (pl_dir/'test'/'outcomes_0.4_nominal.npz').exists(): return None
    fs = float(json.loads((pl_dir/'run.json').read_text())['fs'])
    pl = load(pl_dir, 'outcomes_0.4_nominal.npz', PORELOOM_SERVICE, fs)
    y = pl['is_target']
    used = np.minimum(pl['n'], BUDGET) - SKIP
    late = pl['command'] & (pl['n'] > WINDOW_END)           # PoreLoom saw more than the baseline window
    early = ~late
    row = dict(dataset=ds_dir.name, replicate=rep, fs=fs, n_reads=len(y),
               poreloom=dict(command_rate=float(pl['command'].mean()), acc=float(pl['correct'].mean()),
                             model_input=dict(all=dist(used), host=dist(used[~y]), target=dist(used[y])),
                             raw_at_decision=dict(all=dist(pl['n']), host=dist(pl['n'][~y]), target=dist(pl['n'][y])),
                             seconds_at_decision=dict(host=dist(pl['n'][~y]/fs), target=dist(pl['n'][y]/fs)),
                             frac_over_window=dict(all=float(late.mean()), host=float(late[~y].mean()), target=float(late[y].mean())),
                             frac_forced=float((pl['command'] & (pl['n'] >= BUDGET)).mean()),
                             acc_within_window=float(pl['correct'][early].mean()), acc_over_window=float(pl['correct'][late].mean()) if late.any() else None,
                             wrong_share_over_window=float((late & ~pl['correct']).sum()/max(1, (~pl['correct']).sum()))))
    for tool in BASELINES:
        d = ds_dir/f'{tool}_r{rep}'
        p = d/'test'/'outcomes_max_acc_0.4_measured.npz'
        if not p.exists(): continue
        svc = float(json.loads((d/'selection.json').read_text())['service_seconds'])
        b = load(d, p.name, svc, fs)
        assert np.array_equal(b['read_id'], pl['read_id']), (d, 'read order differs from PoreLoom')
        row[tool] = dict(command_rate=float(b['command'].mean()), acc=float(b['correct'].mean()),
                         model_input=3000, raw_at_decision=dict(host=dist(b['n'][~y]), target=dist(b['n'][y])),
                         acc_within_window=float(b['correct'][early].mean()), acc_over_window=float(b['correct'][late].mean()) if late.any() else None)
    return row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('main', type=Path); ap.add_argument('--json', type=Path); ap.add_argument('--md', type=Path)
    a = ap.parse_args()
    rows = [r for ds in sorted(a.main.iterdir()) if ds.is_dir() for rep in range(3) for r in [one(ds, rep)] if r]
    lines = ['# E0 信号用量（TEST，0.4 s，最大 ACC 工作点）', '',
             'model_input = 模型实际用到的采样点（PoreLoom: min(n,7500)-1500；baseline 固定 3000）。'
             '"超窗" = PoreLoom 决策时已收到 >4500 原始点，即比 ReadCurrent/SquiggleNet 多看了信号。', '',
             '| 数据 | 组 | PoreLoom 输入点 host 均值/中位 | target 均值/中位 | 超窗比例 host / target | 强制决策 | PoreLoom ACC 窗内 / 超窗 | 错误中超窗占比 | 全部 ACC PL / RC / ND / SN | 窗内 ACC PL / RC / ND / SN |',
             '|---|---|---|---|---|---|---|---|---|---|']
    f = lambda v: '–' if v is None else f'{100*v:.1f}'
    for r in rows:
        p = r['poreloom']; mi = p['model_input']
        accs = ' / '.join(f(r[t]['acc']) if t in r else '–' for t in ('poreloom',) + BASELINES)
        win = ' / '.join(f(r[t]['acc_within_window']) if t in r else '–' for t in ('poreloom',) + BASELINES)
        lines.append(f"| {r['dataset']} | r{r['replicate']} | {mi['host']['mean']:.0f} / {mi['host']['median']:.0f} | "
                     f"{mi['target']['mean']:.0f} / {mi['target']['median']:.0f} | {f(p['frac_over_window']['host'])} / {f(p['frac_over_window']['target'])} | "
                     f"{f(p['frac_forced'])} | {f(p['acc_within_window'])} / {f(p['acc_over_window'])} | {f(p['wrong_share_over_window'])} | {accs} | {win} |")
    text = '\n'.join(lines) + '\n'; print(text)
    if a.md: a.md.write_text(text)
    if a.json: a.json.write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
