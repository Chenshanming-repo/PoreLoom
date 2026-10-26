#!/usr/bin/env python3
"""PoreLoom decision-position diagnostics from saved TEST outcomes.

The position is the observed usable signal at a completed Accept or Reject command,
excluding the 1500 skipped samples and capped at the 6000-sample budget. Reads
without a completed command are counted separately. The figure-data producer is
paper/scripts/signal_position_data.py.

    signal_budget.py <runs root>/main [--json out.json] [--md out.md]
"""
import argparse
import json
from pathlib import Path

import numpy as np

PORELOOM_SERVICE = 0.0158467092551291     # poreloom.method.head HEAD['service'] (nominal replay)
SKIP, BUDGET = 1500, 7500



def load(run, name, service, fs):
    z = np.load(run/'test'/name)
    n = np.where(z['command'], np.floor((z['decision_seconds'] - service)*fs + 1e-6), np.nan)
    return dict(read_id=z['read_id'], is_target=z['is_target'].astype(bool), command=z['command'].astype(bool), n=n)


def dist(x):
    x = x[np.isfinite(x)]
    return dict(mean=float(x.mean()), median=float(np.median(x)), p90=float(np.percentile(x, 90)), max=float(x.max())) if len(x) else None


def one(ds_dir, rep):
    run = ds_dir/f'poreloom_r{rep}'
    if not (run/'test/outcomes_0.4_nominal.npz').exists():
        return None
    fs = float(json.loads((run/'run.json').read_text())['fs'])
    values = load(run, 'outcomes_0.4_nominal.npz', PORELOOM_SERVICE, fs)
    used = np.minimum(values['n'], BUDGET) - SKIP
    y = values['is_target']
    classes = {}
    for category, selected in [('all', np.ones(len(y), bool)), ('host', ~y), ('target', y)]:
        commands = values['command'][selected]
        positions = used[selected][commands]
        classes[category] = dict(n_reads=int(selected.sum()), n_decided=int(commands.sum()),
                                 n_no_decision=int((~commands).sum()), position=dist(positions),
                                 fraction_by_3000=float(np.mean(positions <= 3000)) if len(positions) else None,
                                 fraction_at_6000=float(np.mean(positions == 6000)) if len(positions) else None)
    return dict(dataset=ds_dir.name, replicate=rep, fs=fs, classes=classes)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('main', type=Path)
    ap.add_argument('--json', type=Path)
    ap.add_argument('--md', type=Path)
    a = ap.parse_args()
    rows = [row for ds in sorted(a.main.iterdir()) if ds.is_dir()
            for rep in range(3) for row in [one(ds, rep)] if row]
    lines = ['# PoreLoom decision positions', '',
             '| Dataset | Repeat | Class | Completed decisions | No decision | Median usable samples |',
             '|---|---|---|---:|---:|---:|']
    for row in rows:
        for category, values in row['classes'].items():
            position = values['position']
            median = f"{position['median']:.0f}" if position else '–'
            lines.append(f"| {row['dataset']} | r{row['replicate']} | {category} | "
                         f"{values['n_decided']} | {values['n_no_decision']} | {median} |")
    text = '\n'.join(lines) + '\n'
    print(text)
    if a.md:
        a.md.write_text(text)
    if a.json:
        a.json.write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
