#!/usr/bin/env python3
"""One line per finished run under <runs>/<experiment>/<dataset>/<method>_r<rep>: TEST pooled, 0.4 s, nominal/measured
service, DEV-B max-ACC operating point (PoreLoom: poreloom.method test/metrics.json; baselines: method.baselines).

    summarize.py <runs root> [--json out.json]
"""
import argparse
import json
from pathlib import Path


def row(run):
    m = json.loads((run/'test'/'metrics.json').read_text())
    if run.name.startswith('poreloom'):
        r = next(r for r in m['rows'] if r['interval'] == .4 and r['service'] == 'nominal' and r['group'] == 'pooled')
        host = r['host_command_reject']
    else:
        r = next(r for r in m['rows'] if r['rule'] == 'max_acc' and r['interval'] == .4 and r['service'] == 'measured' and r['group'] == 'pooled')
        host = r['host_command_reject_rate']
    method, rep = run.name.rsplit('_r', 1)
    return dict(experiment=run.parent.parent.name, dataset=run.parent.name, method=method, replicate=int(rep), acc=r['acc'],
                host_command_reject=host, target_command_frr=r['target_command_frr'], host_mean_pore_seconds=r['host_mean_pore_seconds'])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('runs', type=Path); ap.add_argument('--json', type=Path); a = ap.parse_args()
    methods = {'poreloom', 'readcurrent', 'nanodeep_upstream', 'squigglenet'}
    rows = [row(p.parent.parent) for p in sorted(a.runs.glob('*/*/*/test/metrics.json'))
            if p.parent.parent.parent.parent.name in {'main', 'lso', 'win6000'}
            and p.parent.parent.name.rsplit('_r', 1)[0] in methods]
    for r in rows:
        print(f"{r['experiment']:5s} {r['dataset']:30s} {r['method']:12s} r{r['replicate']}  ACC {100*r['acc']:5.2f}  "
              f"host {100*r['host_command_reject']:5.1f}  FRR {100*r['target_command_frr']:5.1f}  pore {r['host_mean_pore_seconds']:.3f}")
    if a.json: a.json.write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
