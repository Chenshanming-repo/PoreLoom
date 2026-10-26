#!/usr/bin/env python3
"""6000-sample fixed-window baselines for the main comparison.

All three baselines train and classify on raw[1500:7500], retaining their own
normalization and DEV-B-selected thresholds. NanoDeep uses the authors' recipe.
Reads that end before the complete window arrives are retained.

    win6000.py run --tool T [--recipe upstream] --data <data>/<ds> --fs F --rep r --out <runs>/win6000/<ds>/<T>_r<r>
    win6000.py report --root <runs>/win6000 --main <runs>/main [--md summary_zh.md]
"""
import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'src'))

WINDOW, START = 6000, 1500
TOOLS = ('readcurrent', 'nanodeep_upstream', 'squigglenet')
NAMES = dict(readcurrent='ReadCurrent', nanodeep_upstream='NanoDeep', squigglenet='SquiggleNet')


def run(a):
    from poreloom.baselines import preprocessing as pp
    from poreloom.method import baselines
    recipe = 'upstream' if a.tool == 'nanodeep' else 'project'
    if a.recipe is not None and a.recipe != recipe:
        raise ValueError(f'The manuscript uses recipe={recipe} for {a.tool}')
    pp.WINDOW = WINDOW
    for c in pp.CONTRACTS.values():       # shared dict objects: native / native_policy / baselines see the change
        c.update(raw_start=START, raw_end=START + WINDOW)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/'window.json').write_text(json.dumps(dict(window=WINDOW, raw_start=START, raw_end=START + WINDOW)))
    baselines.main(['--tool', a.tool, '--recipe', recipe, '--data', str(a.data), '--out', str(a.out), '--fs', str(a.fs),
                    '--replicate', str(a.rep), '--device', a.device])


def pooled(path, rule=None, service='nominal'):
    rows = json.loads(path.read_text())['rows']
    return next(r for r in rows if r['interval'] == .4 and r['service'] == service and r['group'] == 'pooled'
                and (rule is None or r['rule'] == rule))


def report(a):
    fmt = lambda r: f"{100*r['acc']:.2f} / {r['host_mean_pore_seconds']:.3f}" if r else '–'
    lines = ['# Fixed-window baselines and PoreLoom', '',
             'Accuracy (%) / mean host pore time (s), TEST at 0.4-s observations.', '',
             '| Dataset | Repeat | PoreLoom | Baseline | 3000 samples | 6000 samples |',
             '|---|---|---|---|---|---|']
    rows = []
    for dataset in sorted(p for p in a.root.iterdir() if p.is_dir()):
        for rep in range(3):
            main = a.main/dataset.name
            pl_path = main/f'poreloom_r{rep}'/'test/metrics.json'
            if not pl_path.exists():
                continue
            pl = pooled(pl_path)
            for tool in TOOLS:
                path = dataset/f'{tool}_r{rep}'/'test/metrics.json'
                if not path.exists():
                    continue
                w6000 = pooled(path, 'max_acc', 'measured')
                short = main/f'{tool}_r{rep}'/'test/metrics.json'
                w3000 = pooled(short, 'max_acc', 'measured') if short.exists() else None
                rows.append(dict(dataset=dataset.name, rep=rep, tool=tool,
                                 poreloom=pl, w3000=w3000, w6000=w6000))
                lines.append(f'| {dataset.name} | r{rep} | {fmt(pl)} | {NAMES[tool]} | {fmt(w3000)} | {fmt(w6000)} |')
    text = '\n'.join(lines) + '\n'
    print(text)
    if a.md:
        a.md.write_text(text)
        a.md.with_suffix('.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('run', 'report'))
    ap.add_argument('--tool', choices=('readcurrent', 'nanodeep', 'squigglenet'))
    ap.add_argument('--recipe', choices=('project', 'upstream'), default=None)
    ap.add_argument('--data', type=Path); ap.add_argument('--out', type=Path); ap.add_argument('--fs', type=float)
    ap.add_argument('--rep', type=int, default=0); ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--root', type=Path); ap.add_argument('--main', type=Path)
    ap.add_argument('--md', type=Path)
    a = ap.parse_args()
    run(a) if a.stage == 'run' else report(a)
