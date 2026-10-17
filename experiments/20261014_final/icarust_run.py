#!/usr/bin/env python3
"""Icarust closed-loop matrix for the final comparison: sim g1 models; target 1%, 10%, 30%;
strategies No-AS, PoreLoom, ReadCurrent, NanoDeep (authors' recipe), SquiggleNet (single window); seeds 20331101-03.
512 channels, 1200 s fixed window, 20 s follow-up (simloop defaults). Order shuffled within each seed block.

    icarust_run.py smoke --strategy PoreLoom --duration 120     short run, not part of the matrix
    icarust_run.py matrix                                       45 runs on parallel lanes (retries failed runs)
    icarust_run.py aggregate                                    -> aggregate.json, summary_zh.md
"""
import argparse
import json
import random
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PYTHON = '/path/to/poreloom_work/.venv-nanodeep/bin/python'
OUT = Path('/path/to/poreloom_work/results/final_20261014/icarust')
SIMLOOP = REPO/'experiments/20261011_simloop'
STRATEGIES = ['No-AS', 'PoreLoom', 'readcurrent', 'nanodeep', 'squigglenet']
PERMILLE = (10, 100, 300)
SEEDS = (20331101, 20331102, 20331103)
LANES = [('cuda:3', 25100), ('cuda:4', 25200), ('cuda:6', 25300)]
CHANNELS, DURATION, FOLLOW = 512, 1200, 20


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def cells():
    out = []
    for seed in SEEDS:
        block = [dict(seed=seed, permille=p, strategy=s) for p in PERMILLE for s in STRATEGIES]
        random.Random(seed).shuffle(block); out += block
    for i, c in enumerate(out):
        c.update(lane=i % len(LANES), name=f"{c['permille']/10:g}pct_{c['seed']}_{c['strategy']}")
    return out


def launch(c, out, device, port, duration=DURATION, follow=FOLLOW):
    cmd = [PYTHON, str(HERE/'icarust_live.py'), '--strategy', c['strategy'], '--out', str(out), '--seed', str(c['seed']),
           '--channels', str(CHANNELS), '--duration', str(duration), '--follow', str(follow),
           '--target-permille', str(c['permille']), '--port', str(port), '--device', device]
    with open(f'{out}.log', 'a') as f:
        code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    if code == 0:
        subprocess.call([PYTHON, str(SIMLOOP/'analyze_live.py'), str(out)])
    return code


def smoke(strategy, duration):
    out = OUT/'smoke'/f"{strategy}_{duration}s_{time.strftime('%H%M%S')}"
    out.parent.mkdir(parents=True, exist_ok=True)
    device, port = LANES[0]
    code = launch(dict(strategy=strategy, seed=SEEDS[0], permille=100), out, device, port, duration, 10)
    log('smoke', strategy, 'exit', code)
    if code == 0: print((out/'summary.json').read_text())


def matrix():
    live = OUT/'live'; live.mkdir(parents=True, exist_ok=True)
    plan = cells()
    if not (OUT/'plan.json').exists():
        (OUT/'plan.json').write_text(json.dumps(dict(written=time.strftime('%Y-%m-%d %H:%M:%S'), cells=plan, lanes=LANES,
                                                     channels=CHANNELS, duration=DURATION, follow=FOLLOW,
                                                     models=json.loads((HERE/'icarust_models.json').read_text()),
                                                     check=json.loads((OUT/'check.json').read_text())), indent=2))

    def lane(i):
        device, port = LANES[i]
        time.sleep(60*i)
        for c in plan:
            if c['lane'] != i: continue
            out = live/c['name']
            for attempt in range(3):
                if (out/'summary.json').exists(): break
                if out.exists(): out.rename(out.with_name(out.name + f'.failed_{int(time.time())}'))
                log('start', c['name'], device, 'attempt', attempt)
                log('done', c['name'], 'exit', launch(c, out, device, port))
                time.sleep(15)
    threads = [threading.Thread(target=lane, args=(i,)) for i in range(len(LANES))]
    for t in threads: t.start()
    for t in threads: t.join()
    (OUT/'matrix.done').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))


def aggregate():
    rows = []
    for c in cells():
        p = OUT/'live'/c['name']/'summary.json'
        if p.exists(): rows.append(dict(c, **json.loads(p.read_text())))
    name = {'No-AS': 'No-AS', 'PoreLoom': 'PoreLoom', 'readcurrent': 'ReadCurrent', 'nanodeep': 'NanoDeep', 'squigglenet': 'SquiggleNet'}
    lines = ['# Icarust 闭环：sim g1 模型，512 通道，1200 s', '',
             '每格为 3 个 seed 的均值（范围）。Y_T = 时限内未被拒绝的 target 碱基 / (通道·小时)；富集 = Y_T / No-AS 的 Y_T（同 seed 配对）。', '']
    table = {}
    for p in PERMILLE:
        lines += [f'## target {p/10:g}%', '',
                  '| 策略 | n | Y_T bp/(ch·h) | 富集 | host 有效排出 % | target 指令误拒 % | host 占孔 s | 延迟 p50 s |', '|---|---:|---:|---:|---:|---:|---:|---:|']
        base = {r['seed']: r['yield_target_bp_per_channel_hour'] for r in rows if r['permille'] == p and r['strategy'] == 'No-AS'}
        for s in STRATEGIES:
            rs = [r for r in rows if r['permille'] == p and r['strategy'] == s]
            if not rs: continue
            y = np.array([r['yield_target_bp_per_channel_hour'] for r in rs])
            e = np.array([r['yield_target_bp_per_channel_hour']/base[r['seed']] for r in rs if r['seed'] in base])
            h = np.array([r['host_effective_reject']['value'] for r in rs]); f = np.array([r['target_command_reject']['value'] for r in rs])
            pore = np.array([r['host_mean_pore_seconds'] for r in rs])
            lat = [r['command_latency_s']['p50'] for r in rs if r.get('command_latency_s')]
            table[f'{p}/{s}'] = dict(n=len(rs), yield_mean=float(y.mean()), enrichment=float(e.mean()) if len(e) else None,
                                     host_effective_reject=float(h.mean()), target_command_frr=float(f.mean()), host_pore=float(pore.mean()))
            lines.append(f"| {name[s]} | {len(rs)} | {y.mean():,.0f} ({y.min():,.0f}–{y.max():,.0f}) | "
                         f"{e.mean():.2f}× | {100*h.mean():.1f} | {100*f.mean():.1f} | {pore.mean():.3f} | "
                         f"{np.mean(lat):.3f} |" if len(e) else f"| {name[s]} | {len(rs)} | {y.mean():,.0f} | – | {100*h.mean():.1f} | {100*f.mean():.1f} | {pore.mean():.3f} | – |")
        lines.append('')
    (OUT/'aggregate.json').write_text(json.dumps(dict(rows=rows, table=table), indent=2, default=str))
    (OUT/'summary_zh.md').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('smoke', 'matrix', 'aggregate'))
    ap.add_argument('--strategy', default='PoreLoom'); ap.add_argument('--duration', type=int, default=120)
    a = ap.parse_args()
    {'smoke': lambda: smoke(a.strategy, a.duration), 'matrix': matrix, 'aggregate': aggregate}[a.stage]()
