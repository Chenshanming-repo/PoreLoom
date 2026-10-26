#!/usr/bin/env python3
"""Icarust closed loop: the 6000-sample baseline extension used in the final 72-run comparison.

Strategies readcurrent_w6000 / nanodeep_w6000 / squigglenet_w6000 (live.py) on exactly the cells of the final matrix
(experiments/20261014_final/icarust_run.py: target 1%, 10%, 30%; seeds 20331101-03; 512 channels, 1200 s, 20 s
follow-up; sim g1 models). The 45 existing runs (No-AS, PoreLoom, W3000 baselines) are read, never rewritten; all new
output goes to OUT. Enrichment uses the existing No-AS run of the same seed and target fraction.

    icarust_w6000.py check [--n 3000] [--device cuda:0]   live W6000 policies vs their offline TEST commands -> check.json
    icarust_w6000.py matrix                               27 runs on parallel lanes (retries failed runs)
    icarust_w6000.py aggregate                            old 45 + new 27 -> aggregate.json, summary_zh.md
"""
import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PYTHON = '/path/to/poreloom_work/.venv-nanodeep/bin/python'
OLD = Path('/path/to/poreloom_work/results/final_20261014/icarust')
OUT = Path('/path/to/poreloom_work/results/icarust_w6000_20261018')
SIMLOOP = Path(os.environ.get('PORELOOM_SIMLOOP', REPO/'experiments/20261011_simloop'))
DATA = Path('/path/to/poreloom_work/results/sim_datasets_20261014/g1')
OLD_STRATEGIES = ['No-AS', 'PoreLoom', 'readcurrent', 'nanodeep', 'squigglenet']
STRATEGIES = ['readcurrent_w6000', 'nanodeep_w6000', 'squigglenet_w6000']
PERMILLE = (10, 100, 300)
SEEDS = (20331101, 20331102, 20331103)
LANES = [('cuda:0', 25400), ('cuda:2', 25500), ('cuda:5', 25600)]
CHANNELS, DURATION, FOLLOW, FS = 512, 1200, 20, 4000
NAMES = {'No-AS': 'No-AS', 'PoreLoom': 'PoreLoom', 'readcurrent': 'ReadCurrent W3000', 'nanodeep': 'NanoDeep W3000',
         'squigglenet': 'SquiggleNet W3000', 'readcurrent_w6000': 'ReadCurrent W6000', 'nanodeep_w6000': 'NanoDeep W6000',
         'squigglenet_w6000': 'SquiggleNet W6000'}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def cells(strategies=STRATEGIES):
    out = []
    for seed in SEEDS:
        block = [dict(seed=seed, permille=p, strategy=s) for p in PERMILLE for s in strategies]
        random.Random(seed).shuffle(block); out += block
    for i, c in enumerate(out):
        c.update(lane=i % len(LANES), name=f"{c['permille']/10:g}pct_{c['seed']}_{c['strategy']}")
    return out


def check(a):
    """Feed sim g1 TEST reads to each live W6000 policy as 0.4 s chunks with the offline per-read phase (seed 20261012);
    the live command must equal the offline zero-service command wherever the offline replay issued one."""
    sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO/'src'))
    import live
    import torch
    from poreloom.method import native_policy as offline_policy
    from poreloom.chunk_replay import arrivals
    from poreloom.method import data as D
    torch.cuda.set_device(a.device)
    data = D.load_split(DATA, 'test')
    rows = np.sort(np.random.default_rng(7).choice(len(data.read_id), a.n, replace=False))
    u = np.random.default_rng(20261012).random(len(data.read_id))[rows]
    m = arrivals(u*.4*FS, .4*FS, 13500); raw = np.asarray(data.raw_length)[rows]
    result = {}
    for name in STRATEGIES:
        tool = name[:-len('_w6000')]
        run = Path(live.W6[tool])
        with np.load(run/'test/scores.npz', allow_pickle=False) as saved:
            scores = {key: saved[key] for key in saved.files}
        selection = json.loads((run/'selection.json').read_text())
        offline_policy.FS = FS
        meta = dict(read_id=data.read_id, is_target=np.asarray(data.is_target, bool), raw_length=data.raw_length)
        off = offline_policy.native_policy(scores, meta, threshold=selection['rules']['max_acc']['threshold'],
                                   service=0., delta=.4, raw_end=7500)
        policy = live.build(name, a.device, live.cl.TRAIN_CAL)
        states = [dict(channel=i, read_id=str(data.read_id[r]), read_number=1, received=0, k=0, h=None,
                       buf=np.asarray(data.adc[r]).astype(np.int16).copy()) for i, r in enumerate(rows)]
        got = np.full(len(rows), None, object)
        for j in range(m.shape[1]):
            for i, s in enumerate(states):
                if got[i] is not None or not np.isfinite(m[i, j]): continue
                s['received'] = int(min(np.floor(m[i, j]), raw[i], len(s['buf'])))
                need = policy.needed(s)
                if need is not None and s['received'] >= need: got[i] = policy.decide([s])[0][1]
        want = np.where(off['command'][rows], np.where(off['command_reject'][rows], 'reject', 'accept'), None)
        cmp = [(g, w) for g, w in zip(got, want) if w is not None]
        agree = sum(g == w for g, w in cmp)
        result[name] = dict(compared=len(cmp), agree=agree, agreement=agree/max(1, len(cmp)),
                            live_reject=int(sum(g == 'reject' for g in got)), offline_reject=int(off['command_reject'][rows].sum()),
                            live_none=int(sum(g is None for g in got)), offline_none=int(sum(w is None for w in want)))
        print(name, json.dumps(result[name]), flush=True)
        assert agree == len(cmp), (name, 'live commands differ from the offline reference')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'check.json').write_text(json.dumps(dict(n=a.n, results=result), indent=2))


def launch(c, out, device, port):
    for name in ('closed_loop.py', 'analyze_live.py'):
        if not (SIMLOOP/name).is_file():
            raise FileNotFoundError(f'{SIMLOOP/name} is required; set PORELOOM_SIMLOOP to the external simloop support directory')
    cmd = [PYTHON, str(HERE/'live.py'), '--strategy', c['strategy'], '--out', str(out), '--seed', str(c['seed']),
           '--channels', str(CHANNELS), '--duration', str(DURATION), '--follow', str(FOLLOW),
           '--target-permille', str(c['permille']), '--port', str(port), '--device', device]
    with open(f'{out}.log', 'a') as f:
        code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    if code == 0: subprocess.call([PYTHON, str(SIMLOOP/'analyze_live.py'), str(out)])
    return code


def matrix(a):
    live_dir = OUT/'live'; live_dir.mkdir(parents=True, exist_ok=True)
    plan = cells()
    if not (OUT/'plan.json').exists():
        (OUT/'plan.json').write_text(json.dumps(dict(written=time.strftime('%Y-%m-%d %H:%M:%S'), cells=plan, lanes=LANES,
                                                     channels=CHANNELS, duration=DURATION, follow=FOLLOW,
                                                     models=json.loads((HERE/'models.json').read_text()),
                                                     check=json.loads((OUT/'check.json').read_text())), indent=2))

    def lane(i):
        device, port = LANES[i]
        time.sleep(60*i)
        for c in plan:
            if c['lane'] != i: continue
            out = live_dir/c['name']
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


def aggregate(a):
    rows = []
    for root, strategies in ((OLD, OLD_STRATEGIES), (OUT, STRATEGIES)):
        for c in cells(strategies):
            p = root/'live'/c['name']/'summary.json'
            if p.exists(): rows.append(dict(c, source=str(root), **json.loads(p.read_text())))
    lines = ['# Icarust 闭环：加入 6000 单窗口 baseline（sim g1 模型，512 通道，1200 s）', '',
             'No-AS / PoreLoom / W3000 为 2026-10-14 原矩阵（只读），W6000 为本次新跑。每格为 3 个 seed 的均值（范围）。',
             'Y_T = 时限内未被拒绝的 target 碱基 / (通道·小时)；富集 = Y_T / 同 seed、同比例 No-AS 的 Y_T。', '']
    table = {}
    for p in PERMILLE:
        lines += [f'## target {p/10:g}%', '',
                  '| 策略 | n | Y_T bp/(ch·h) | 富集 | host 有效排出 % | target 指令误拒 % | host 占孔 s | 延迟 p50 s |', '|---|---:|---:|---:|---:|---:|---:|---:|']
        base = {r['seed']: r['yield_target_bp_per_channel_hour'] for r in rows if r['permille'] == p and r['strategy'] == 'No-AS'}
        for s in OLD_STRATEGIES + STRATEGIES:
            rs = [r for r in rows if r['permille'] == p and r['strategy'] == s]
            if not rs: continue
            y = np.array([r['yield_target_bp_per_channel_hour'] for r in rs])
            e = np.array([r['yield_target_bp_per_channel_hour']/base[r['seed']] for r in rs if r['seed'] in base])
            h = np.array([r['host_effective_reject']['value'] for r in rs]); f = np.array([r['target_command_reject']['value'] for r in rs])
            pore = np.array([r['host_mean_pore_seconds'] for r in rs])
            lat = [r['command_latency_s']['p50'] for r in rs if r.get('command_latency_s')]
            table[f'{p}/{s}'] = dict(n=len(rs), yield_mean=float(y.mean()), enrichment=float(e.mean()) if len(e) else None,
                                     host_effective_reject=float(h.mean()), target_command_frr=float(f.mean()), host_pore=float(pore.mean()),
                                     latency_p50=float(np.mean(lat)) if lat else None)
            lines.append(f"| {NAMES[s]} | {len(rs)} | {y.mean():,.0f} ({y.min():,.0f}–{y.max():,.0f}) | "
                         + (f"{e.mean():.2f}× | " if len(e) else '– | ')
                         + f"{100*h.mean():.1f} | {100*f.mean():.1f} | {pore.mean():.3f} | "
                         + (f"{np.mean(lat):.3f} |" if lat else '– |'))
        lines.append('')
    (OUT/'aggregate.json').write_text(json.dumps(dict(rows=rows, table=table), indent=2, default=str))
    (OUT/'summary_zh.md').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('check', 'matrix', 'aggregate'))
    ap.add_argument('--n', type=int, default=3000); ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    dict(check=check, matrix=matrix, aggregate=aggregate)[a.stage](a)
