#!/usr/bin/env python3
"""Icarust chunk time (break_read_ms) for PoreLoom.

Same closed loop as the final Icarust matrix (experiments/20261014_final/icarust_run.py: sim g1 models, 512 channels,
1200 s, 20 s follow-up, seeds 20331101-03) at target 10%, with the simulated sequencer's chunk time changed:
  0.2 s  No-AS, PoreLoom (C-p head trained at 0.4 s), PoreLoom-head0.2 (same encoder, C-p trained at 0.2 s)
  0.8 s  No-AS, PoreLoom          (0.8 s = MinKNOW >= 6.0 default)
  1.0 s  No-AS, PoreLoom          (1.0 s = earlier MinKNOW default)
  0.4 s  reused from the final matrix (10pct No-AS and PoreLoom cells)
PoreLoom-head0.2 = ablation.py cadence/head_0.2 of sim_g1 r0 (selected on DEV-B at 0.2 s), bundled as the main
bundle with that head. Enrichment = Y_T / No-AS Y_T of the same seed and chunk time.

    icarust_chunk.py bundle --head <runs>/ablation/sim_g1/r0/cadence/head_0.2
    icarust_chunk.py matrix
    icarust_chunk.py aggregate
"""
import argparse
import json
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FINAL = REPO/'experiments/20261014_final'
sys.path.insert(0, str(FINAL))
import icarust_run as IR  # noqa: E402

OUT = Path('/path/to/poreloom_work/results/final_20261014/icarust_chunk')
PERMILLE = 100
CELLS = {200: ['No-AS', 'PoreLoom', 'PoreLoom-head0.2'], 800: ['No-AS', 'PoreLoom'], 1000: ['No-AS', 'PoreLoom']}
BUNDLE = OUT/'poreloom_sh_cp_head0.2.pt'


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def cells():
    out = []
    for seed in IR.SEEDS:
        block = [dict(seed=seed, permille=PERMILLE, break_ms=ms, strategy=s) for ms, ss in CELLS.items() for s in ss]
        random.Random(seed + 7).shuffle(block); out += block
    for i, c in enumerate(out):
        c.update(lane=i % len(IR.LANES), name=f"{c['break_ms']}ms_{c['seed']}_{c['strategy']}")
    return out


def bundle(a):
    """Main sim_g1 r0 bundle with the 0.2 s head (beta locked on DEV-B at 0.2 s); registered in icarust_models.json."""
    import torch
    sys.path.insert(0, str(REPO/'src'))
    from poreloom.method import head as H
    run = Path(json.loads((FINAL/'icarust_models.json').read_text())['PoreLoom'])
    main = torch.load(run/'poreloom_sh_cp.pt', map_location='cpu')
    choice = json.loads((a.head/'selection.json').read_text())
    ctrl = H.load_controller(a.head/'controllers.pt', choice['beta'])
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(dict(main, head_state=ctrl.state_dict(), beta=choice['beta'], interval_trained=.2, head_source=str(a.head),
                    head=dict(main['head'], input_dim=int(ctrl.state_dict()['mean'].shape[0]))), BUNDLE)
    models = json.loads((FINAL/'icarust_models.json').read_text()); models['PoreLoom-head0.2'] = str(BUNDLE)
    (FINAL/'icarust_models.json').write_text(json.dumps(models, indent=2) + '\n')
    log('bundle', BUNDLE, 'beta', choice['beta'], 'DEV-B ACC', round(choice['sel']['acc'], 4))


def launch(c, out, device, port):
    cmd = [IR.PYTHON, str(FINAL/'icarust_live.py'), '--strategy', c['strategy'], '--out', str(out), '--seed', str(c['seed']),
           '--channels', str(IR.CHANNELS), '--duration', str(IR.DURATION), '--follow', str(IR.FOLLOW),
           '--target-permille', str(c['permille']), '--port', str(port), '--device', device, '--break-ms', str(c['break_ms'])]
    with open(f'{out}.log', 'a') as f:
        code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    if code == 0: subprocess.call([IR.PYTHON, str(IR.SIMLOOP/'analyze_live.py'), str(out)])
    return code


def matrix(a):
    assert BUNDLE.exists(), 'run the bundle stage first'
    live = OUT/'live'; live.mkdir(parents=True, exist_ok=True)
    plan = cells()
    if not (OUT/'plan.json').exists():
        (OUT/'plan.json').write_text(json.dumps(dict(written=time.strftime('%Y-%m-%d %H:%M:%S'), cells=plan, lanes=IR.LANES,
                                                     channels=IR.CHANNELS, duration=IR.DURATION, follow=IR.FOLLOW,
                                                     models=json.loads((FINAL/'icarust_models.json').read_text())), indent=2))

    def lane(i):
        device, port = IR.LANES[i]
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
    threads = [threading.Thread(target=lane, args=(i,)) for i in range(len(IR.LANES))]
    for t in threads: t.start()
    for t in threads: t.join()
    (OUT/'matrix.done').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))


def aggregate(a):
    rows = []
    for c in cells():
        p = OUT/'live'/c['name']/'summary.json'
        if p.exists(): rows.append({**json.loads(p.read_text()), **c})
    for seed in IR.SEEDS:                                   # 0.4 s cells of the final matrix
        for s in ('No-AS', 'PoreLoom'):
            p = IR.OUT/'live'/f'{PERMILLE/10:g}pct_{seed}_{s}'/'summary.json'
            if p.exists(): rows.append({**json.loads(p.read_text()), **dict(seed=seed, permille=PERMILLE, break_ms=400, strategy=s)})
    lines = ['# Icarust chunk 时间（sim g1 模型，target 10%，512 通道，1200 s）', '',
             '每格为 3 个 seed 的均值（范围）。富集 = Y_T / 同 seed、同 chunk 时间的 No-AS Y_T。0.4 s 一行来自最终 Icarust 矩阵。', '',
             '| chunk 时间 | 策略（动作头训练节奏） | n | 富集 | host 有效排出 % | target 指令误拒 % | host 占孔 s | 延迟 p50 s |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    table = {}
    for ms in sorted({r['break_ms'] for r in rows}):
        base = {r['seed']: r['yield_target_bp_per_channel_hour'] for r in rows if r['break_ms'] == ms and r['strategy'] == 'No-AS'}
        for s in ('PoreLoom', 'PoreLoom-head0.2'):
            rs = [r for r in rows if r['break_ms'] == ms and r['strategy'] == s]
            if not rs: continue
            e = np.array([r['yield_target_bp_per_channel_hour']/base[r['seed']] for r in rs if r['seed'] in base])
            h = np.array([r['host_effective_reject']['value'] for r in rs]); f = np.array([r['target_command_reject']['value'] for r in rs])
            pore = np.array([r['host_mean_pore_seconds'] for r in rs])
            lat = [r['command_latency_s']['p50'] for r in rs if r.get('command_latency_s')]
            table[f'{ms}/{s}'] = dict(n=len(rs), enrichment=e.tolist(), host_effective_reject=float(h.mean()),
                                      target_command_frr=float(f.mean()), host_pore=float(pore.mean()))
            enr = f'{e.mean():.2f}× ({e.min():.2f}–{e.max():.2f})' if len(e) else '–'
            label = 'PoreLoom（0.2 s 训练）' if s.endswith('0.2') else 'PoreLoom（0.4 s 训练）'
            lines.append(f"| {ms/1000:g} s | {label} | {len(rs)} | {enr} | {100*h.mean():.1f} | {100*f.mean():.1f} | {pore.mean():.3f} | "
                         f"{np.mean(lat):.3f} |" if lat else f"| {ms/1000:g} s | {label} | {len(rs)} | {enr} | {100*h.mean():.1f} | "
                         f"{100*f.mean():.1f} | {pore.mean():.3f} | – |")
    (OUT/'aggregate.json').write_text(json.dumps(dict(rows=rows, table=table), indent=2, default=str))
    (OUT/'summary_zh.md').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('bundle', 'matrix', 'aggregate'))
    ap.add_argument('--head', type=Path)
    a = ap.parse_args()
    globals()[a.stage](a)
