#!/usr/bin/env python3
"""E2: do the single-window baselines catch up with PoreLoom when they may also look longer on hard reads?

Adaptive cascade baseline (no retraining): the trained native model (main/<tool>_r<rep>/fit/best.pt) scores up to
three windows of its native length, shifted by 1500 samples (ReadCurrent / SquiggleNet raw[1500:4500], [3000:6000],
[4500:7500] = PoreLoom's 7500-sample budget; NanoDeep raw[1000:4000], [2500:5500], [4000:7000]). A window is scored at
the first 0.4 s chunk arrival holding it, plus the tool's measured single-window service (selection.json). Score at
window k: the host margin of window k ('last') or the mean margin of windows 0..k ('mean'). Windows 0 and 1: reject
if score >= r, accept if score <= a, otherwise wait; window 2: reject if score >= f, otherwise accept. A read that
ends first gets no command (kept), as in every other replay. Thresholds (r, a, f) and the aggregation are chosen on
DEV-B by maximum command ACC (ties: lower target FRR, then lower host pore time) before TEST is touched; the DEV-B
Pareto front (ACC vs host pore time) is also replayed on TEST for curves.

PoreLoom curve: every C-p controller of the beta grid (head/controllers.pt) replayed on TEST at 0.4 s, nominal service
(the locked beta must reproduce test/metrics.json).

    cascade.py run --main <runs>/main/<ds> --data <data>/<ds> --out <runs>/e2/<ds> --rep r [--device cuda:0]
    cascade.py report --e2 <runs>/e2 --main <runs>/main [--md out.md]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from poreloom.baselines.preprocessing import CONTRACTS, prepare_windows
from poreloom.method import data as D, head as H, native, native_policy as policy

TOOLS = ('readcurrent', 'nanodeep_upstream', 'squigglenet')
SHIFT, N_WINDOWS, DELTA, EJECT = 1500, 3, .4, .1
QUANTILES = np.linspace(.025, .975, 39)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


@torch.inference_mode()
def window_margins(tool, model, prefix, device, batch=256):
    """Host margin [N, N_WINDOWS] of the shifted native windows (nan: read too short or invalid normalization)."""
    c = CONTRACTS[tool]; n = len(prefix)
    out = np.full((n, N_WINDOWS), np.nan, np.float32)
    for k in range(N_WINDOWS):
        lo, hi = c['raw_start'] + k*SHIFT, c['raw_end'] + k*SHIFT
        assert prefix.adc.shape[1] >= hi
        rows = np.flatnonzero(prefix.raw_length >= hi)
        for s in range(0, len(rows), batch):
            part = rows[s:s + batch]
            raw = prefix.pa(part, lo, hi) if c['units'] == 'pA' else np.asarray(prefix.adc[part, lo:hi], np.int16)
            x, valid = prepare_windows(raw, tool)
            if x.ndim == 2: x = x[:, None]
            ok = np.flatnonzero(valid)
            if not len(ok): continue
            logits = native._forward(model, x[ok].astype(np.float32), tool, device)
            out[part[ok], k] = (logits[:, c['host_index']] - logits[:, c['microbe_index']]).cpu().numpy()
    return out


def timing(raw_length, raw_end, service):
    """Per window: completion time [N,K], natural end [N], read still sequencing at completion [N,K]."""
    phase = policy.phase_uniform(len(raw_length))*DELTA*policy.FS; chunk = DELTA*policy.FS
    end = np.asarray(raw_length, np.float64)/policy.FS
    comp, alive = [], []
    for k in range(N_WINDOWS):
        e = raw_end + k*SHIFT
        callback = (phase + np.maximum(0., np.ceil((e - phase)/chunk))*chunk)/policy.FS   # = native_policy._timing
        comp.append(callback + service); alive.append((raw_length >= e) & (callback + service < end))
    return np.stack(comp, 1), end, np.stack(alive, 1)


def aggregate(m, how):
    if how == 'last': return m
    s = np.nancumsum(m, 1); c = np.cumsum(np.isfinite(m), 1)
    return np.where(np.isfinite(m), s/np.maximum(c, 1), np.nan)


def replay(score, comp, end, alive, r, a, f, windows=N_WINDOWS):
    n = len(end)
    pending = np.ones(n, bool)
    command, reject, dec, window = np.zeros(n, bool), np.zeros(n, bool), np.full(n, np.nan), np.full(n, -1)
    for k in range(windows):
        s = score[:, k]; can = pending & alive[:, k] & np.isfinite(s)
        if k < windows - 1:
            rej, acc = can & (s >= r), can & (s <= a) & ~(s >= r)
        else:
            rej, acc = can & (s >= f), can & ~(s >= f)
        act = rej | acc
        command |= act; reject |= rej; dec[act] = comp[act, k]; window[act] = k; pending &= ~act
    effective = reject & (dec + EJECT < end)
    return dict(command=command, command_reject=reject, effective_reject=effective,
                pore_seconds=np.where(effective, dec + EJECT, end), decision_seconds=dec, window=window)


def stats(o, y):
    host = ~y
    return dict(acc=float((o['command_reject'][host].sum() + (~o['command_reject'][y]).sum())/len(y)),
                host_command_reject=float(o['command_reject'][host].mean()), target_command_frr=float(o['command_reject'][y].mean()),
                host_effective_reject=float(o['effective_reject'][host].mean()), host_mean_pore_seconds=float(o['pore_seconds'][host].mean()))


def grid_search(margins, y, comp, end, alive):
    """Every (aggregation, r, a, f) on the quantile grid; the last-window threshold f is vectorized (it only acts on
    reads still pending at the last window). Same outcomes as replay() (checked for the selected configuration)."""
    rows = []
    host = ~y; nh, nt = int(host.sum()), int(y.sum()); K = N_WINDOWS - 1
    for how in ('last', 'mean'):
        s = aggregate(margins, how)
        q = np.quantile(s[np.isfinite(s)], QUANTILES)
        grid = np.r_[-np.inf, q, np.inf]
        for r in grid:
            for a in grid:
                if a >= r: continue
                o = replay(s, comp, end, alive, r, a, np.inf, windows=N_WINDOWS)   # f = inf: last window accepts all
                early = o['command'] & (o['window'] < K)
                last = o['window'] == K                                             # reads that reach the last window and act
                s_last = s[last, K]
                rej = s_last[:, None] >= grid[None, :]                              # [L, F]
                eff = rej & (comp[last, K] + EJECT < end[last])[:, None]
                yl = y[last]
                er = o['command_reject'] & early
                host_rej = int((er & host).sum()) + (rej & ~yl[:, None]).sum(0)
                tgt_rej = int((er & y).sum()) + (rej & yl[:, None]).sum(0)
                base_pore = np.where(o['effective_reject'] & early, o['pore_seconds'], end)[host].sum()
                gain = np.where(eff & ~yl[:, None], (end[last] - comp[last, K] - EJECT)[:, None], 0.).sum(0)
                eff_host = int((o['effective_reject'] & early & host).sum()) + (eff & ~yl[:, None]).sum(0)
                for j, f in enumerate(grid):
                    rows.append(dict(how=how, r=float(r), a=float(a), f=float(f), acc=float((host_rej[j] + nt - tgt_rej[j])/len(y)),
                                     host_command_reject=float(host_rej[j]/nh), target_command_frr=float(tgt_rej[j]/nt),
                                     host_effective_reject=float(eff_host[j]/nh), host_mean_pore_seconds=float((base_pore - gain[j])/nh)))
    best = max(rows, key=lambda x: (x['acc'], -x['target_command_frr'], -x['host_mean_pore_seconds']))
    check = stats(replay(aggregate(margins, best['how']), comp, end, alive, best['r'], best['a'], best['f']), y)
    assert all(abs(check[k] - best[k]) < 1e-9 for k in check), ('vectorized grid differs from replay', check, best)
    pts = sorted(rows, key=lambda x: (x['host_mean_pore_seconds'], -x['acc']))
    front, top = [], -1.
    for p in pts:
        if p['acc'] > top: front.append(p); top = p['acc']
    return best, front, len(rows)


def run_cascade(tool, base, data_root, out, device):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'metrics.json').exists(): return
    name = tool.split('_')[0]
    sel = json.loads((base/'selection.json').read_text()); service = float(sel['service_seconds'])
    model = native._load(name, base/'fit'/'best.pt', device)
    c = CONTRACTS[name]
    # DEV-B: margins, selection (locked before TEST)
    _, dev_b = D.dev_halves(data_root)
    mb = window_margins(name, model, dev_b, device)
    yb = np.asarray(dev_b.is_target, bool)
    comp, end, alive = timing(dev_b.raw_length, c['raw_end'], service)
    t0 = time.time(); best, front, n = grid_search(mb, yb, comp, end, alive)
    single = stats(replay(mb, comp, end, alive, np.inf, -np.inf, sel['rules']['max_acc']['threshold']
                          if sel['rules']['max_acc']['threshold'] != 'all_keep' else np.inf, windows=1), yb)
    (out/'selection.json').write_text(json.dumps(dict(tool=tool, service_seconds=service, best=best, single_window_dev_b=single,
                                                      configs=n, front=front, search_seconds=time.time() - t0,
                                                      written=time.strftime('%Y-%m-%d %H:%M:%S') + ', before any TEST input'), indent=2))
    log(tool, 'DEV-B cascade', {k: best[k] for k in ('how', 'r', 'a', 'f', 'acc', 'host_mean_pore_seconds')}, 'single', round(single['acc'], 4))
    # TEST
    test = D.load_split(data_root, 'test', verify=True); y = np.asarray(test.is_target, bool)
    mt = window_margins(name, model, test, device)
    ref = np.load(base/'test'/'scores.npz')
    assert np.array_equal(ref['read_id'], test.read_id)
    diff = np.nanmax(np.abs(ref['margin'] - mt[:, 0]))
    assert np.array_equal(np.isfinite(ref['margin']), np.isfinite(mt[:, 0])) and diff < 1e-3, ('window 0 differs from main scores', diff)
    comp, end, alive = timing(test.raw_length, c['raw_end'], service)
    th = sel['rules']['max_acc']['threshold']; th = np.inf if th == 'all_keep' else float(th)
    o1 = replay(mt, comp, end, alive, np.inf, -np.inf, th, windows=1)
    main = np.load(base/'test'/'outcomes_max_acc_0.4_measured.npz')
    agree = float(np.mean(o1['command_reject'] == main['command_reject']))
    s = aggregate(mt, best['how'])
    o = replay(s, comp, end, alive, best['r'], best['a'], best['f'])
    k_used = o.pop('window')
    curve = [dict(p, test=stats(replay(aggregate(mt, p['how']), comp, end, alive, p['r'], p['a'], p['f']), y)) for p in front]
    np.savez(out/'test_outcomes.npz', read_id=test.read_id, is_target=y, window=k_used, margins=mt, **o)
    res = dict(tool=tool, single_window=stats(o1, y), single_window_agreement_with_main=agree, window0_max_abs_diff=float(diff),
               cascade=dict(stats(o, y), how=best['how'], r=best['r'], a=best['a'], f=best['f'],
                            decided_at_window=[float(np.mean(k_used == k)) for k in range(N_WINDOWS)],
                            no_command=float(np.mean(k_used < 0))), test_curve=curve)
    (out/'metrics.json').write_text(json.dumps(res, indent=2))
    log(tool, 'TEST single', round(res['single_window']['acc'], 4), 'cascade', round(res['cascade']['acc'], 4),
        'host pore', round(res['cascade']['host_mean_pore_seconds'], 3), 'agree', round(agree, 4))


def run_curve(base, data_root, out, fs, device):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'curve.json').exists(): return
    from poreloom.stream_training import load_stream_r0
    choice = json.loads((base/'head'/'selection.json').read_text())
    net = load_stream_r0(base/'stream'/'best.pt', device)[0].eval()
    test = D.load_split(data_root, 'test', verify=True)
    z = H.extract(net, test, fs, H.HEAD['interval'], H.HEAD['phase_seed'], device)
    saved = torch.load(base/'head'/'controllers.pt', map_location='cpu')
    rows = []
    for beta in saved['betas']:
        o = H.evaluate(H.load_controller(base/'head'/'controllers.pt', beta, device), z, fs, device)
        rows.append(dict(beta=float(beta), **H.metrics({k: v for k, v in o.items() if k != 'missing'}, z['is_target'])))
    locked = min(rows, key=lambda r: abs(r['beta'] - choice['beta']))
    want = next(r for r in json.loads((base/'test'/'metrics.json').read_text())['rows']
                if r['interval'] == .4 and r['service'] == 'nominal' and r['group'] == 'pooled')
    check = abs(locked['acc'] - want['acc'])
    dev_b = json.loads((base/'head'/'table.json').read_text())
    (out/'curve.json').write_text(json.dumps(dict(beta=choice['beta'], locked_test_acc=locked['acc'], main_test_acc=want['acc'],
                                                  test=rows, dev_b=dev_b), indent=2))
    log('PoreLoom curve', len(rows), 'betas; locked TEST ACC', round(locked['acc'], 4), 'main', round(want['acc'], 4), 'diff', check)


def run(a):
    torch.set_num_threads(4)
    if a.device.startswith('cuda'): torch.cuda.set_device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    for tool in TOOLS:
        base = a.main/f'{tool}_r{a.rep}'
        if not (base/'JOB_DONE').exists(): log('skip', base, '(not finished)'); continue
        policy.FS = float(json.loads((base/'run.json').read_text())['fs'])
        run_cascade(tool, base, a.data, a.out/f'{tool}_r{a.rep}', a.device)
    base = a.main/f'poreloom_r{a.rep}'
    if (base/'JOB_DONE').exists():
        run_curve(base, a.data, a.out/f'poreloom_r{a.rep}', float(json.loads((base/'run.json').read_text())['fs']), a.device)
    else: log('skip', base, '(not finished)')


def report(a):
    """Per dataset/replicate: PoreLoom locked point; each baseline single window vs cascade; PoreLoom's best TEST ACC
    among betas whose host pore time is <= the cascade's (matched or lower cost)."""
    lines = ['# E2 自适应级联 baseline vs PoreLoom（TEST，0.4 s）', '',
             '每格 ACC % / host 占孔 s。级联 = 同一模型看最多 3 个窗口（最多到 raw 7500，与 PoreLoom 相同），阈值与聚合方式在 DEV-B 上按最大 ACC 锁定。'
             '"PL@同占孔" = PoreLoom beta 曲线上 host 占孔 <= 该级联 baseline 时的最高 TEST ACC。', '',
             '| 数据 | 组 | PoreLoom | 工具 | 单窗口 | 级联 | 级联在第 1/2/3 窗决策 % | PL@同占孔 |', '|---|---|---|---|---|---|---|---|']
    out = []
    for ds in sorted(p for p in a.e2.iterdir() if p.is_dir()):
        for rep in range(3):
            pc = ds/f'poreloom_r{rep}'/'curve.json'
            curve = json.loads(pc.read_text()) if pc.exists() else None
            pl_main = a.main/ds.name/f'poreloom_r{rep}'/'test'/'metrics.json'
            pl = next((r for r in json.loads(pl_main.read_text())['rows'] if r['interval'] == .4 and r['service'] == 'nominal'
                       and r['group'] == 'pooled'), None) if pl_main.exists() else None
            for tool in TOOLS:
                p = ds/f'{tool}_r{rep}'/'metrics.json'
                if not p.exists(): continue
                m = json.loads(p.read_text()); s, c = m['single_window'], m['cascade']
                matched = None
                if curve:
                    ok = [r['acc'] for r in curve['test'] if r['host_mean_pore_seconds'] <= c['host_mean_pore_seconds']]
                    matched = max(ok) if ok else None
                out.append(dict(dataset=ds.name, rep=rep, tool=tool, poreloom=pl, single=s, cascade=c, poreloom_matched_acc=matched))
                fmt = lambda r: f"{100*r['acc']:.1f} / {r['host_mean_pore_seconds']:.2f}"
                lines.append(f"| {ds.name} | r{rep} | {fmt(pl) if pl else '–'} | {tool} | {fmt(s)} | {fmt(c)} | "
                             f"{' / '.join(f'{100*v:.0f}' for v in c['decided_at_window'])} | {'–' if matched is None else f'{100*matched:.1f}'} |")
    text = '\n'.join(lines) + '\n'; print(text)
    if a.md: a.md.write_text(text); a.md.with_suffix('.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('run', 'report'))
    ap.add_argument('--main', type=Path); ap.add_argument('--data', type=Path); ap.add_argument('--out', type=Path)
    ap.add_argument('--e2', type=Path); ap.add_argument('--rep', type=int, default=0); ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--md', type=Path)
    a = ap.parse_args()
    run(a) if a.stage == 'run' else report(a)
