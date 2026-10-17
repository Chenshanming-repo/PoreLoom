#!/usr/bin/env python3
"""PoreLoom ablations.

Every variant changes one part of the default method (poreloom.method, docs/method.md) and keeps everything else: the
same dataset, DEV-A (fit / calibration) and DEV-B (selection) roles, TEST chunk phases (head.HEAD phase_seed), service
0.0158 s per look, eject 0.1 s, operating point by maximum DEV-B command ACC (ties: lower target FRR, then lower host
pore time) locked before TEST is read. Reference: the main run runs/main/<ds>/poreloom_r<rep>.

Steps (one removed at a time)
  window_cp   no streaming training: the stage A window encoder scores the latest 1500-sample window at every chunk
              arrival (MAD-normalized as in stage A; no score before raw 3000 or past the read end); the C-p head is
              trained on that margin trajectory exactly as on the S1 one, its 768-sample time buckets starting at the
              first scored look (stream length 1500: 6 buckets)
  scratch_s1  no stage A: S1 from a seeded random init (backbone lr 1e-3, lr 1e-3, up to 40 epochs, patience 5 = the
              2026-10-14 a3 recipe), then the default head and TEST stages
  const_thr   naive decision on the S1 encoder: constant thresholds on p(target) at every look; reject if p < rho,
              accept if p >= tau, forced look reject if p < phi; (rho, tau, phi) on a DEV-B quantile grid
  calib_thr   length-calibrated thresholds on the S1 encoder (2026-10-14 a4/a5): reject below the alpha_r quantile of
              DEV-A target p at the look's stream length, forced look below alpha_f, accept if p >= tau
  fixed_len   no adaptive stopping: a single decision at the first look with stream length >= L (or the forced look),
              reject if p < phi; (L, phi) on DEV-B
  window_thr  both removed: window encoder + constant thresholds (as const_thr)
Parameters
  budget_B    C-p with the forced decision at B in {3000, 4500, 6000} samples (default 7500); S1 encoder unchanged; the
              768-sample time buckets stop where the budget does (2 / 4 / 6 buckets; 7500 -> the default 8)
  window_W    stage A window W in {1000, 3000} (default 1500); stage A, S1, head and TEST retrained (sputum_g1 and
              r10_run01, replicate 0)
Chunk time (offline part; the live part is icarust_chunk.py)
  cadence     C-p trained at 0.2 s chunks (default 0.4 s); the 0.2 s and the 0.4 s heads replayed on TEST at 0.2 / 0.4 /
              0.8 / 1.0 s
Per variant: the locked TEST point (0.4 s, nominal service), the paired difference to the main run (bootstrap over
reads) and the TEST replay of its DEV-B Pareto front (C-p: every beta of the grid).

    ablation.py register                                   registration.json (before any job)
    ablation.py steps   --main <runs>/main/<ds> --data <data>/<ds> --rep r --out <runs>/ablation/<ds>/r<r>
    ablation.py scratch --data <data>/<ds> --fs F --rep r --out <runs>/ablation/<ds>/r<r>/scratch_s1
    ablation.py window  --data <data>/<ds> --fs F --window W --out <runs>/ablation/<ds>/r0/window_<W>
    ablation.py report  --root <runs>/ablation --main <runs>/main [--md summary_zh.md]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO/'src'))
from poreloom.chunk_replay import calibrated_thresholds, replay, threshold_actions  # noqa: E402
from poreloom.method import data as D, head as H, pipeline as P  # noqa: E402
from poreloom.method.pretrain import STAGEA, train_window_encoder  # noqa: E402
from poreloom.stream_controller import encode_looks, look_grid  # noqa: E402

SKIP, BUDGET, STRIDE = 1500, 7500, 48
SCRATCH_S1 = dict(backbone_lr=1e-3, lr=1e-3, epochs=40, patience=5)
BUDGETS = (3000, 4500, 6000)
WINDOWS = (1000, 3000)
CADENCE_TRAIN, CADENCE_TEST = (.2,), (.2, .4, .8, 1.)
QUANTILES = np.linspace(.025, .975, 39)
ALPHAS = np.concatenate([[0.], np.geomspace(1e-4, .5, 40)])
TAUS_A = [.5, .8, .9, .95, .98, .99, .995, .999, None]
CALIB_USED = np.unique(np.r_[np.arange(432, 6000, 96), 6000])
FIXED_USED = (432, 1200, 1968, 2736, 3504, 4272, 5040, 6000)
SERVICE, EJECT = H.HEAD['service'], H.HEAD['eject']
NAMES = dict(main='PoreLoom（完整）', window_cp='去掉流式训练（stage A 滑窗 + C-p）', scratch_s1='去掉 stage A（S1 随机初始化）',
             const_thr='C-p → 常数置信度阈值', calib_thr='C-p → 按长度校准阈值', fixed_len='C-p → 固定长度决策',
             window_thr='去掉流式训练和 C-p（滑窗 + 常数阈值）',
             **{f'budget_{b}': f'信号预算 {b - SKIP}（读到第 {b} 点）' for b in BUDGETS}, **{f'window_{w}': f'stage A 窗口 {w}' for w in WINDOWS})


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=float) + '\n')


def strip(o):
    return {k: v for k, v in o.items() if k != 'missing'}


def stats(o, y):
    y = np.asarray(y, bool)
    return dict(acc=H.command_acc(o, y), host_command_reject=float(o['command_reject'][~y].mean()),
                target_command_frr=float(o['command_reject'][y].mean()),
                host_effective_reject=float(o['effective_reject'][~y].mean()),
                host_mean_pore_seconds=float(o['pore_seconds'][~y].mean()))


def key(r):
    return r['acc'], -r['target_command_frr'], -r['host_mean_pore_seconds']


def front(rows):
    """Pareto front of rows (higher ACC, lower host pore time), ordered by pore time."""
    out, top = [], -1.
    for r in sorted(rows, key=lambda x: (x['host_mean_pore_seconds'], -x['acc'])):
        if r['acc'] > top: out.append(r); top = r['acc']
    return out


def save_outcome(path, z, o):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, read_id=z['read_id'], is_target=z['is_target'], **strip(o))


# ------------------------------------------------------------------ trajectories
def cached(path, make):
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as f:
            z = {k: f[k] for k in f.files}
        for k in ('budget', 'n_buckets', 'min_samples'):
            if k in z: z[k] = int(z[k])
        return z
    z = make(); z['read_id'] = np.asarray(z['read_id']).astype(str); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.tmp.npz'); np.savez(tmp, **z); tmp.rename(path); return z


def truncate(z, budget, forced_ell):
    """The same chunk phases with the forced decision at `budget`: arrivals after the first one >= budget dropped, and
    that forced look scored on the first `budget` samples (forced_ell [N], as the default 7500 cap of StreamR0)."""
    m = z['m'].copy(); ell = z['ell'].copy()
    first = np.argmax(np.nan_to_num(m, nan=-1.) >= budget, 1)
    late = np.arange(m.shape[1])[None] > first[:, None]
    m[late] = np.nan; ell[late] = np.nan
    ell[np.arange(len(m)), first] = forced_ell
    keep = int(np.max(first)) + 1
    return dict(z, m=m[:, :keep], ell=ell[:, :keep], budget=budget, n_buckets=budget_buckets(budget))


def forced_margin(net, data, budget, device):
    """S1 margin of every read at exactly `budget` samples."""
    return encode_looks(net, data, np.full((len(data), 1), budget, np.int64), device)[1][:, 0]


def budget_buckets(budget, min_samples=432, width=768):
    """C-p time buckets that a budget reaches, at the default 768-sample width (7500 -> 8, the default)."""
    return int(np.ceil((budget - SKIP - min_samples)/width))


def load_stage_a(path, device):
    from poreloom.model import build_model
    ck = torch.load(path, map_location='cpu')
    model = build_model(None, max_windows=STAGEA['max_windows']); model.load_state_dict(ck['model_state'])
    temporary = torch.nn.Linear(128, 2); temporary.load_state_dict(ck['temporary_head'])
    return model.to(device).eval(), temporary.to(device).eval()


@torch.no_grad()
def window_extract(stage_a, data, fs, interval, seed, device, window=STAGEA['window'], batch=2048):
    """As head.extract, with ell = stage A margin (target - host) of raw[e - window:e], e = min(n, 7500), at each arrival
    n; nan when the window starts before raw 1500, passes the read end or the prefix cache, or has MAD 0."""
    from poreloom.chunk_replay import arrivals
    model, temporary = stage_a
    u = np.random.default_rng(seed).random(len(data))
    m = arrivals(u*interval*fs, interval*fs, BUDGET)
    n = np.where(np.isfinite(m), np.floor(m), -1).astype(np.int64)
    ell = np.full(n.shape, np.nan, np.float32)
    end = np.minimum(n, BUDGET)                   # forced look: the last window inside the 7500-sample budget, as S1
    lo = end - window
    ok = (n >= 0) & (lo >= STAGEA['window_start']) & (end <= np.asarray(data.raw_length)[:, None]) & (end <= data.adc.shape[1])
    rows, cols = np.nonzero(ok)
    order = np.argsort(rows, kind='stable'); rows, cols = rows[order], cols[order]
    for s in range(0, len(rows), batch):
        r, c = rows[s:s + batch], cols[s:s + batch]
        start = lo[r, c]
        block = np.asarray(data.adc[r.min():r.max() + 1])
        x = block[(r - r.min())[:, None], start[:, None] + np.arange(window)]
        zx, valid = D.mad_normalize(x)
        if not valid.any(): continue
        logits = temporary(model.encode(torch.as_tensor(zx[valid], device=device)[:, None])).float().cpu().numpy()
        ell[r[valid], c[valid]] = logits[:, 1] - logits[:, 0]
    return dict(read_id=data.read_id, is_target=np.asarray(data.is_target, bool), raw_length=data.raw_length, m=m, ell=ell,
                budget=BUDGET, min_samples=window, n_buckets=int(np.ceil((BUDGET - SKIP - window)/768)))


# ------------------------------------------------------------------ C-p
def cp_train(fit, sel, fs, out, device, rep):
    out = Path(out)
    if not (out/'selection.json').exists():
        choice = H.train_heads(fit, sel, fs, out, device, log=log, train_seed=H.HEAD['train_seed'] + rep)
        choice['written'] = time.strftime('%Y-%m-%d %H:%M:%S') + ', before any TEST input'
        write(out/'selection.json', choice)
    return json.loads((out/'selection.json').read_text())


def cp_test(head_dir, z, fs, device, out_npz=None):
    """Every beta replayed on z: (rows, locked row); the locked outcome is saved to out_npz."""
    head_dir = Path(head_dir)
    choice = json.loads((head_dir/'selection.json').read_text())
    betas = torch.load(head_dir/'controllers.pt', map_location='cpu')['betas']
    rows, locked = [], None
    for beta in betas:
        o = strip(H.evaluate(H.load_controller(head_dir/'controllers.pt', beta, device), z, fs, device))
        row = dict(beta=float(beta), **stats(o, z['is_target'])); rows.append(row)
        if abs(beta - choice['beta']) < 1e-12:
            locked = row
            if out_npz: save_outcome(out_npz, z, o)
    assert locked is not None
    return rows, locked


# ------------------------------------------------------------------ threshold rules
def look_arrays(z, fs):
    n, t, valid, forced, _ = look_grid(z['m'], fs=fs, max_samples=z['budget'])
    ok = np.isfinite(z['ell'])
    p = np.where((valid | forced) & ok, 1/(1 + np.exp(-np.nan_to_num(z['ell']).astype(np.float64))), np.nan)
    used = (np.clip(n, SKIP, z['budget']) - SKIP)//STRIDE*STRIDE
    return dict(t=t, p=p, forced=forced & ok, used=used, t_end=np.asarray(z['raw_length'])/fs)


def run_actions(L, actions, service=SERVICE):
    return replay(L['t'] + service, actions, L['t'], service, L['t_end'], EJECT)


def const_actions(L, rho, tau, phi):
    return threshold_actions(L['p'], L['forced'], rho, phi, tau)


def calib_actions(q):
    return lambda L, ir, iff, tau: threshold_actions(L['p'], L['forced'], calibrated_thresholds(L['used'], CALIB_USED, q[:, ir]),
                                                     calibrated_thresholds(L['used'], CALIB_USED, q[:, iff]), np.inf if tau is None else tau)


def fixed_actions(L, used_at, phi):
    """One decision at the first look with stream length >= used_at, or at the forced look if that comes first."""
    can = np.isfinite(L['p'])
    at = can & ((L['used'] >= used_at) | L['forced'])
    first = at & (np.cumsum(at, 1) == 1)
    p = np.where(first, L['p'], np.nan)
    return threshold_actions(p, first, -np.inf, phi, np.inf)


def const_grid(L):
    q = np.quantile(L['p'][np.isfinite(L['p'])], QUANTILES)
    lows, highs = np.r_[-np.inf, q], np.r_[q, np.inf]
    return [(float(r), float(t), float(f)) for r in lows for t in highs if r < t for f in np.r_[-np.inf, q, np.inf]]


def calib_quantiles(net, dev_a, device):
    _, ell = encode_looks(net, dev_a, np.tile(SKIP + CALIB_USED, (len(dev_a), 1)), device)
    p = 1/(1 + np.exp(-ell.astype(np.float64)))
    alive = np.asarray(dev_a.raw_length)[:, None] >= SKIP + CALIB_USED
    return np.stack([np.quantile(p[alive[:, g] & dev_a.is_target & np.isfinite(p[:, g]), g], ALPHAS) for g in range(len(CALIB_USED))])


def rule_search(name, configs, actions_of, Ls, Lt, y_sel, y_test, out, z_test):
    """Every config on DEV-B; the max-ACC config locked (selection.json) before TEST; the DEV-B front replayed on TEST."""
    out = Path(out)
    if (out/'metrics.json').exists(): return json.loads((out/'metrics.json').read_text())
    t0 = time.time(); rows = []
    for c in configs:
        rows.append(dict(config=c, **stats(run_actions(Ls, actions_of(Ls, *c)), y_sel)))
    best = max(rows, key=key)
    write(out/'selection.json', dict(variant=name, best=best, configs=len(rows), search_seconds=time.time() - t0,
                                     written=time.strftime('%Y-%m-%d %H:%M:%S') + ', before any TEST input'))
    log(name, 'DEV-B', {k: best[k] for k in ('config', 'acc', 'host_mean_pore_seconds')}, f'{len(rows)} configs {time.time() - t0:.0f}s')
    o = run_actions(Lt, actions_of(Lt, *best['config'])); save_outcome(out/'test_outcomes_0.4.npz', z_test, o)
    curve = [dict(config=r['config'], dev_b=dict((k, r[k]) for k in r if k != 'config'),
                  **stats(run_actions(Lt, actions_of(Lt, *r['config'])), y_test)) for r in front(rows)]
    res = dict(variant=name, config=best['config'], locked=stats(o, y_test), test_curve=curve)
    write(out/'metrics.json', res)
    log(name, 'TEST', {k: round(v, 4) for k, v in res['locked'].items()})
    return res


# ------------------------------------------------------------------ stages
def steps(a):
    P._setup(a.device)
    base = a.main/f'poreloom_r{a.rep}'
    fs = float(json.loads((base/'run.json').read_text())['fs'])
    out = a.out; cache = out/'cache'
    write(out/'run.json', dict(main=str(base), data=str(a.data), rep=a.rep, fs=fs, started=time.strftime('%Y-%m-%d %H:%M:%S')))
    dev_a, dev_b = D.dev_halves(a.data)
    test = D.load_split(a.data, 'test', verify=True)
    if a.limit:                                   # smoke test only: evenly spaced rows of each split, no reproduction check
        dev_a, dev_b, test = (D.subset(x, np.unique(np.linspace(0, len(x) - 1, a.limit).astype(int))) for x in (dev_a, dev_b, test))
    y = np.asarray(test.is_target, bool)
    net = P._encoder(base, a.device)
    ph, cp = H.HEAD['phase_seed'], H.HEAD['copy_seed']

    def stream_z(data, tag, interval, seed):
        return cached(cache/f'stream_{tag}_{interval}_{seed}.npz', lambda: H.extract(net, data, fs, interval, seed, a.device))

    fit = [stream_z(dev_a, 'deva', .4, s) for s in (ph, cp)]; sel = stream_z(dev_b, 'devb', .4, ph)
    zt = stream_z(test, 'test', .4, ph)

    # reference: the main head on the recomputed TEST trajectory must reproduce test/metrics.json
    if not (out/'main'/'metrics.json').exists():
        rows, locked = cp_test(base/'head', zt, fs, a.device, out/'main'/'test_outcomes_0.4.npz')
        want = next(r for r in json.loads((base/'test'/'metrics.json').read_text())['rows']
                    if r['interval'] == .4 and r['service'] == 'nominal' and r['group'] == 'pooled')
        diff = abs(locked['acc'] - want['acc']); assert a.limit or diff < 1e-9, ('main TEST not reproduced', locked['acc'], want['acc'])
        write(out/'main'/'metrics.json', dict(variant='main', locked=locked, test_curve=rows, main_test_acc=want['acc']))
        log('main reproduced', round(locked['acc'], 4))

    # naive decision layers on the S1 encoder
    Ls, Lt = look_arrays(sel, fs), look_arrays(zt, fs)
    rule_search('const_thr', const_grid(Ls), const_actions, Ls, Lt, sel['is_target'], y, out/'const_thr', zt)
    qpath = cache/'calib_quantiles.npy'
    if not qpath.exists(): np.save(qpath, calib_quantiles(net, dev_a, a.device))
    q = np.load(qpath)
    rule_search('calib_thr', [(i, j, t) for i in range(len(ALPHAS)) for j in range(len(ALPHAS)) for t in TAUS_A],
                calib_actions(q), Ls, Lt, sel['is_target'], y, out/'calib_thr', zt)
    qs = np.quantile(Ls['p'][np.isfinite(Ls['p'])], QUANTILES)
    rule_search('fixed_len', [(u, float(f)) for u in FIXED_USED for f in np.r_[-np.inf, qs, np.inf]], fixed_actions,
                Ls, Lt, sel['is_target'], y, out/'fixed_len', zt)

    # decision budget (same encoder and chunk phases, forced decision earlier)
    for b in BUDGETS:
        d = out/f'budget_{b}'
        if (d/'metrics.json').exists(): continue
        fa, fb, ft = (forced_margin(net, x, b, a.device) for x in (dev_a, dev_b, test))
        cp_train([truncate(f, b, fa) for f in fit], truncate(sel, b, fb), fs, d/'head', a.device, a.rep)
        rows, locked = cp_test(d/'head', truncate(zt, b, ft), fs, a.device, d/'test_outcomes_0.4.npz')
        write(d/'metrics.json', dict(variant=f'budget_{b}', locked=locked, test_curve=rows)); log(f'budget {b}', round(locked['acc'], 4))

    # chunk time: heads trained at 0.2 s (and the main 0.4 s head) replayed at every TEST chunk time
    d = out/'cadence'
    if not (d/'metrics.json').exists():
        heads = {.4: base/'head'}
        for tr in CADENCE_TRAIN:
            heads[tr] = d/f'head_{tr}'
            cp_train([stream_z(dev_a, 'deva', tr, s) for s in (ph, cp)], stream_z(dev_b, 'devb', tr, ph), fs, heads[tr], a.device, a.rep)
        cells = []
        for tr, hd in sorted(heads.items()):
            for te in CADENCE_TEST:
                rows, locked = cp_test(hd, stream_z(test, 'test', te, ph), fs, a.device, d/f'test_outcomes_train{tr}_test{te}.npz')
                cells.append(dict(train=tr, test=te, locked=locked)); log('cadence train', tr, 'test', te, round(locked['acc'], 4))
        write(d/'metrics.json', dict(cells=cells))

    # window encoder: with C-p (no streaming training) and with constant thresholds (neither)
    stage_a = load_stage_a(base/'stageA'/'stage_A.pt', a.device)
    wz = lambda data, tag, seed: cached(cache/f'window_{tag}_0.4_{seed}.npz', lambda: window_extract(stage_a, data, fs, .4, seed, a.device))
    wfit = [wz(dev_a, 'deva', s) for s in (ph, cp)]; wsel = wz(dev_b, 'devb', ph); wt = wz(test, 'test', ph)
    assert np.array_equal(np.isfinite(wt['m']), np.isfinite(zt['m'])) and np.allclose(np.nan_to_num(wt['m']), np.nan_to_num(zt['m']))
    d = out/'window_cp'
    if not (d/'metrics.json').exists():
        cp_train(wfit, wsel, fs, d/'head', a.device, a.rep)
        rows, locked = cp_test(d/'head', wt, fs, a.device, d/'test_outcomes_0.4.npz')
        write(d/'metrics.json', dict(variant='window_cp', locked=locked, test_curve=rows)); log('window_cp', round(locked['acc'], 4))
    Lws, Lwt = look_arrays(wsel, fs), look_arrays(wt, fs)
    rule_search('window_thr', const_grid(Lws), const_actions, Lws, Lwt, wsel['is_target'], y, out/'window_thr', wt)
    (out/'STEPS_DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))


def finish_run(a):
    """Default head and TEST stages on a.out, then the TEST beta curve."""
    for st in ('head', 'test'): log(st); getattr(P, st)(a)
    if not (a.out/'metrics.json').exists():
        test = D.load_split(a.data, 'test', verify=True)
        zt = H.extract(P._encoder(a.out, a.device), test, a.fs, .4, H.HEAD['phase_seed'], a.device)
        rows, locked = cp_test(a.out/'head', zt, a.fs, a.device, a.out/'test_outcomes_0.4.npz')
        write(a.out/'metrics.json', dict(variant=a.out.name, locked=locked, test_curve=rows))
    (a.out/'JOB_DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))


def scratch(a):
    from poreloom.model import build_model
    from poreloom.stream_training import R0_DEFAULTS, train_stream_r0
    P._setup(a.device); a.out.mkdir(parents=True, exist_ok=True); a.select = 'max_acc'; a.replicate = a.rep
    if not (a.out/'stream'/'done.json').exists():
        init = a.out/'random_init.pt'
        if not init.exists():
            torch.manual_seed(STAGEA['seed'] + a.rep)
            torch.save(dict(model_state=build_model(None).state_dict(), temporary_head=torch.nn.Linear(128, 2).state_dict(),
                            initialization='random', pretrained=False), init)
        train = D.load_split(a.data, 'train'); dev_a, _ = D.dev_halves(a.data)
        train_stream_r0(train, dev_a, init, a.out/'stream', device=a.device, log=log, seed=R0_DEFAULTS['seed'] + a.rep, **SCRATCH_S1)
        best = torch.load(a.out/'stream'/'best.pt', map_location='cpu')
        write(a.out/'stream'/'done.json', dict(best_epoch=int(best['epoch'])))
    finish_run(a)


def window(a):
    """Stage A with a W-sample window. PoreLoom.encode only accepts 1500-sample windows; the network itself (CNN with
    global average pooling + projection) takes any length, so encode is relaxed here, for this job only."""
    from poreloom import model as M
    M.PoreLoom.encode = lambda self, x: self.projection(self.encoder(x).mean(-1))
    P._setup(a.device); a.out.mkdir(parents=True, exist_ok=True); a.select = 'max_acc'; a.replicate = 0
    if not (a.out/'stageA'/'stageA_history.json').exists():
        train = D.load_split(a.data, 'train'); dev_a, _ = D.dev_halves(a.data)
        train_window_encoder(train, dev_a, a.out/'stageA', a.device, log=log, window=a.window)
    P.stream(a); finish_run(a)


FAMILY = (('r10', 'R10'), ('r9', 'R9'), ('sim', 'Sim'), ('sputum', 'Sputum'))
STEP_VARIANTS = ('window_cp', 'scratch_s1', 'const_thr', 'calib_thr', 'fixed_len', 'window_thr')
PARAM_VARIANTS = tuple(f'budget_{b}' for b in BUDGETS) + tuple(f'window_{w}' for w in WINDOWS)


def unit_rows(unit, rng):
    """One row per variant of a unit directory: locked TEST point, paired difference to the main run (bootstrap over
    reads; pore time over host reads), ACC at the main run's host pore time and pore time at the main run's ACC (curves)."""
    ref = json.loads((unit/'main'/'metrics.json').read_text())['locked']
    with np.load(unit/'main'/'test_outcomes_0.4.npz') as f: r = {k: f[k] for k in f.files}
    y = r['is_target'].astype(bool); host = np.flatnonzero(~y)
    corr = lambda o: np.where(y, ~o['command_reject'], o['command_reject']).astype(float)
    out = []
    for v in ('main',) + STEP_VARIANTS + PARAM_VARIANTS:
        d = unit/v
        mpath = d/'metrics.json'
        if not mpath.exists(): continue
        m = json.loads(mpath.read_text()); row = dict(variant=v, **m['locked'])
        if v != 'main':
            with np.load(d/'test_outcomes_0.4.npz') as f: o = {k: f[k] for k in f.files}
            assert np.array_equal(o['read_id'], r['read_id'])
            da = corr(o) - corr(r); dp = (o['pore_seconds'] - r['pore_seconds'])[host]
            ia = rng.integers(0, len(da), (2000, len(da))); ip = rng.integers(0, len(dp), (2000, len(dp)))
            row.update(dacc=float(da.mean()), dacc_ci=np.percentile(da[ia].mean(1), [2.5, 97.5]).tolist(),
                       dpore=float(dp.mean()), dpore_ci=np.percentile(dp[ip].mean(1), [2.5, 97.5]).tolist())
        curve = m['test_curve']
        ok = [c['acc'] for c in curve if c['host_mean_pore_seconds'] <= ref['host_mean_pore_seconds'] + 1e-12]
        row['acc_at_main_pore'] = max(ok) if ok else None
        ok = [c['host_mean_pore_seconds'] for c in curve if c['acc'] >= ref['acc'] - 1e-12]
        row['pore_at_main_acc'] = min(ok) if ok else None
        out.append(row)
    return out


def report(a):
    rng = np.random.default_rng(20261015)
    units = sorted(p.parent.parent for p in a.root.glob('*/r*/main/metrics.json'))
    per = {f'{u.parent.name}/{u.name}': unit_rows(u, rng) for u in units}
    fam = lambda name: next(label for prefix, label in FAMILY if name.startswith(prefix))
    pct = lambda x: '–' if x is None else f'{100*x:.1f}'
    sec = lambda x: '–' if x is None else f'{x:.2f}'
    lines = ['# PoreLoom 消融（TEST，0.4 s，DEV-B 最大 ACC 锁定）', '',
             '每格：各数据集族 3 个单元的均值（R10 = 3 个 run，R9 = 3 个重复，Sim / Sputum = 3 个组）。'
             'ΔACC / Δ占孔 = 与同一单元完整 PoreLoom 按 read 配对之差的均值，括号为 3 个单元中 95% CI 不含 0 的个数。'
             '"@PL 占孔" = 该变体 TEST 曲线上 host 占孔 ≤ PoreLoom 时的最高 ACC；"达 PL ACC 的占孔" = 曲线上 ACC ≥ PoreLoom 时的最短占孔（– = 达不到）。', '']
    for title, variants in (('步骤消融', ('main',) + STEP_VARIANTS), ('参数消融', ('main',) + PARAM_VARIANTS)):
        lines += [f'## {title}', '', '| 数据 | 版本 | ACC % | target 误拒 % | host 占孔 s | ΔACC pp | Δhost 占孔 s | @PL 占孔 ACC % | 达 PL ACC 的占孔 s |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for _, label in FAMILY:
            for v in variants:
                rs = [r for u, rows in per.items() if fam(u) == label for r in rows if r['variant'] == v]
                if not rs: continue
                mean = lambda k: None if any(r.get(k) is None for r in rs) else float(np.mean([r[k] for r in rs]))
                sig = lambda k: sum((r[f'{k}_ci'][0] > 0) or (r[f'{k}_ci'][1] < 0) for r in rs)
                dacc = '' if v == 'main' else f"{100*mean('dacc'):+.2f} ({sig('dacc')}/{len(rs)})"
                dpore = '' if v == 'main' else f"{mean('dpore'):+.3f} ({sig('dpore')}/{len(rs)})"
                lines.append(f"| {label} | {NAMES[v]} (n={len(rs)}) | {pct(mean('acc'))} | {pct(mean('target_command_frr'))} | "
                             f"{sec(mean('host_mean_pore_seconds'))} | {dacc} | {dpore} | {pct(mean('acc_at_main_pore'))} | "
                             f"{'–' if any(r['pore_at_main_acc'] is None for r in rs) else sec(mean('pore_at_main_acc'))} |")
        lines.append('')
    cad = [(u, json.loads((a.root/u/'cadence'/'metrics.json').read_text())['cells']) for u in per if (a.root/u/'cadence'/'metrics.json').exists()]
    if cad:
        lines += ['## chunk 时间（离线回放）', '', '每格 ACC % / host 占孔 s，各数据集族 3 个单元的均值。行 = 动作头训练节奏，列 = TEST 回放节奏。', '',
                  '| 数据 | 训练 | ' + ' | '.join(f'测 {t:g} s' for t in CADENCE_TEST) + ' |', '|---|---|' + '---:|'*len(CADENCE_TEST)]
        for _, label in FAMILY:
            for tr in sorted({c['train'] for _, cs in cad for c in cs}):
                cells = []
                for te in CADENCE_TEST:
                    xs = [c['locked'] for u, cs in cad if fam(u) == label for c in cs if c['train'] == tr and c['test'] == te]
                    cells.append(f"{100*np.mean([x['acc'] for x in xs]):.1f} / {np.mean([x['host_mean_pore_seconds'] for x in xs]):.2f}" if xs else '–')
                if any(c != '–' for c in cells): lines.append(f'| {label} | {tr:g} s | ' + ' | '.join(cells) + ' |')
        lines.append('')
    text = '\n'.join(lines) + '\n'; print(text)
    write(a.root/'report.json', per)
    if a.md: a.md.write_text(text)


def register(a):
    path = HERE/'registration.json'
    if path.exists(): raise FileExistsError(path)
    write(path, dict(written=time.strftime('%Y-%m-%d %H:%M:%S') + ', before any ablation job', design=__doc__,
                     scratch_s1=SCRATCH_S1, budgets=BUDGETS, windows=WINDOWS, window_datasets=['sputum_g1', 'r10_run01'],
                     cadence_train=CADENCE_TRAIN, cadence_test=CADENCE_TEST, const_quantiles=QUANTILES.tolist(),
                     alphas=ALPHAS.tolist(), taus=TAUS_A, calib_used=CALIB_USED.tolist(), fixed_used=FIXED_USED,
                     service=SERVICE, eject=EJECT, bootstrap=dict(resamples=2000, seed=20261015),
                     units=['r10_run01/r0', 'r10_run02/r0', 'r10_run03/r0', 'r9/r0', 'r9/r1', 'r9/r2', 'sim_g1/r0', 'sim_g2/r0',
                            'sim_g3/r0', 'sputum_g1/r0', 'sputum_g2/r0', 'sputum_g3/r0']))
    print(path.read_text())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('register', 'steps', 'scratch', 'window', 'report'))
    ap.add_argument('--main', type=Path); ap.add_argument('--data', type=Path); ap.add_argument('--out', type=Path)
    ap.add_argument('--root', type=Path); ap.add_argument('--md', type=Path)
    ap.add_argument('--rep', type=int, default=0); ap.add_argument('--fs', type=float); ap.add_argument('--window', type=int)
    ap.add_argument('--device', default='cuda:0'); ap.add_argument('--limit', type=int, default=0, help='smoke test: rows per split')
    a = ap.parse_args()
    globals()[a.stage](a)
