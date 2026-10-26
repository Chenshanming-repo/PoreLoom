#!/usr/bin/env python3
"""Encoding and stopping controls reported in the manuscript and supplement.

Every control retains the main datasets, frozen operating-point selection on DEV-B,
0.4-s TEST observations, nominal processing delay and 0.1-s ejection delay.
  window_cp: the stage-A encoder scores the latest 1500 samples, with its own fitted FACP.
  const_thr: constant rejection/acceptance bounds on the frozen streaming encoder.
  calib_thr: length-calibrated rejection bounds on the same frozen encoder.
  fixed_len: one DEV-B-selected length and threshold on the same frozen encoder.
The main FACP reference replays all 41 penalties; stopping-rule candidates retain the
DEV-B accuracy--time frontier needed for the supplementary matched-rejection analysis.

    ablation.py steps --main <runs>/main/<ds> --data <data>/<ds> --rep r --out <runs>/ablation/<ds>/r<r>
    ablation.py report --root <runs>/ablation --main <runs>/main [--md summary_zh.md]
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
from poreloom.method.pretrain import STAGEA  # noqa: E402
from poreloom.stream_controller import encode_looks, look_grid  # noqa: E402

SKIP, BUDGET, STRIDE = 1500, 7500, 48
QUANTILES = np.linspace(.025, .975, 39)
ALPHAS = np.concatenate([[0.], np.geomspace(1e-4, .5, 40)])
TAUS_A = [.5, .8, .9, .95, .98, .99, .995, .999, None]
CALIB_USED = np.unique(np.r_[np.arange(432, 6000, 96), 6000])
FIXED_USED = (432, 1200, 1968, 2736, 3504, 4272, 5040, 6000)
SERVICE, EJECT = H.HEAD['service'], H.HEAD['eject']
NAMES = dict(main='PoreLoom', window_cp='Sliding window + FACP',
             const_thr='Constant threshold', calib_thr='Length-calibrated threshold', fixed_len='Fixed length')


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

    sel = stream_z(dev_b, 'devb', .4, ph)
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

    # Sliding-window encoder with its separately fitted FACP.
    stage_a = load_stage_a(base/'stageA'/'stage_A.pt', a.device)
    wz = lambda data, tag, seed: cached(cache/f'window_{tag}_0.4_{seed}.npz', lambda: window_extract(stage_a, data, fs, .4, seed, a.device))
    wfit = [wz(dev_a, 'deva', s) for s in (ph, cp)]; wsel = wz(dev_b, 'devb', ph); wt = wz(test, 'test', ph)
    assert np.array_equal(np.isfinite(wt['m']), np.isfinite(zt['m'])) and np.allclose(np.nan_to_num(wt['m']), np.nan_to_num(zt['m']))
    d = out/'window_cp'
    if not (d/'metrics.json').exists():
        cp_train(wfit, wsel, fs, d/'head', a.device, a.rep)
        rows, locked = cp_test(d/'head', wt, fs, a.device, d/'test_outcomes_0.4.npz')
        write(d/'metrics.json', dict(variant='window_cp', locked=locked, test_curve=rows)); log('window_cp', round(locked['acc'], 4))
    (out/'STEPS_DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))


FAMILY = (('r10', 'R10'), ('r9', 'R9'), ('sim', 'Sim'), ('sputum', 'Sputum'))
STEP_VARIANTS = ('window_cp', 'const_thr', 'calib_thr', 'fixed_len')


def unit_rows(unit):
    """Locked TEST metrics and descriptive differences from the unit's main model."""
    ref = json.loads((unit/'main'/'metrics.json').read_text())['locked']
    rows = []
    for variant in ('main',) + STEP_VARIANTS:
        path = unit/variant/'metrics.json'
        if not path.exists():
            continue
        metrics = json.loads(path.read_text())['locked']
        row = dict(variant=variant, **metrics)
        if variant != 'main':
            row.update(dacc=metrics['acc'] - ref['acc'],
                       dpore=metrics['host_mean_pore_seconds'] - ref['host_mean_pore_seconds'])
        rows.append(row)
    return rows


def report(a):
    units = sorted(p.parent.parent for p in a.root.glob('*/r*/main/metrics.json'))
    per = {f'{u.parent.name}/{u.name}': unit_rows(u) for u in units}
    family = lambda name: next(label for prefix, label in FAMILY if name.startswith(prefix))
    lines = ['# Encoding and stopping controls', '',
             'TEST at 0.4-s intervals; each operating point was selected on DEV-B.', '',
             '| Dataset | Control | Units | Accuracy (%) | Target rejection (%) | Host pore time (s) |',
             '|---|---|---:|---:|---:|---:|']
    for _, label in FAMILY:
        for variant in ('main',) + STEP_VARIANTS:
            rows = [r for unit, values in per.items() if family(unit) == label
                    for r in values if r['variant'] == variant]
            if not rows:
                continue
            mean = lambda key: float(np.mean([r[key] for r in rows]))
            lines.append(f"| {label} | {NAMES[variant]} | {len(rows)} | {100*mean('acc'):.2f} | "
                         f"{100*mean('target_command_frr'):.2f} | {mean('host_mean_pore_seconds'):.3f} |")
    text = '\n'.join(lines) + '\n'
    print(text)
    write(a.root/'report.json', per)
    if a.md:
        a.md.write_text(text)


def register(a):
    path = HERE/'registration.json'
    if path.exists():
        raise FileExistsError(path)
    write(path, dict(written=time.strftime('%Y-%m-%d %H:%M:%S'), design=__doc__,
                     variants=list(STEP_VARIANTS), const_quantiles=QUANTILES.tolist(),
                     alphas=ALPHAS.tolist(), taus=TAUS_A, calib_used=CALIB_USED.tolist(),
                     fixed_used=FIXED_USED, service=SERVICE, eject=EJECT))
    print(path.read_text())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('register', 'steps', 'report'))
    ap.add_argument('--main', type=Path); ap.add_argument('--data', type=Path); ap.add_argument('--out', type=Path)
    ap.add_argument('--root', type=Path); ap.add_argument('--md', type=Path)
    ap.add_argument('--rep', type=int, default=0); ap.add_argument('--fs', type=float)
    ap.add_argument('--device', default='cuda:0'); ap.add_argument('--limit', type=int, default=0, help='smoke test: rows per split')
    a = ap.parse_args()
    globals()[a.stage](a)
