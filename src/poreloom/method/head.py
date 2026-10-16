"""PoreLoom action head (C-p): a bucketed factorized controller on the streaming encoder's logit trajectory.

Looks: chunk arrivals every `interval` seconds with a per-read random phase; at each arrival the frozen StreamR0 gives
its logit margin ell(n). C-p sees (ell, n/budget, log1p(n/fs), previous ell, minimum ell, look count) — no hidden
state. A read acts at most once per bucket (later looks in the same bucket wait, as next_look_used does live), so
any chunk interval works. Training (poreloom.stream_controller.train_backward): FIT = DEV-A with two phase draws, SEL = DEV-B as the stop
set; one controller per beta on the M3b grid. Selection: maximum SEL command ACC, ties lower SEL target FRR.
Times are samples / fs (fs = the data's sampling rate); every length is in samples.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..chunk_replay import arrivals, summarize
from ..stream_controller import (BucketController, Looks, encode_looks, look_grid, look_inputs, one_look_per_bucket, outcome,
                                 policy_actions, train_backward)
from ..stream_config import (DEFAULT_FS, DEFAULT_MAX_SAMPLES, DEFAULT_SKIP, DEFAULT_MIN_SAMPLES,
                             DEFAULT_STRIDE, DEFAULT_BUCKET_SAMPLES, bucket_count, sampling_rate)

HEAD = dict(interval=.4, budget=DEFAULT_MAX_SAMPLES, phase_seed=20261012, copy_seed=20261013, train_seed=20261013, lr=3e-3, epochs=100,
            patience=5, service=0.0158467092551291, eject=.1, betas=np.geomspace(.03, 300, 41).tolist(), n_buckets=8, hidden=64)


def extract(model, data, fs, interval, seed, device, budget=None):
    """Arrival sample counts and frozen-encoder logit margins: dict(read_id, is_target, raw_length, m, ell, budget);
    `budget` defaults to the encoder's raw endpoint. Feature metadata preserves its training convention."""
    fs = sampling_rate(fs)
    budget = model.max_samples if budget is None else int(budget)
    if budget > model.max_samples:
        raise ValueError('decision endpoint exceeds the encoder configuration')
    if data.adc.shape[1] < model.max_samples:
        raise ValueError(f'prefix cache must cover the encoder endpoint of {model.max_samples} samples')
    u = np.random.default_rng(seed).random(len(data))
    m = arrivals(u*interval*fs, interval*fs, budget)
    n = np.where(np.isfinite(m), np.minimum(np.floor(m), budget), -1).astype(np.int64)   # forced look: first `budget` samples
    _, ell = encode_looks(model, data, n, device)
    return dict(read_id=data.read_id, is_target=np.asarray(data.is_target, bool), raw_length=data.raw_length, m=m, ell=ell,
                budget=budget, skip=model.skip, stride=model.stride, min_samples=model.min_samples,
                bucket_samples=DEFAULT_BUCKET_SAMPLES,
                n_buckets=bucket_count(budget, model.skip, model.min_samples),
                feature_fs=model.fs, feature_max_samples=budget)


def look_config(z):
    """Feature/bucket contract; cached trajectories without metadata keep the historical feature scales."""
    budget = int(z.get('budget', HEAD['budget']))
    skip, minimum = int(z.get('skip', DEFAULT_SKIP)), int(z.get('min_samples', DEFAULT_MIN_SAMPLES))
    width = int(z.get('bucket_samples', DEFAULT_BUCKET_SAMPLES))
    return dict(budget=budget, skip=skip, min_samples=minimum, stride=int(z.get('stride', DEFAULT_STRIDE)),
                bucket_samples=width, n_buckets=int(z.get('n_buckets', bucket_count(budget, skip, minimum, width))),
                feature_fs=sampling_rate(z.get('feature_fs', DEFAULT_FS)),
                feature_max_samples=int(z.get('feature_max_samples', DEFAULT_MAX_SAMPLES)))


def looks(z, fs):
    cfg = look_config(z)
    n, t, valid, forced, bucket = look_grid(z['m'], fs=fs, max_samples=cfg['budget'], skip=cfg['skip'],
                                          min_samples=cfg['min_samples'], stride=cfg['stride'],
                                          bucket=cfg['bucket_samples'], n_buckets=cfg['n_buckets'])
    ok = np.isfinite(z['ell'])
    valid &= ok
    valid = one_look_per_bucket(valid, np.where(valid, bucket, -1))   # chunks < bucket samples: wait for the next bucket
    bucket = np.where(valid, bucket, -1)
    x = look_inputs(np.nan_to_num(z['ell']), n, valid, None,
                    max_samples=cfg['feature_max_samples'], fs=cfg['feature_fs'])
    return Looks(t, valid, forced, bucket, x, np.asarray(z['raw_length'])/fs, z['is_target'])


def stack(a, b):
    width = max(a.t.shape[1], b.t.shape[1])
    def pad(x, v): return np.pad(x, [(0, 0), (0, width - x.shape[1])] + [(0, 0)]*(x.ndim - 2), constant_values=v)
    return Looks(np.concatenate([pad(a.t, np.nan), pad(b.t, np.nan)]), np.concatenate([pad(a.valid, False), pad(b.valid, False)]),
                 np.concatenate([pad(a.forced, False), pad(b.forced, False)]), np.concatenate([pad(a.bucket, -1), pad(b.bucket, -1)]),
                 np.concatenate([pad(a.x, 0), pad(b.x, 0)]), np.concatenate([a.t_end, b.t_end]), np.concatenate([a.is_target, b.is_target]))


def command_acc(o, is_target):
    y = np.asarray(is_target, bool)
    return float((o['command_reject'][~y].sum() + (~o['command_reject'][y]).sum())/len(y))


def metrics(o, is_target):
    y = np.asarray(is_target, bool)
    return dict(acc=command_acc(o, y), host_command_reject=float(o['command_reject'][~y].mean()), **summarize(o, y))


def train_heads(fit_z, sel_z, fs, output, device='cuda', log=print, **overrides):
    """One C-p controller per beta; returns the selection (beta, SEL metrics) and writes controllers.pt / table.json."""
    cfg = {**HEAD, **overrides}
    features = look_config(fit_z[0])
    if any(look_config(z) != features for z in [*fit_z[1:], sel_z]):
        raise ValueError('head fitting and selection trajectories have different feature configurations')
    cfg['n_buckets'] = features['n_buckets']
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    fit = stack(looks(fit_z[0], fs), looks(fit_z[1], fs)); sel = looks(sel_z, fs)
    ref = float(np.mean(np.asarray(fit_z[0]['raw_length'])[~fit_z[0]['is_target']]/fs))
    states, table = {}, []
    for beta in cfg['betas']:
        t0 = time.time()
        c, _, pol, hist = train_backward(fit, sel, float(beta), ref, cfg['service'], cfg['eject'], device, n_buckets=cfg['n_buckets'],
                                         hidden=cfg['hidden'], lr=cfg['lr'], epochs=cfg['epochs'], patience=cfg['patience'],
                                         seed=cfg['train_seed'])
        states[f'{beta:.6g}'] = {k: v.cpu() for k, v in c.state_dict().items()}
        row = dict(beta=float(beta), epochs=len(hist), seconds=time.time() - t0, **metrics(outcome(sel, pol, cfg['service'], cfg['eject']), sel.is_target))
        table.append(row); log(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}))
    best = select(table, cfg.get('select', 'max_acc'))
    torch.save(dict(states=states, n_buckets=cfg['n_buckets'], hidden=cfg['hidden'], host_reference=ref, betas=cfg['betas'],
                    input_dim=fit.x.shape[-1], fs=sampling_rate(fs), look_config=features), output/'controllers.pt')
    (output/'table.json').write_text(json.dumps(table, indent=2))
    return dict(beta=best['beta'], sel=best, rule=RULES[cfg.get('select', 'max_acc').split(':')[0]], select=cfg.get('select', 'max_acc'))


RULES = {'max_acc': 'maximum SEL command ACC, ties lower SEL target command FRR (default)',
         'frr': 'among betas with SEL target command FRR <= limit, maximum SEL host command reject, ties shorter host pore'}


def select(table, rule='max_acc'):
    """Operating point from the SEL table: 'max_acc' (default) or 'frr:<limit>' (e.g. frr:0.01)."""
    if rule == 'max_acc': return max(table, key=lambda r: (r['acc'], -r['target_command_frr']))
    if rule.startswith('frr:'):
        limit = float(rule[4:]); ok = [r for r in table if r['target_command_frr'] <= limit]
        if not ok: raise ValueError(f'no beta meets SEL target FRR <= {limit}')
        return max(ok, key=lambda r: (r['host_command_reject'], -r['host_mean_pore_seconds']))
    raise ValueError(f'unknown selection rule {rule}')


def load_controller(path, beta, device='cpu'):
    saved = torch.load(path, map_location=device)
    state = saved['states'][f'{beta:.6g}']
    c = BucketController(state['mean'].shape[0], saved['n_buckets'], saved['hidden'], beta)
    c.load_state_dict(state)
    c.look_config = saved.get('look_config')
    return c.to(device).eval()


def evaluate(controller, z, fs, device='cpu', service=HEAD['service'], eject=HEAD['eject']):
    stored = getattr(controller, 'look_config', None)
    if stored is not None:
        if any(k in z and z[k] != value for k, value in stored.items()):
            raise ValueError('trajectory feature configuration differs from the trained controller')
        z = {**stored, **z}
    if z.get('n_buckets', controller.n_buckets) != controller.n_buckets:
        raise ValueError('trajectory bucket configuration differs from the trained controller')
    lk = looks(z, fs)
    return outcome(lk, policy_actions(controller, lk, device), service, eject)
