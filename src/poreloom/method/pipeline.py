"""PoreLoom (PoreLoom-SH-ACC C-p): complete training and offline evaluation on one dataset, from random initialization.

    python -m poreloom.method all --data <dataset root> --out <run dir> --fs 5000 --device cuda:0

Stages (each skips when its output exists; run them in order or use `all`):
  stageA   window encoder (CNN + projection) from a seeded random init on 1500-sample windows   -> stageA/stage_A.pt
  stream   StreamR0 (streaming encoder, causal normalizer, time term) from stage A, S1 recipe  -> stream/best.pt
  head     C-p action heads with beta=1.0 by default, fitted on training reads and early-stopped on validation -> head/selection.json
  test     replay on TEST at --interval with the measured encoder-update delay
                                                                                              -> test/metrics.json
The final deployable bundle is <out>/poreloom_sh_cp.pt (encoder checkpoint path, head state, beta, fs).
--replicate r (default 0) adds r to the stage A, S1 and head training seeds (independent training replicates);
The manuscript protocol fixes beta=1.0; it does not select an operating point.
--interval s (default 0.4) is the chunk interval in seconds the head is trained on and the primary TEST interval.
--fs Hz (default 4000) sets the data sampling rate for replay. Manuscript length-feature scales stay at 4000/7500.
--max-samples N (default 7500) or --max-decision-seconds T sets the raw decision endpoint, including prefix skipping.
Dataset layout: poreloom.method.data. No weights trained on other data are used at any stage.
"""
import argparse
from decimal import Decimal
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import head as H
from .pretrain import STAGEA, train_window_encoder
from ..stream_config import DEFAULT_FS, DEFAULT_MAX_SAMPLES, bucket_count, sampling_rate

STAGES = ('stageA', 'stream', 'head', 'test')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def _setup(device):
    torch.set_num_threads(4)
    if str(device).startswith('cuda'): torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False


def _encoder(out, device, *, fs=None, max_samples=None):
    from ..stream_training import load_stream_r0
    model = load_stream_r0(Path(out)/'stream'/'best.pt', device)[0].eval()
    if (fs is not None and model.fs != fs) or (max_samples is not None and model.max_samples != max_samples):
        raise ValueError('encoder configuration differs from the run sampling features or decision endpoint')
    return model


def stageA(a):
    if (a.out/'stageA'/'stageA_history.json').exists(): return
    train = D.load_split(a.data, 'train'); dev_a = D.validation(a.data)
    train_window_encoder(train, dev_a, a.out/'stageA', a.device, log=log, seed=STAGEA['seed'] + a.replicate)


def stream(a):
    from ..stream_training import train_stream_r0
    if (a.out/'stream'/'done.json').exists(): return
    train = D.load_split(a.data, 'train'); dev_a = D.validation(a.data)
    from ..stream_training import R0_DEFAULTS
    train_stream_r0(train, dev_a, a.out/'stageA'/'stage_A.pt', a.out/'stream', device=a.device, log=log,
                    seed=R0_DEFAULTS['seed'] + a.replicate, fs=a.feature_fs, max_samples=a.max_samples,
                    feature_max_samples=a.feature_max_samples)
    best = torch.load(a.out/'stream'/'best.pt', map_location='cpu')
    (a.out/'stream'/'done.json').write_text(json.dumps(dict(best_epoch=int(best['epoch'])), indent=2))


def protocol_for(t_norm):
    """Protocol tag of a run: the 4 November 2026 manuscript used each training set's own T_norm (t_norm None)."""
    return 'manuscript_20261104' if t_norm is None else 'fixed_tnorm_20261106'


def head(a):
    if (a.out/'head'/'selection.json').exists(): return
    t_norm = getattr(a, 't_norm', H.HEAD['t_norm'])
    if getattr(a, 'protocol', protocol_for(t_norm)) != protocol_for(t_norm):
        raise ValueError('use a new output directory to fit heads with the current protocol')
    train, validation = D.load_split(a.data, 'train'), D.validation(a.data)
    if np.intersect1d(train.read_id, validation.read_id).size:
        raise ValueError('training and validation read IDs overlap')
    net = _encoder(a.out, a.device, fs=a.feature_fs, max_samples=a.max_samples)
    fit = [H.extract(net, train, a.fs, a.interval, s, a.device) for s in (H.HEAD['phase_seed'], H.HEAD['copy_seed'])]
    sel = H.extract(net, validation, a.fs, a.interval, H.HEAD['phase_seed'], a.device)
    choice = H.train_heads(fit, sel, a.fs, a.out/'head', a.device, log=log, train_seed=H.HEAD['train_seed'] + a.replicate,
                           betas=[1.0], t_norm=t_norm)
    choice['protocol'] = protocol_for(t_norm)
    choice['fit_partition'] = 'train'
    choice['validation_partition'] = 'dev_hash_parity_0'
    choice['written'] = time.strftime('%Y-%m-%d %H:%M:%S') + ', before any TEST input'
    (a.out/'head'/'selection.json').write_text(json.dumps(choice, indent=2))
    ctrl = H.load_controller(a.out/'head'/'controllers.pt', choice['beta'])
    torch.save(dict(method='PoreLoom-SH-ACC C-p', encoder=str((a.out/'stream'/'best.pt').resolve()), head_state=ctrl.state_dict(),
                    head=dict(n_buckets=ctrl.n_buckets, hidden=H.HEAD['hidden'], input_dim=int(ctrl.state_dict()['mean'].shape[0]),
                              look_config=ctrl.look_config),
                    beta=choice['beta'], fs=a.fs, max_samples=net.max_samples, max_decision_seconds=net.max_samples/a.fs,
                    service_seconds=H.HEAD['service'], interval_trained=a.interval, t_norm=choice['t_norm']),
               a.out/'poreloom_sh_cp.pt')
    log('fixed beta', choice['beta'], 'validation ACC', round(choice['sel']['acc'], 4))


def test(a):
    if (a.out/'test'/'metrics.json').exists(): return
    choice = json.loads((a.out/'head'/'selection.json').read_text())
    data = D.load_split(a.data, 'test', verify=True)
    with np.load(Path(a.data)/'test'/'metadata.npz', allow_pickle=False) as z:
        repeat = z['repeat'] if 'repeat' in z.files else np.ones(len(data), int)
    net = _encoder(a.out, a.device, fs=a.feature_fs, max_samples=a.max_samples)
    ctrl = H.load_controller(a.out/'head'/'controllers.pt', choice['beta'], a.device)
    (a.out/'test').mkdir(parents=True, exist_ok=True)
    rows = []
    z = H.extract(net, data, a.fs, a.interval, H.HEAD['phase_seed'], a.device)
    o = H.evaluate(ctrl, z, a.fs, a.device, service=H.HEAD['service'])
    np.savez(a.out/'test'/f'outcomes_{a.interval}_nominal.npz', read_id=z['read_id'], is_target=z['is_target'], repeat=repeat,
             **{k: v for k, v in o.items() if k != 'missing'})
    groups = {'pooled': np.ones(len(repeat), bool), **{f'repeat_{r}': repeat == r for r in np.unique(repeat)}}
    for g, mask in groups.items():
        rows.append(dict(interval=a.interval, service='nominal', group=g,
                         **H.metrics({k: v[mask] for k, v in o.items() if k != 'missing'}, z['is_target'][mask])))
    (a.out/'test'/'metrics.json').write_text(json.dumps(dict(beta=choice['beta'], fs=a.fs, max_samples=a.max_samples,
                                                          max_decision_seconds=a.max_samples/a.fs, rows=rows), indent=2))
    for r in rows:
        if r['interval'] == a.interval and r['service'] == 'nominal':
            log(r['group'], {k: round(r[k], 4) for k in ('acc', 'host_command_reject', 'target_command_frr', 'host_mean_pore_seconds')})


def t_norm_arg(text):
    """--t-norm: seconds (finite, > 0), or 'train' for the mean natural duration of the training host reads."""
    if text == 'train': return None
    value = float(text)
    if not math.isfinite(value) or value <= 0: raise argparse.ArgumentTypeError('--t-norm must be positive seconds or train')
    return value


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m poreloom.method', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=STAGES + ('all',))
    ap.add_argument('--data', type=Path, required=True, help='dataset root with train/, dev/, test/ prefix caches')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--fs', type=float, default=DEFAULT_FS,
                    help='sampling rate in Hz for replay timing; feature scales stay fixed (default: %(default)s)')
    budget = ap.add_mutually_exclusive_group()
    budget.add_argument('--max-samples', type=int,
                        help=f'raw decision endpoint in sample points, including skipped prefix (default: {DEFAULT_MAX_SAMPLES})')
    budget.add_argument('--max-decision-seconds', type=float,
                        help='raw observation-time budget; endpoint = floor(seconds * fs), including skipped prefix')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--replicate', type=int, default=0, help='training replicate r: seeds of stage A, S1 and head + r')
    ap.add_argument('--select', default='max_acc', choices=('max_acc',), help=argparse.SUPPRESS)
    ap.add_argument('--interval', type=float, default=H.HEAD['interval'],
                    help='chunk interval in seconds for head training and the primary TEST replay (default 0.4)')
    ap.add_argument('--t-norm', type=t_norm_arg, default=H.HEAD['t_norm'],
                    help='T_norm in seconds normalizing host pore time in the FACP labels, or train for the mean natural '
                         'training host duration (default: %(default)s s, one value for all datasets)')
    a = ap.parse_args(argv)
    try:
        a.fs = sampling_rate(a.fs)
        if a.max_decision_seconds is not None:
            if not math.isfinite(a.max_decision_seconds) or a.max_decision_seconds <= 0:
                raise ValueError('--max-decision-seconds must be finite and positive')
            a.max_samples = int(Decimal(str(a.max_decision_seconds))*Decimal(str(a.fs)))
        if a.max_samples is None: a.max_samples = DEFAULT_MAX_SAMPLES
        bucket_count(a.max_samples)
        if not math.isfinite(a.interval) or a.interval <= 0:
            raise ValueError('--interval must be finite and positive')
    except ValueError as error:
        ap.error(str(error))
    a.max_decision_seconds = a.max_samples/a.fs
    a.feature_fs = DEFAULT_FS
    a.feature_max_samples = DEFAULT_MAX_SAMPLES
    a.protocol = protocol_for(a.t_norm)
    a.out.mkdir(parents=True, exist_ok=True); _setup(a.device)
    run = dict(data=str(a.data.resolve()), fs=a.fs, max_samples=a.max_samples, max_decision_seconds=a.max_decision_seconds,
               feature_fs=a.feature_fs, feature_max_samples=a.feature_max_samples, protocol=a.protocol, t_norm=a.t_norm,
               replicate=a.replicate, select=a.select, interval=a.interval,
               started=time.strftime('%Y-%m-%d %H:%M:%S'))
    if not (a.out/'run.json').exists(): (a.out/'run.json').write_text(json.dumps(run, indent=2))
    else:
        old = json.loads((a.out/'run.json').read_text())
        old.setdefault('interval', H.HEAD['interval'])   # runs from before --interval used 0.4 s
        old.setdefault('max_samples', DEFAULT_MAX_SAMPLES)
        if any(old.get(k) != run[k] for k in ('data', 'fs', 'max_samples', 'replicate', 'select', 'interval')):
            raise ValueError('run directory belongs to another dataset, sampling rate, decision budget, replicate, selection rule or chunk interval')
        if old.get('t_norm') != a.t_norm:   # runs from before --t-norm used the per-dataset mean (None)
            raise ValueError(f"run directory was started with t_norm={old.get('t_norm')} (None: training-set mean); "
                             f"resume it with the same --t-norm (train for None) or use a new directory")
        # Historical runs used 4000 in model features even when replay used a different sampling rate.
        a.feature_fs = sampling_rate(old.get('feature_fs', DEFAULT_FS))
        a.feature_max_samples = old.get('feature_max_samples', old['max_samples'])
        a.protocol = old.get('protocol', 'legacy')
    for st in (STAGES if a.stage == 'all' else (a.stage,)):
        log('stage', st); globals()[st](a)
