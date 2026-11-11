"""Live strategies against their offline definitions, before any closed-loop run.

A random sample of test reads of the dataset the models were trained on is fed to each live strategy as growing
0.4-s chunks with the offline per-read phases (seed 20261012, drawn over the whole test set). As in a live run, each
read's ADC is first mapped with its own calibration to the strategy's ADC grid (the client does the same with the
simulator's calibration). The live command (reject, accept or none) is compared with the command of the offline
replay of the same reads and phases with zero service time (PoreLoom: poreloom.method.head; baselines: native scores
and the native 0.5 decision), for every read where the offline replay issues one.
"""
from pathlib import Path

import numpy as np

INTERVAL, PHASE_SEED, ROW_SEED, BUFFER = .4, 20261012, 7, 13500


def offline_poreloom(bundle_path, data, phase, fs, device):
    import torch
    from ..method import head as H
    from ..minknow.policy import resolve_encoder
    from ..stream_controller import BucketController
    from ..stream_training import load_stream_r0
    bundle = torch.load(bundle_path, map_location='cpu')
    h = bundle['head']
    controller = BucketController(h['input_dim'], h['n_buckets'], h['hidden'], bundle['beta'])
    controller.load_state_dict(bundle['head_state'])
    controller.look_config = h.get('look_config')
    controller = controller.to(device).eval()
    model = load_stream_r0(resolve_encoder(bundle, bundle_path), device)[0].eval()
    z = H.extract(model, data, fs, INTERVAL, None, device, phase=phase)
    return H.evaluate(controller, z, fs, device, service=0.)


def offline_native(name, path, data, phase, fs, device):
    from ..baselines.preprocessing import CONTRACTS
    from ..method import native, native_policy
    from .strategies import checkpoint, parse, set_window
    tool, window = parse(name)
    set_window(window)
    scores = native.score(tool, checkpoint(path), native.prepare(tool, data), device)
    native_policy.FS = fs
    meta = dict(read_id=data.read_id, is_target=np.asarray(data.is_target, bool), raw_length=data.raw_length)
    return native_policy.native_policy(scores, meta, threshold=None, service=0., delta=INTERVAL,
                                       raw_end=CONTRACTS[tool]['raw_end'], phase=phase)


def model_grid(adc, calibration, model):
    """One read's int16 ADC on the strategy's ADC grid (calibration: offset, range, digitisation)."""
    from .client import adc_map
    gain, shift = adc_map(dict(offset=calibration[0], range=calibration[1], digitisation=calibration[2]), model)
    return np.rint(np.asarray(adc, np.float64)*gain + shift).astype(np.int16)


def live_commands(policy, data, phase, fs, model_calibration):
    """Commands of a live strategy on every read of `data`, one read and one decision at a time."""
    from ..chunk_replay import arrivals
    m = arrivals(phase*INTERVAL*fs, INTERVAL*fs, BUFFER)
    raw = np.asarray(data.raw_length)
    cal = np.asarray(data.calibration, np.float64)
    states = [dict(channel=i, read_id=str(data.read_id[i]), read_number=1, received=0, k=0, h=None,
                   buf=model_grid(data.adc[i], cal[i], model_calibration)) for i in range(len(raw))]
    got = np.full(len(raw), None, object)
    for j in range(m.shape[1]):
        for i, s in enumerate(states):
            if got[i] is not None or not np.isfinite(m[i, j]): continue
            s['received'] = int(min(np.floor(m[i, j]), raw[i], len(s['buf'])))
            need = policy.needed(s)
            while need is not None and got[i] is None and s['received'] >= need:
                action = policy.decide([s])[0][1]
                if action != 'defer': got[i] = action
                need = policy.needed(s) if got[i] is None else None
    return got


def check(models, data_root, strategies, device, n=3000, fs=4000., workdir='.', log=None):
    """Returns {strategy: agreement counts}; log(name, counts) after each strategy. Baselines with 6000-sample windows
    are checked after the others."""
    import torch
    from ..method import data as D
    from .strategies import MODEL_CALIBRATION, build, poreloom_bundle
    full = D.load_split(Path(data_root), 'test')
    rows = np.sort(np.random.default_rng(ROW_SEED).choice(len(full.read_id), min(n, len(full.read_id)), replace=False))
    phase = np.random.default_rng(PHASE_SEED).random(len(full.read_id))[rows]
    data = D.subset(full, rows)
    result = {}
    for name in sorted(strategies, key=lambda s: s.endswith('_w6000')):
        if name == 'No-AS': continue
        if name == 'PoreLoom':
            off = offline_poreloom(poreloom_bundle(models[name], workdir), data, phase, fs, device)
        else:
            off = offline_native(name, models[name], data, phase, fs, device)
        policy = build(name, models[name], device, MODEL_CALIBRATION, workdir)
        got = live_commands(policy, data, phase, fs, MODEL_CALIBRATION)
        want = np.where(off['command'], np.where(off['command_reject'], 'reject', 'accept'), None)
        compared = [(g, w) for g, w in zip(got, want) if w is not None]
        agree = sum(g == w for g, w in compared)
        result[name] = dict(reads=len(rows), compared=len(compared), agree=int(agree), agreement=agree/max(1, len(compared)),
                            live_reject=int(sum(g == 'reject' for g in got)), offline_reject=int(off['command_reject'].sum()),
                            live_none=int(sum(g is None for g in got)), offline_none=int(sum(w is None for w in want)),
                            pins=policy.pins)
        if log is not None: log(name, result[name])
        del policy
        if str(device).startswith('cuda'): torch.cuda.empty_cache()
    return result
