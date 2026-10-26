#!/usr/bin/env python3
"""Offline consistency of the live Icarust strategies (icarust_live.py) with their offline TEST definitions.

Sim g1 TEST reads are fed to each live policy as 0.4 s chunks with the offline per-read phase (seed 20261012), exactly
as poreloom.method / method.baselines replay them; the live command (reject / accept / none) must equal the offline
command for every read where a zero-service offline replay issues one. This explicit validation computes its
reference from the trained policy or saved baseline scores; it does not require zero-service experiment outputs.

    icarust_check.py [--n 3000] [--device cuda:0]   -> <results>/final_20261014/icarust/check.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import importlib.util
OUT = Path('/path/to/poreloom_work/results/final_20261014/icarust')
DATA = Path('/path/to/poreloom_work/results/sim_datasets_20261014/g1')


def replay(policy, data, rows, interval=.4):
    from poreloom.chunk_replay import arrivals
    u = np.random.default_rng(20261012).random(len(data.read_id))[rows]
    m = arrivals(u*interval*live.FS, interval*live.FS, 13500)
    states = [dict(channel=i, read_id=str(data.read_id[r]), read_number=1, received=0, k=0, h=None,
                   buf=np.asarray(data.adc[r]).astype(np.int16).copy()) for i, r in enumerate(rows)]
    raw = np.asarray(data.raw_length)[rows]
    decided = np.full(len(rows), None, object)
    for j in range(m.shape[1]):
        jobs = []
        for i, s in enumerate(states):
            if decided[i] is not None or not np.isfinite(m[i, j]): continue
            s['received'] = int(min(np.floor(m[i, j]), raw[i], len(s['buf'])))
            need = policy.needed(s)
            while need is not None and decided[i] is None and s['received'] >= need:
                out = policy.decide([s])[0]
                if out[1] != 'defer': decided[i] = out[1]
                need = policy.needed(s) if decided[i] is None else None
    return decided


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=3000); ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    global live
    spec = importlib.util.spec_from_file_location('final_icarust_live', HERE/'icarust_live.py')
    live = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(live)
    import torch
    from poreloom.method import data as D, head as H, pipeline as P, native_policy as offline_policy
    torch.cuda.set_device(a.device)
    data = D.load_split(DATA, 'test')
    rows = np.sort(np.random.default_rng(7).choice(len(data.read_id), a.n, replace=False))
    result = {}
    for name in ('PoreLoom', *live.TOOLS):
        run = Path(live.MODELS[name])
        if name == 'PoreLoom':
            selection = json.loads((run/'head/selection.json').read_text())
            controller = H.load_controller(run/'head/controllers.pt', selection['beta'], a.device)
            trajectory = H.extract(P._encoder(run, a.device), data, live.FS, .4, H.HEAD['phase_seed'], a.device)
            off = H.evaluate(controller, trajectory, live.FS, a.device, service=0.)
        else:
            with np.load(run/'test/scores.npz', allow_pickle=False) as saved:
                scores = {key: saved[key] for key in saved.files}
            selection = json.loads((run/'selection.json').read_text())
            offline_policy.FS = live.FS
            meta = dict(read_id=data.read_id, is_target=np.asarray(data.is_target, bool), raw_length=data.raw_length)
            off = offline_policy.native_policy(scores, meta, threshold=selection['rules']['max_acc']['threshold'],
                                       service=0., delta=.4, raw_end=4000 if name == 'nanodeep' else 4500)
        policy = live.build(name, a.device, live.cl.TRAIN_CAL)
        got = replay(policy, data, rows)
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


if __name__ == '__main__':
    main()
