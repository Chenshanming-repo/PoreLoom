#!/usr/bin/env python3
"""Trained single-window baselines (3000-sample windows) as sliding-window scorers + C-p.

Per unit, the main-run baseline models are reused unchanged (no retraining): runs/main/<ds>/<m>_r<r>/fit/best.pt for
m = readcurrent, squigglenet, nanodeep (project recipe), nanodeep_upstream (authors' recipe). Each keeps its own native
input contract (poreloom.baselines.preprocessing: NanoDeep ADC (x - mean)/var, ReadCurrent ADC MAD + outlier fill,
SquiggleNet calibrated pA + official normalization).
  looks  at every 0.4 s chunk arrival n (head.HEAD phase seeds), ell = logit(target) - logit(host) of the latest 3000
         samples raw[e-3000:e], e = min(n, 7500), when the window starts at or after the tool's native start (NanoDeep
         raw 1000, others raw 1500) and lies inside the read; so the first look is the tool's own native window
         (decision possible from raw 4000 / 4500) and later looks slide it up to raw[4500:7500] (forced look).
  head   default C-p (head.train_heads, DEV-A fit on two phase copies, beta by max DEV-B ACC, locked before TEST);
         768-sample buckets from the first scored stream length (raw - 1500 = 2500 NanoDeep / 3000 others: 5 / 4
         buckets + forced).
  TEST   0.4 s, nominal service 0.0158 s, eject 0.1 s.
Report (window_tools.py report): the same model's native single-window decision (max-ACC, measured service), streaming
PoreLoom and the 1000-window tools.

    native_cp.py run --data <data>/<ds> --main <runs>/main/<ds> --fs F --rep r --out <runs>/native_cp/<ds>/r<r> --device cuda:0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import window_tools as WT  # noqa: E402  (sets sys.path for poreloom and the ablation helpers)
from window_tools import AB, D, H, P  # noqa: E402

MODELS = dict(readcurrent='readcurrent', squigglenet='squigglenet', nanodeep='nanodeep', nanodeep_upstream='nanodeep')
WINDOW, SKIP, BUDGET, BUCKET = 3000, 1500, 7500, 768


@torch.no_grad()
def native_extract(model, tool, data, fs, interval, seed, device, batch=1024):
    from poreloom.baselines.preprocessing import CONTRACTS, prepare_windows
    from poreloom.chunk_replay import arrivals
    from poreloom.method import native
    c = CONTRACTS[tool]; first_lo = c['raw_start']
    u = np.random.default_rng(seed).random(len(data))
    m = arrivals(u*interval*fs, interval*fs, BUDGET)
    n = np.where(np.isfinite(m), np.floor(m), -1).astype(np.int64)
    ell = np.full(n.shape, np.nan, np.float32)
    end = np.minimum(n, BUDGET); lo = end - WINDOW
    ok = (n >= 0) & (lo >= first_lo) & (end <= np.asarray(data.raw_length)[:, None]) & (end <= data.adc.shape[1])
    rows, cols = np.nonzero(ok)
    for s in range(0, len(rows), batch):
        r, k = rows[s:s + batch], cols[s:s + batch]
        ur, inv = np.unique(r, return_inverse=True)
        sig = data.pa(ur, 0, BUDGET) if c['units'] == 'pA' else np.asarray(data.adc[ur, :BUDGET])
        x = sig[inv[:, None], lo[r, k][:, None] + np.arange(WINDOW)]
        x, valid = prepare_windows(x.astype(np.float32 if c['units'] == 'pA' else np.int16), tool)
        if x.ndim == 2: x = x[:, None]
        if not valid.any(): continue
        logits = native._forward(model, x[valid], tool, device).float()
        ell[r[valid], k[valid]] = (logits[:, c['microbe_index']] - logits[:, c['host_index']]).cpu().numpy()
    min_samples = first_lo + WINDOW - SKIP
    return dict(read_id=data.read_id, is_target=np.asarray(data.is_target, bool), raw_length=data.raw_length, m=m, ell=ell,
                budget=BUDGET, min_samples=min_samples, n_buckets=int(np.ceil((BUDGET - SKIP - min_samples)/BUCKET)))


def run(a):
    from poreloom.method import native
    P._setup(a.device)
    AB.write(a.out/'run.json', dict(data=str(a.data), main=str(a.main), rep=a.rep, fs=a.fs, started=time.strftime('%Y-%m-%d %H:%M:%S')))
    dev_a, dev_b = D.dev_halves(a.data); test = D.load_split(a.data, 'test', verify=True)
    ph, cp = H.HEAD['phase_seed'], H.HEAD['copy_seed']
    for name, tool in MODELS.items():
        out = a.out/name
        if (out/'metrics.json').exists(): continue
        src = a.main/f'{name}_r{a.rep}'
        info = json.loads((src/'run.json').read_text()); assert info['tool'] == tool and info['replicate'] == a.rep, src
        model = native._load(tool, src/'fit'/'best.pt', a.device)
        z = lambda data, tag, seed: AB.cached(out/'cache'/f'native_{tag}_0.4_{seed}.npz',
                                              lambda: native_extract(model, tool, data, a.fs, .4, seed, a.device))
        fit = [z(dev_a, 'deva', s) for s in (ph, cp)]; sel = z(dev_b, 'devb', ph); zt = z(test, 'test', ph)
        AB.cp_train(fit, sel, a.fs, out/'head', a.device, a.rep)
        rows, locked = AB.cp_test(out/'head', zt, a.fs, a.device, out/'test_outcomes_0.4.npz')
        AB.write(out/'metrics.json', dict(model=name, source=str(src), locked=locked, test_curve=rows))
        AB.log(name, 'TEST', {k: round(v, 4) for k, v in locked.items()})
    (a.out/'JOB_DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('run',))
    ap.add_argument('--data', type=Path, required=True); ap.add_argument('--main', type=Path, required=True)
    ap.add_argument('--fs', type=float, required=True); ap.add_argument('--rep', type=int, default=0)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--device', default='cuda:0')
    run(ap.parse_args())
