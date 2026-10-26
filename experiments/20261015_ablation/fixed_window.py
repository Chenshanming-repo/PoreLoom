#!/usr/bin/env python3
"""Fixed-prefix accuracy of the frozen streaming encoder (main Fig. 2a).

At each usable prefix length, the decision threshold maximizes DEV-B accuracy and
is locked before TEST. Short reads use the available signal; reads shorter than
1500 skipped plus 432 usable samples are retained.

    fixed_window.py run --main <runs>/main/<ds> --data <data>/<ds> --rep r --out <runs>/fixed_window/<ds>/r<r>
    fixed_window.py report --root <runs>/fixed_window [--md summary_zh.md]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO/'src'))
from poreloom.method import data as D, pipeline as P  # noqa: E402
from poreloom.stream_controller import encode_looks  # noqa: E402

SKIP, BUDGET, MIN_USED = 1500, 7500, 432
GRID = np.unique(np.r_[np.arange(MIN_USED, 6001, 192), 768, 1200, 3000, 6000])
FAMILY = (('r10', 'R10'), ('r9', 'R9'), ('sim', 'Sim'), ('sputum', 'Sputum'))


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp'); tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.rename(path)


def margins(net, data, device):
    """[N, G] logit margins (target - host) at stream length GRID, nan where the read is shorter than 1500 + 432."""
    length = np.asarray(data.raw_length, np.int64)
    n = np.minimum(SKIP + GRID[None], length[:, None])
    n = np.where(n >= SKIP + MIN_USED, n, -1)
    return encode_looks(net, data, n, device)[1].astype(np.float64)


def best_threshold(ell, y):
    """Max-ACC threshold t (reject = host call when ell < t; nan = keep) on one grid column; ties -> lower t."""
    ok = np.isfinite(ell)
    cand = np.r_[-np.inf, np.unique(ell[ok]), np.inf]
    e, t = ell[ok], y[ok]
    order = np.argsort(e); e, t = e[order], t[order]
    # rejects below cand[k]: host rejected correctly, targets rejected wrongly; non-scored reads are kept
    host_below = np.searchsorted(e, cand, 'left')
    hcum = np.r_[0, np.cumsum(~t)]; tcum = np.r_[0, np.cumsum(t)]
    correct = hcum[host_below] + (y.sum() - tcum[host_below])
    k = int(np.argmax(correct))
    return float(cand[k]), float(correct[k]/len(y))


def correct_matrix(ell, thr, y):
    reject = np.where(np.isfinite(ell), ell < thr[None], False)
    return np.where(y[:, None], ~reject, reject)


def run(a):
    P._setup(a.device)
    base = a.main/f'poreloom_r{a.rep}'
    if (a.out/'metrics.json').exists(): log('done', a.out); return
    fs = float(json.loads((base/'run.json').read_text())['fs'])
    _, dev_b = D.dev_halves(a.data)
    test = D.load_split(a.data, 'test', verify=True)
    if a.limit:
        dev_b, test = (D.subset(x, np.unique(np.linspace(0, len(x) - 1, a.limit).astype(int))) for x in (dev_b, test))
    net = P._encoder(base, a.device)
    yb, yt = np.asarray(dev_b.is_target, bool), np.asarray(test.is_target, bool)
    eb = margins(net, dev_b, a.device)
    thr, dev_acc = map(np.array, zip(*(best_threshold(eb[:, g], yb) for g in range(len(GRID)))))
    write(a.out/'thresholds.json', dict(grid=GRID.tolist(), threshold=thr.tolist(), dev_b_acc=dev_acc.tolist(),
                                        written=time.strftime('%Y-%m-%d %H:%M:%S') + ', before any TEST input'))
    log('DEV-B thresholds locked; DEV-B ACC', dict(zip(GRID[[0, 4, -1]].tolist(), np.round(dev_acc[[0, 4, -1]], 4).tolist())))
    et = margins(net, test, a.device)
    C = correct_matrix(et, thr, yt)
    np.savez(a.out/'test_margins.npz', read_id=np.asarray(test.read_id).astype(str), is_target=yt,
             raw_length=np.asarray(test.raw_length), grid=GRID, ell=et, correct=C)
    acc = C.mean(0)
    res = dict(main=str(base), data=str(a.data), rep=a.rep, fs=fs, n_test=int(len(yt)), grid=GRID.tolist(),
               acc=acc.tolist(), dev_b_acc=dev_acc.tolist(), limit=a.limit)
    write(a.out/'metrics.json', res)
    log('TEST ACC', {int(GRID[g]): round(acc[g], 4) for g in (0, 4, len(GRID) - 1)})


def report(a):
    units = sorted(a.root.glob('*/r*/metrics.json'))
    keep = ('main', 'data', 'rep', 'fs', 'n_test', 'grid', 'acc', 'dev_b_acc', 'limit')
    per = {f'{p.parent.parent.name}/{p.parent.name}':
           {k: v for k, v in json.loads(p.read_text()).items() if k in keep} for p in units}
    family = lambda name: next(label for prefix, label in FAMILY if name.startswith(prefix))
    show = [432, 768, 1200, 1968, 3000, 4464, 6000]
    lines = ['# Fixed-prefix accuracy', '',
             'Three-unit dataset means; each length has its own DEV-B-selected threshold.', '',
             '| Dataset | ' + ' | '.join(map(str, show)) + ' |', '|---|' + '---:|'*len(show)]
    for _, label in FAMILY:
        rows = [r for unit, r in per.items() if family(unit) == label]
        if rows:
            values = [100*np.mean([r['acc'][r['grid'].index(length)] for r in rows]) for length in show]
            lines.append(f'| {label} | ' + ' | '.join(f'{v:.2f}' for v in values) + ' |')
    text = '\n'.join(lines) + '\n'
    print(text)
    write(a.root/'report.json', per)
    if a.md:
        a.md.write_text(text)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('run', 'report'))
    ap.add_argument('--main', type=Path); ap.add_argument('--data', type=Path); ap.add_argument('--out', type=Path)
    ap.add_argument('--root', type=Path); ap.add_argument('--md', type=Path)
    ap.add_argument('--rep', type=int, default=0); ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--limit', type=int, default=0, help='smoke test: rows per split')
    a = ap.parse_args()
    globals()[a.stage](a)
