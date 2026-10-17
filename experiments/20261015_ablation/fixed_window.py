#!/usr/bin/env python3
"""Why one fixed signal window is the wrong design.

The frozen S1 encoder of the main run (trained on every prefix length 432-6000 of the stream) scores every DEV-B and TEST
read at a common grid of stream lengths u (samples after the 1500-sample skip; prefix n = min(1500 + u, read length),
no decision when the read ends before 1500 + 432). At every u a fixed-window rule "reject if margin < t_u" has t_u
locked at maximum DEV-B ACC before TEST is read. On TEST:
  A  more signal -> higher accuracy: ACC(u) with a bootstrap CI over reads (2000 resamples)
  B  reads differ in how much signal they need: per read, u* = the shortest grid length from which the fixed-window
     decision stays correct up to 6000 (reads >= 7500 samples, so every grid point is real signal); the share of
     reads with u* <= 432 / 1200 / 3000, > 3000, and never correct. At the 3000-sample window (NanoDeep / ReadCurrent
     size): "wasted" = correct and stable already by 1200, "too short" = wrong at 3000 but stable correct later
  C  the early-decidable reads can be recognized online: TEST reads ranked by |margin - t_u| at u = 432 / 1200, ACC
     of the most confident 25 / 50 % at u and at 6000, against the rest
  D  per-read stopping vs one window: the oracle (each read stops at u*) and the realized PoreLoom C-p point against
     the best fixed window, from runs/ablation/<unit>/{main,fixed_len}/metrics.json (chunk replay, host pore time)

    fixed_window.py run    --main <runs>/main/<ds> --data <data>/<ds> --rep r --out <runs>/fixed_window/<ds>/r<r>
    fixed_window.py report --root <runs>/fixed_window --ablation <runs>/ablation [--md summary_zh.md]
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
EARLY, WINDOW = (432, 1200), 3000
FAMILY = (('r10', 'R10'), ('r9', 'R9'), ('sim', 'Sim'), ('sputum', 'Sputum'))
BOOT, SEED = 2000, 20261015


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
    rng = np.random.default_rng(SEED + a.rep)
    idx = rng.integers(0, len(yt), (BOOT, len(yt)))
    acc = C.mean(0); ci = np.percentile(C[idx].mean(1), [2.5, 97.5], axis=0)

    # B: u* on reads with the full 7500 samples
    full = np.asarray(test.raw_length) >= BUDGET
    Cf = C[full]
    stable = np.flip(np.cumprod(np.flip(Cf, 1), 1), 1).astype(bool)          # stable[:, g]: correct at every g' >= g
    first = np.where(stable.any(1), np.argmax(stable, 1), -1)
    ustar = np.where(first >= 0, GRID[np.maximum(first, 0)], -1)
    g3 = int(np.flatnonzero(GRID == WINDOW)[0])
    share = lambda m: float(m.mean())
    B = dict(n_full=int(full.sum()), share_full=share(full),
             le=dict((int(u), share((ustar >= 0) & (ustar <= u))) for u in (432, 768, 1200, 1968, 3000, 4464, 6000)),
             gt_3000=share(ustar > WINDOW), never=share(ustar < 0),
             by_class={c: dict(le_1200=share((ustar[m] >= 0) & (ustar[m] <= 1200)), gt_3000=share(ustar[m] > WINDOW),
                               never=share(ustar[m] < 0)) for c, m in (('target', yt[full]), ('host', ~yt[full]))},
             at_3000=dict(wasted=share((ustar >= 0) & (ustar <= 1200)), too_short=share(~Cf[:, g3] & (ustar > WINDOW)),
                          wrong=share(~Cf[:, g3]), flips=share((np.diff(Cf.astype(int), axis=1) != 0).any(1))),
             oracle=dict(acc=share(ustar >= 0), mean_used=float(np.mean(np.where(ustar >= 0, ustar, GRID[-1])))),
             hist=dict(zip([int(u) for u in GRID], [share(ustar == u) for u in GRID])))
    # the shortest fixed window reaching the oracle ACC on the same reads
    reach = [int(GRID[g]) for g in range(len(GRID)) if Cf[:, g].mean() >= B['oracle']['acc'] - 1e-12]
    B['oracle']['fixed_window_same_acc'] = reach[0] if reach else None
    B['fixed_acc_full'] = dict(zip([int(u) for u in GRID], Cf.mean(0).tolist()))

    # C: confidence at an early window predicts correctness
    Cs = {}
    for u in EARLY:
        g = int(np.flatnonzero(GRID == u)[0])
        ok = np.isfinite(et[:, g])
        conf = np.where(ok, np.abs(et[:, g] - thr[g]), -np.inf)
        rank = np.argsort(-conf)
        rows = {}
        for q in (.25, .5):
            top = np.zeros(len(yt), bool); top[rank[:int(q*len(yt))]] = True
            rows[f'top{int(100*q)}'] = dict(acc_at_u=float(C[top, g].mean()), acc_at_6000=float(C[top, -1].mean()),
                                           rest_acc_at_u=float(C[~top, g].mean()), rest_acc_at_6000=float(C[~top, -1].mean()))
        Cs[str(u)] = rows
    res = dict(main=str(base), data=str(a.data), rep=a.rep, fs=fs, n_test=int(len(yt)), grid=GRID.tolist(),
               acc=acc.tolist(), acc_ci=ci.T.tolist(), dev_b_acc=dev_acc.tolist(), B=B, C=Cs, limit=a.limit)
    write(a.out/'metrics.json', res)
    log('TEST ACC', {int(GRID[g]): round(acc[g], 4) for g in (0, 4, g3, len(GRID) - 1)}, 'u*<=1200', round(B['le'][1200], 3),
        '>3000', round(B['gt_3000'], 3), 'never', round(B['never'], 3))


def report(a):
    units = sorted(a.root.glob('*/r*/metrics.json'))
    per = {f'{p.parent.parent.name}/{p.parent.name}': json.loads(p.read_text()) for p in units}
    fam = lambda name: next(label for prefix, label in FAMILY if name.startswith(prefix))
    groups = {label: [u for u in per if fam(u) == label] for _, label in FAMILY}
    pct = lambda x: f'{100*x:.1f}'
    show = [432, 768, 1200, 1968, 3000, 4464, 6000]
    gi = lambda m, u: m['grid'].index(u)
    L = ['# 固定窗口为什么不合理（TEST，主方法 S1 编码器，每个长度的阈值在 DEV-B 上按最大 ACC 锁定）', '',
         '每格为各数据集族 3 个单元的均值。长度 = 跳过开头 1500 点之后使用的信号点数。', '',
         '## A 信号越多，准确度越高', '', '| 数据 | ' + ' | '.join(f'{u}' for u in show) + ' |', '|---|' + '---:|'*len(show)]
    for label, us in groups.items():
        if us: L.append(f'| {label} | ' + ' | '.join(pct(np.mean([per[u]['acc'][gi(per[u], s)] for u in us])) for s in show) + ' |')
    L += ['', '## B 每条 read 需要的信号量差别很大', '',
          'u* = 从该长度起直到 6000 点，固定窗口判定一直正确的最短长度（只统计 ≥ 7500 点的 read）。'
          '在 3000 点窗口处："浪费" = 1200 点内已稳定判对；"不够" = 3000 点判错、更长后稳定判对。', '',
          '| 数据 | u* ≤ 432 % | ≤ 1200 % | ≤ 3000 % | > 3000 % | 始终判不对 % | 3000 处浪费 % | 3000 处不够 % | 判定翻转过 % |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label, us in groups.items():
        if not us: continue
        m = lambda f: pct(np.mean([f(per[u]['B']) for u in us]))
        L.append(f"| {label} | {m(lambda b: b['le']['432'])} | {m(lambda b: b['le']['1200'])} | {m(lambda b: b['le']['3000'])} | "
                 f"{m(lambda b: b['gt_3000'])} | {m(lambda b: b['never'])} | {m(lambda b: b['at_3000']['wasted'])} | "
                 f"{m(lambda b: b['at_3000']['too_short'])} | {m(lambda b: b['at_3000']['flips'])} |")
    L += ['', '## C 能早判的 read 在线就认得出来', '',
          '按早期窗口处 |margin − 阈值| 排序，取最有把握的 25% / 50%。', '',
          '| 数据 | 早期长度 | 前 25%：此时 ACC | 前 25%：6000 点 ACC | 前 50%：此时 ACC | 其余 50%：此时 ACC | 其余 50%：6000 点 ACC |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for label, us in groups.items():
        for e in EARLY:
            if not us: continue
            m = lambda q, k: pct(np.mean([per[u]['C'][str(e)][q][k] for u in us]))
            L.append(f"| {label} | {e} | {m('top25', 'acc_at_u')} | {m('top25', 'acc_at_6000')} | {m('top50', 'acc_at_u')} | "
                     f"{m('top50', 'rest_acc_at_u')} | {m('top50', 'rest_acc_at_6000')} |")
    L += ['', '## D 按 read 停止 vs 单一固定窗口', '',
          'oracle：每条 read 在 u* 停止（≥ 7500 点的 read，上界）。实测：chunk 回放（0.4 s），固定长度 = 消融 fixed_len（L 与阈值在 DEV-B 选）。', '',
          '| 数据 | oracle ACC % | oracle 平均长度 | 同 ACC 所需固定窗口 | PoreLoom ACC % / host 占孔 s | 最优固定长度 ACC % / host 占孔 s |',
          '|---|---:|---:|---:|---:|---:|']
    for label, us in groups.items():
        if not us: continue
        reach = [per[u]['B']['oracle']['fixed_window_same_acc'] for u in us]
        abl = []
        for u in us:
            d = a.ablation/u
            if (d/'main'/'metrics.json').exists() and (d/'fixed_len'/'metrics.json').exists():
                abl.append([json.loads((d/v/'metrics.json').read_text())['locked'] for v in ('main', 'fixed_len')])
        real = '–' if not abl else (f"{pct(np.mean([x[0]['acc'] for x in abl]))} / {np.mean([x[0]['host_mean_pore_seconds'] for x in abl]):.2f} | "
                                    f"{pct(np.mean([x[1]['acc'] for x in abl]))} / {np.mean([x[1]['host_mean_pore_seconds'] for x in abl]):.2f}")
        L.append(f"| {label} | {pct(np.mean([per[u]['B']['oracle']['acc'] for u in us]))} | "
                 f"{np.mean([per[u]['B']['oracle']['mean_used'] for u in us]):.0f} | "
                 f"{'达不到' if any(r is None for r in reach) else f'{np.mean(reach):.0f}'} | {real} |")
    text = '\n'.join(L) + '\n'; print(text)
    write(a.root/'report.json', per)
    if a.md: Path(a.md).write_text(text)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('run', 'report'))
    ap.add_argument('--main', type=Path); ap.add_argument('--data', type=Path); ap.add_argument('--out', type=Path)
    ap.add_argument('--root', type=Path); ap.add_argument('--ablation', type=Path); ap.add_argument('--md', type=Path)
    ap.add_argument('--rep', type=int, default=0); ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--limit', type=int, default=0, help='smoke test: rows per split')
    a = ap.parse_args()
    globals()[a.stage](a)
