#!/usr/bin/env python3
"""Score separability behind the cross-dataset results: is the cross-dataset failure a
shifted operating point (scores still rank host vs target, AUROC high) or lost discrimination (AUROC near 0.5)?

PoreLoom: the frozen stream encoder's logit margin ell at fixed sample counts (N_LOOKS), reads too short for a look
are left out of that look. Baselines: minus the native-window margin saved in each test's scores.npz (that margin is host-positive; eligible reads only).
Pairs: each model on its own TEST (self), and the matched cross pairs r10_run0k <-> sputum_gk in both directions and
both sampling variants (reverse: at4k / native; forward: at5k / native). 'oracle ACC' = best single threshold on the
margin, fit on the TEST reads themselves (an upper bound for re-calibration, not a deployable number).

    separability.py --main <runs>/main --data <data> --fwd <runs>/cross_domain --rev <runs>/cross_domain_rev --out <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_domain import METHODS, NAMES, log  # noqa: E402

N_LOOKS = (3000, 5000, 7500)
PAIRS = (('r10_run01', 'sputum_g1'), ('r10_run02', 'sputum_g2'), ('r10_run03', 'sputum_g3'))


def auroc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1*(n1 + 1)/2)/(n1*n0))


def oracle(s, y):
    o = np.argsort(-s); ys = y[o]
    tp = np.concatenate([[0], np.cumsum(ys)]); fp = np.concatenate([[0], np.cumsum(~ys)])
    return float(((tp + (len(y) - y.sum()) - fp)/len(y)).max())         # predict target above the cut


def cells(a):
    out = []                                                                # (label, model dir, data dir, fs, test dir)
    for r, g in PAIRS:
        out += [('self', r, r, a.main/r, a.data/r, 4000., a.main/r), ('self', g, g, a.main/g, a.data/g, 5000., a.main/g)]
        out += [('rev_at4k', r, g, a.main/r, a.data/f'{g}_at4k', 4000., a.rev/'at4k'/f'{r}_on_{g}'),
                ('rev_native', r, g, a.main/r, a.data/g, 5000., a.rev/'native'/f'{r}_on_{g}'),
                ('fwd_at5k', g, r, a.main/g, a.data/f'{r}_at5k', 5000., a.fwd/'at5k'/f'{g}_on_{r}'),
                ('fwd_native', g, r, a.main/g, a.data/r, 4000., a.fwd/'native'/f'{g}_on_{r}')]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('main', 'data', 'fwd', 'rev', 'out'): ap.add_argument(f'--{k}', type=Path, required=True)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    from poreloom.method import data as D, pipeline as P
    P._setup(a.device)
    rows = []
    for kind, src, tgt, model, data, fs, test in cells(a):
        prefix = D.load_split(data, 'test', verify=False); y = np.asarray(prefix.is_target, bool)
        net = P._encoder(model/'poreloom_r0', a.device)
        n = np.tile(np.array(N_LOOKS, np.int64), (len(prefix), 1))
        n = np.where(prefix.raw_length[:, None] >= n, n, -1)
        _, ell = __import__('poreloom.stream_controller', fromlist=['encode_looks']).encode_looks(net, prefix, n, a.device)
        for j, k in enumerate(N_LOOKS):
            ok = np.isfinite(ell[:, j])
            rows.append(dict(kind=kind, src=src, tgt=tgt, method='poreloom', look=k, n=int(ok.sum()),
                             auroc=auroc(ell[ok, j], y[ok]), oracle_acc=oracle(ell[ok, j], y[ok]),
                             host_mean=float(ell[ok & ~y, j].mean()), target_mean=float(ell[ok & y, j].mean())))
        for m in METHODS[1:]:
            with np.load(test/f'{m}_r0'/'test'/'scores.npz') as z:
                s, ok, ids = -z['margin'], z['eligible'].astype(bool), z['read_id'].astype(str)   # tool margin is host-positive
            assert np.array_equal(ids, np.asarray(prefix.read_id).astype(str))
            rows.append(dict(kind=kind, src=src, tgt=tgt, method=m, look='native', n=int(ok.sum()),
                             auroc=auroc(s[ok], y[ok]), oracle_acc=oracle(s[ok], y[ok]),
                             host_mean=float(s[ok & ~y].mean()), target_mean=float(s[ok & y].mean())))
        log(kind, src, '->', tgt, {r['method']: round(r['auroc'], 3) for r in rows[-len(METHODS) - 2:]})
    (a.out/'separability.json').write_text(json.dumps(rows, indent=1))
    KINDS = [('self', '自测（R10 / 痰液各自）'), ('fwd_at5k', '正方向 痰液→R10（5 kHz）'), ('fwd_native', '正方向 痰液→R10（原始 4 kHz）'),
             ('rev_at4k', '反方向 R10→痰液（4 kHz）'), ('rev_native', '反方向 R10→痰液（原始 5 kHz）')]
    L = ['# 跨数据集的分数可分性（AUROC）', '',
         'AUROC 衡量分数能否把 target 排在 host 前面，与阈值无关；"oracle ACC" = 在测试集自身上取最优单一阈值的 ACC（重新校准的上限，'
         '不可部署）。PoreLoom 用冻结流式编码器在固定样本数处的 logit margin；工具用原生窗口 margin。3 对匹配组合（run0k ↔ g k）的均值。', '']
    for look in N_LOOKS:
        L += [f'## PoreLoom，第 {look} 个样本处', '', '| 场景 | AUROC | oracle ACC % | host 均值 | target 均值 |', '|---|---:|---:|---:|---:|']
        for kind, title in KINDS:
            rs = [r for r in rows if r['kind'] == kind and r['method'] == 'poreloom' and r['look'] == look]
            if kind == 'self':
                for tag, pre in (('R10', 'r10'), ('痰液', 'sputum')):
                    q = [r for r in rs if r['src'].startswith(pre)]
                    L.append(f"| 自测 {tag} | {np.mean([r['auroc'] for r in q]):.3f} | {100*np.mean([r['oracle_acc'] for r in q]):.1f} | "
                             f"{np.mean([r['host_mean'] for r in q]):+.2f} | {np.mean([r['target_mean'] for r in q]):+.2f} |")
                continue
            L.append(f"| {title} | {np.mean([r['auroc'] for r in rs]):.3f} | {100*np.mean([r['oracle_acc'] for r in rs]):.1f} | "
                     f"{np.mean([r['host_mean'] for r in rs]):+.2f} | {np.mean([r['target_mean'] for r in rs]):+.2f} |")
        L.append('')
    L += ['## 工具（原生窗口 margin）：AUROC', '', '| 场景 | ' + ' | '.join(NAMES[m] for m in METHODS[1:]) + ' |', '|---|' + '---:|'*(len(METHODS) - 1)]
    for kind, title in KINDS:
        groups = [('自测 R10', 'r10'), ('自测 痰液', 'sputum')] if kind == 'self' else [(title, '')]
        for tag, pre in groups:
            L.append(f'| {tag} | ' + ' | '.join(f"{np.mean([r['auroc'] for r in rows if r['kind'] == kind and r['method'] == m and r['src'].startswith(pre)]):.3f}"
                                               for m in METHODS[1:]) + ' |')
    text = '\n'.join(L) + '\n'; print(text); (a.out/'separability_zh.md').write_text(text)


if __name__ == '__main__':
    main()
