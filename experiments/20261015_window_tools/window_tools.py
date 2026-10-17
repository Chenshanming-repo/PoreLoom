#!/usr/bin/env python3
"""Tool networks as sliding-window scorers + C-p vs streaming PoreLoom (unified stage A recipe and
unified MAD normalization).

Per unit (dataset, replicate) and tool (readcurrent | nanodeep | squigglenet):
  window net  the tool's complete native network (poreloom.baselines.models.build_model, upstream architecture incl. its
              own classifier; NanoDeep with its SE block) from a seeded random init (torch seed 20261009 + r), trained
              with the stage A recipe (poreloom.method.pretrain.STAGEA): each epoch one random valid 1000-sample window
              per TRAIN read from the grid raw[1500 + 1000k : 2500 + 1000k], k < 6 (inside raw[1500:7500]), MAD-
              normalized (data.mad_normalize); AdamW lr 1e-3, wd 1e-4, batch 128, clip 1, 30 epochs, keep the epoch
              with the lowest DEV-A per-read window cross entropy. Classes: 0 host, 1 target (trained from scratch).
  looks       at every 0.4 s chunk arrival n (head.HEAD phase seeds), ell = logit(target) - logit(host) of the latest
              window raw[e-1000:e], e = min(n, 7500), MAD-normalized (ablation.window_extract): first score at raw
              2500, none past the read end; the forced look scores raw[6500:7500].
  C-p head    as the default method (head.train_heads: DEV-A fit on two phase copies, beta by max DEV-B ACC, locked
              before TEST); 768-sample time buckets from stream length 1000 (7 buckets + forced).
  TEST        0.4 s, nominal service 0.0158 s, eject 0.1 s; locked point + every beta of the grid replayed.
Report: paired bootstrap over TEST reads (2000, seed 20261015) against streaming PoreLoom (runs/main/<ds>/poreloom_r<r>,
the effect of streaming with the decision layer fixed to C-p) and against the tool's own single-window decision
(runs/main/<ds>/<tool>_r<r>, max-ACC, measured service). Confound: the window scorer can act from raw 2500, PoreLoom
from raw 1932.

    window_tools.py run    --tool T --data <data>/<ds> --fs F --rep r --out <runs>/window_tools/<ds>/r<r>/<T> --device cuda:0
    window_tools.py report --root <runs>/window_tools --main <runs>/main [--ablation <runs>/ablation] [--native_cp <runs>/native_cp] [--md summary_zh.md]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO/'src')); sys.path.insert(0, str(REPO/'experiments'/'20261015_ablation'))
import ablation as AB  # noqa: E402
from poreloom.method import data as D, head as H, pipeline as P  # noqa: E402
from poreloom.method.pretrain import STAGEA, encoder_windows  # noqa: E402

TOOLS = ('readcurrent', 'nanodeep', 'squigglenet')
WINDOW, START, BUDGET = 1000, 1500, 7500
MAX_WINDOWS = (BUDGET - START)//WINDOW
UNITS = ['r10_run01/r0', 'r10_run02/r0', 'r10_run03/r0', 'r9/r0', 'r9/r1', 'r9/r2', 'sim_g1/r0', 'sim_g2/r0', 'sim_g3/r0',
         'sputum_g1/r0', 'sputum_g2/r0', 'sputum_g3/r0']
FAMILY = (('r10', 'R10'), ('r9', 'R9'), ('sim', 'Sim'), ('sputum', 'Sputum'))
NAMES = dict(readcurrent='ReadCurrent', nanodeep='NanoDeep', squigglenet='SquiggleNet')
NATIVE_CP = dict(readcurrent='ReadCurrent', squigglenet='SquiggleNet', nanodeep='NanoDeep（项目配方）',
                 nanodeep_upstream='NanoDeep（作者配方）')   # native_cp.py: main-run baselines reused
log = AB.log


class Scorer(torch.nn.Module):
    """ablation.window_extract interface: encode([B,1,W]) -> logits [B,2] (host, target); `temporary` = identity."""
    def __init__(self, net, tool):
        super().__init__(); self.net, self.tool = net, tool

    def encode(self, x):
        return self.net(x[:, 0] if self.tool in ('readcurrent', 'squigglenet') else x)


def train(a, out):
    if (out/'history.json').exists(): return
    from poreloom.baselines.models import build_model
    cfg = dict(STAGEA, window=WINDOW, window_start=START, max_windows=MAX_WINDOWS, seed=STAGEA['seed'] + a.rep)
    torch.manual_seed(cfg['seed']); rng = np.random.default_rng(cfg['seed'])
    train_d = D.load_split(a.data, 'train'); dev_a, _ = D.dev_halves(a.data)
    out.mkdir(parents=True, exist_ok=True)
    xt, vt = encoder_windows(train_d, START, WINDOW, MAX_WINDOWS, out/'train_windows.npy')
    xd, vd = encoder_windows(dev_a, START, WINDOW, MAX_WINDOWS)
    yt, yd = train_d.is_target.astype(np.int64), dev_a.is_target.astype(np.int64)
    net = Scorer(build_model(a.tool), a.tool).to(a.device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['encoder_lr'], weight_decay=cfg['weight_decay'])
    active, dev_active = np.flatnonzero(vt.any(1)), np.flatnonzero(vd.any(1))
    AB.write(out/'config.json', dict(tool=a.tool, config=cfg, train_active=len(active), dev_a_active=len(dev_active),
                                     params=sum(p.numel() for p in net.parameters()), valid_windows_train=vt.sum(0).tolist()))

    @torch.inference_mode()
    def validate():
        net.eval(); total = np.zeros(len(vd)); pairs = np.argwhere(vd)
        for s in range(0, len(pairs), 512):
            ids = pairs[s:s + 512]
            x = torch.as_tensor(xd[ids[:, 0], ids[:, 1]], device=a.device)[:, None]
            loss = F.cross_entropy(net.encode(x), torch.as_tensor(yd[ids[:, 0]], device=a.device), reduction='none')
            np.add.at(total, ids[:, 0], loss.float().cpu().numpy())
        return float((total[dev_active]/vd.sum(1)[dev_active]).mean())

    best, history, t0 = np.inf, [], time.time()
    for epoch in range(1, cfg['epochs'] + 1):
        net.train(); order = rng.permutation(active); total = 0.
        for s in range(0, len(order), cfg['batch_size']):
            ids = order[s:s + cfg['batch_size']]
            if len(ids) < 2: continue                     # BatchNorm needs > 1 row
            ks = np.array([rng.integers(vt[i].sum()) for i in ids])
            x = torch.as_tensor(np.asarray(xt[ids, ks]), device=a.device)[:, None]
            loss = F.cross_entropy(net.encode(x), torch.as_tensor(yt[ids], device=a.device))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['clip']); opt.step()
            total += loss.item()*len(ids)
        val = validate()
        history.append(dict(epoch=epoch, train_loss=total/len(active), dev_loss=val, elapsed_seconds=time.time() - t0))
        log(json.dumps(dict(tool=a.tool, **history[-1])))
        if val < best:
            best = val; torch.save(dict(model_state=net.net.state_dict(), epoch=epoch, dev_loss=val, config=cfg), out/'best.pt')
    (out/'train_windows.npy').unlink()
    AB.write(out/'history.json', history)


def run(a):
    P._setup(a.device)
    out = a.out; cache = out/'cache'
    AB.write(out/'run.json', dict(tool=a.tool, data=str(a.data), rep=a.rep, fs=a.fs, window=WINDOW, started=time.strftime('%Y-%m-%d %H:%M:%S')))
    train(a, out/'net')
    from poreloom.baselines.models import build_model
    net = build_model(a.tool); net.load_state_dict(torch.load(out/'net'/'best.pt', map_location='cpu')['model_state'])
    scorer = (Scorer(net, a.tool).to(a.device).eval(), torch.nn.Identity())
    dev_a, dev_b = D.dev_halves(a.data); test = D.load_split(a.data, 'test', verify=True)
    ph, cp = H.HEAD['phase_seed'], H.HEAD['copy_seed']
    wz = lambda data, tag, seed: AB.cached(cache/f'window_{tag}_0.4_{seed}.npz',
                                           lambda: AB.window_extract(scorer, data, a.fs, .4, seed, a.device, window=WINDOW))
    fit = [wz(dev_a, 'deva', s) for s in (ph, cp)]; sel = wz(dev_b, 'devb', ph); zt = wz(test, 'test', ph)
    assert fit[0]['n_buckets'] == 7 and zt['min_samples'] == WINDOW
    if not (out/'metrics.json').exists():
        AB.cp_train(fit, sel, a.fs, out/'head', a.device, a.rep)
        rows, locked = AB.cp_test(out/'head', zt, a.fs, a.device, out/'test_outcomes_0.4.npz')
        AB.write(out/'metrics.json', dict(tool=a.tool, locked=locked, test_curve=rows))
        log(a.tool, 'TEST', {k: round(v, 4) for k, v in locked.items()})
    (out/'JOB_DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))


def load(path):
    with np.load(path) as z:
        y = z['is_target'].astype(bool)
        return dict(id=z['read_id'].astype(str), y=y, ok=np.where(y, ~z['command_reject'], z['command_reject']).astype(float),
                    rej=z['command_reject'].astype(bool), pore=z['pore_seconds'].astype(float))


def report(a):
    rng = np.random.default_rng(20261015); rows, cmp = [], []
    for u in UNITS:
        ds, rep = u.split('/')
        o = dict(main=a.main/ds/f'poreloom_{rep}'/'test'/'outcomes_0.4_nominal.npz')
        if a.ablation: o['window_cp_rc1500'] = a.ablation/ds/rep/'window_cp'/'test_outcomes_0.4.npz'
        for t in TOOLS:
            o[f'win_{t}'] = a.root/ds/rep/t/'test_outcomes_0.4.npz'
            o[f'native_{t}'] = a.main/ds/f'{t}_{rep}'/'test'/'outcomes_max_acc_0.4_measured.npz'
        o['native_nanodeep_upstream'] = a.main/ds/f'nanodeep_upstream_{rep}'/'test'/'outcomes_max_acc_0.4_measured.npz'
        if a.native_cp:
            for t in NATIVE_CP: o[f'ncp_{t}'] = a.native_cp/ds/rep/t/'test_outcomes_0.4.npz'
        o = {k: load(p) for k, p in o.items() if p.exists()}
        if not any(k.startswith(('win_', 'ncp_')) for k in o): continue
        for k, v in o.items():
            assert np.array_equal(v['id'], o['main']['id']), (u, k)
            rows.append(dict(unit=u, method=k, acc=v['ok'].mean(), frr=v['rej'][v['y']].mean(), host_pore=v['pore'][~v['y']].mean()))
        idx = rng.integers(0, len(o['main']['id']), (2000, len(o['main']['id']))); host = ~o['main']['y']
        pairs = [('main', f'win_{t}') for t in TOOLS] + [(f'win_{t}', f'native_{t}') for t in TOOLS] + \
                [('win_nanodeep', 'native_nanodeep_upstream'), ('win_readcurrent', 'window_cp_rc1500')] + \
                [(f'ncp_{t}', f'native_{t}') for t in NATIVE_CP] + [('main', f'ncp_{t}') for t in NATIVE_CP] + \
                [(f'ncp_{t}', f'win_{t}') for t in TOOLS]
        for p, q in pairs:
            if p in o and q in o:
                d = o[p]['ok'] - o[q]['ok']
                cmp.append(dict(unit=u, a=p, b=q, dacc=float(d.mean()), ci=np.percentile(d[idx].mean(1), [2.5, 97.5]).tolist(),
                                dpore=float((o[p]['pore'] - o[q]['pore'])[host].mean())))
    label = dict(main='PoreLoom（流式，主方法）', window_cp_rc1500='PoreLoom stage A 滑窗 1500 + C-p（消融 window_cp）',
                 native_nanodeep_upstream='NanoDeep 单窗口（作者配方）',
                 **{f'win_{t}': f'{NAMES[t]} 滑窗 1000 + C-p' for t in TOOLS}, **{f'native_{t}': f'{NAMES[t]} 单窗口（项目配方）' for t in TOOLS},
                 **{f'ncp_{t}': f'{NATIVE_CP[t]} 已训练 3000 窗口滑窗 + C-p' for t in NATIVE_CP})
    fam = lambda u: next(l for p, l in FAMILY if u.startswith(p))
    L = ['# 工具网络滑窗 + C-p vs 流式 PoreLoom（TEST 0.4 s，DEV-B 最大 ACC 锁定）', '',
         '工具网络（完整原始结构）按 stage A 配方在 raw[1500:7500] 的 1000 点窗口上从随机初始化训练，逐 chunk 对最近 1000 点打分，'
         'C-p 动作头与主方法相同。滑窗最早在第 2500 点决策，PoreLoom 在第 1932 点。每格 = 各数据集族 3 个单元均值。', '',
         '## 绝对性能（ACC % / target 误拒 % / host 占孔 s）', '', '| 方法 | ' + ' | '.join(l for _, l in FAMILY) + ' |', '|---|' + '---|'*len(FAMILY)]
    for m in label:
        cells = []
        for _, l in FAMILY:
            rs = [r for r in rows if r['method'] == m and fam(r['unit']) == l]
            cells.append((f"{100*np.mean([r['acc'] for r in rs]):.1f} / {100*np.mean([r['frr'] for r in rs]):.1f} / "
                          f"{np.mean([r['host_pore'] for r in rs]):.2f}" + ('' if len(rs) == 3 else f'（{len(rs)}/3）')) if rs else '–')
        if any(c != '–' for c in cells): L.append(f'| {label[m]} | ' + ' | '.join(cells) + ' |')
    L += ['', '## 配对差（ΔACC pp [显著更好/显著更差 单元数] / Δhost 占孔 s）', '', '| 比较 | ' + ' | '.join(l for _, l in FAMILY) + ' |',
          '|---|' + '---|'*len(FAMILY)]
    for p, q in dict.fromkeys((c['a'], c['b']) for c in cmp):
        cells = []
        for _, l in FAMILY:
            xs = [c for c in cmp if (c['a'], c['b']) == (p, q) and fam(c['unit']) == l]
            cells.append(f"{100*np.mean([x['dacc'] for x in xs]):+.1f} [{sum(x['ci'][0] > 0 for x in xs)}/{sum(x['ci'][1] < 0 for x in xs)}] / "
                         f"{np.mean([x['dpore'] for x in xs]):+.2f}" if xs else '–')
        L.append(f'| {label[p]} − {label[q]} | ' + ' | '.join(cells) + ' |')
    text = '\n'.join(L) + '\n'; print(text)
    if a.md: Path(a.md).write_text(text); Path(a.md).with_suffix('.json').write_text(json.dumps(dict(rows=rows, cmp=cmp), indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('run', 'report'))
    ap.add_argument('--tool', choices=TOOLS); ap.add_argument('--data', type=Path); ap.add_argument('--fs', type=float)
    ap.add_argument('--rep', type=int, default=0); ap.add_argument('--out', type=Path); ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--root', type=Path); ap.add_argument('--main', type=Path); ap.add_argument('--ablation', type=Path)
    ap.add_argument('--native_cp', type=Path, help='<runs>/native_cp (native_cp.py outputs)')
    ap.add_argument('--md', type=Path)
    a = ap.parse_args()
    globals()[a.stage](a)
