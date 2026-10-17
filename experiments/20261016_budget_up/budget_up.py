#!/usr/bin/env python3
"""Decision-budget increase on one dataset.

Dataset: r10_run01 (R10.4.1, HG002 host vs Zymo target, 4 kHz, TRAIN 40k / DEV 20k / TEST 20k). The budget has to
bind, and here it does: 77-86 % of host and of target reads in every split are at least raw 19500 long, and the length
distributions of TRAIN / DEV / TEST match. sputum_g1 was not chosen because its TEST targets are much shorter than its
TRAIN / DEV targets (median 13.4k vs 18.8k / 22.5k samples): a higher budget would mix in that split shift.

Budget B = signal samples after the 1500-sample skip (the paper convention). B in {6000 (method default),
9000, 12000, 15000, 18000}: the forced decision lands at raw 1500 + B (7500 ... 19500, i.e. 1.9 ... 4.9 s at 4 kHz).
Each (B, replicate r) changes only the budget. Everything else is the method default (poreloom.method, docs/method.md):
  stage A  shared by every budget of replicate r: the method's stage A on the default 9000-sample cache (r0: the main
           run's stage_A.pt; r1, r2: `python -m poreloom.method stageA --replicate r`). Stage A only ever sees windows
           up to raw 9000.
  S1       StreamR0 from that stage A with R0_DEFAULTS, except max_samples = 1500 + B (training prefix lengths uniform up
           to the budget, normalizer prior over [1500, 1500 + B)), seed + r. DEV-A early stopping uses dev_lengths =
           the default ten plus 6000 + 1488 k up to the budget plus the budget's own stream length (the default for B = 6000).
  head     C-p on DEV-A looks up to the budget, with 768-sample buckets: ceil((B - 432) / 768) of them (6000 -> 8,
           9000 -> 12, 12000 -> 16, 15000 -> 19, 18000 -> 23). The forced look is scored on exactly the first 1500 + B
           samples. The time inputs n/7500 and log1p(n/4000) are the method's, but n is clamped at the budget, not at
           7500 (looks_capped; head.looks clamps at 7500, which would make every look past raw 7500 look like 7500;
           for B = 6000 the two are identical). beta is picked by max DEV-B ACC and locked before TEST.
  test     the same as pipeline.test (0.4 s primary; 0.2 / 1.0 s and zero service as sensitivity), plus the TEST replay
           of every beta at 0.4 s nominal (the ACC vs host pore time front) and decision diagnostics.
Data: the same reads with a 27000-sample prefix cache built from the original raw records (stream_real run.py reader;
budgets up to 22500 keep S1's full 1500-sample start jitter). Before use, the first 9000 samples and all metadata are
checked equal to the main-run cache.

Second dataset: sputum_g1 (5 kHz) at B in {6000, 9000, 12000}, 16500-sample
cache from the POD5 records (stream_sputum prepare.py reader), same design. Its B = 6000 r0 is the main run itself: the main run's S1 and head
are copied unchanged and only replayed on TEST by this script
(adds the front and the diagnostics); r1 / r2 at 6000 are trained as for R10, as same-replicate references.

    budget_up.py prefix   --dataset D [--out <root>]   local (.venv-nanodeep): long cache + equality check
    budget_up.py register --dataset D --main <main run> [--out registration[_D].json]   (before any training)
    budget_up.py run      --data <27000 root> --stagea <stage_A.pt> --budget B --rep r --fs 4000 --out <dir>
    budget_up.py report   --dataset D --root <runs>/budget_up/D --main <main run dir> [--md summary_zh.md]
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO/'src'))
from poreloom.method import data as D, head as H, pipeline as P  # noqa: E402
from poreloom.stream_controller import look_grid, look_inputs, one_look_per_bucket, Looks  # noqa: E402

SKIP, STRIDE, MIN_LOOK, BUCKET, SCALE = 1500, 48, 432, 768, 7500
REPS = (0, 1, 2)
SE = Path('/path/to/poreloom_work')
DATASETS = {
    'r10_run01': dict(fs=4000., prefix=27000, budgets=(6000, 9000, 12000, 15000, 18000),
                      old=SE/'results/stream_real_20261013/prefix/run_01',
                      local=SE/'results/budget_up_20261016/data/r10_run01_p27000'),
    'sputum_g1': dict(fs=5000., prefix=16500, budgets=(6000, 9000, 12000),
                      old=SE/'results/stream_sputum_20261013/groups/g1/stream/prefix',
                      local=SE/'results/budget_up_20261016/data/sputum_g1_p16500',
                      splits_dir=SE/'results/stream_sputum_20261013/groups/g1/dataset'),
}
BUDGETS = tuple(sorted({b for d in DATASETS.values() for b in d['budgets']}))
SPLITS = ('train', 'dev', 'test')
BOOT, BOOT_SEED = 2000, 20261016


def n_buckets(budget):
    return int(np.ceil((budget - MIN_LOOK)/BUCKET))


def dev_lengths(budget):
    from poreloom.stream_training import R0_DEFAULTS
    top = budget//STRIDE*STRIDE
    extra = [6000 + 1488*k for k in range(1, 100) if 6000 + 1488*k < top]
    return tuple(sorted(set(R0_DEFAULTS['dev_lengths']) | set(extra) | {top}))


# ---------------------------------------------------------------- data

def prefix(a):
    ds = DATASETS[a.dataset]; out = a.out or ds['local']
    if a.dataset == 'r10_run01':
        spec = importlib.util.spec_from_file_location('stream_real_run', REPO/'experiments'/'20261013_stream_real'/'run.py')
        real = importlib.util.module_from_spec(spec); spec.loader.exec_module(real)
        real.PREFIX = ds['prefix']
        for split in SPLITS: real.build_prefix('run_01', split, dest=out/split)
    else:   # the sputum POD5 reader writes OUT/stream/prefix/<split> from OUT/dataset/splits.csv (worker processes fork)
        spec = importlib.util.spec_from_file_location('sputum_prepare', REPO/'experiments'/'20261013_stream_sputum'/'prepare.py')
        prep = importlib.util.module_from_spec(spec); sys.modules['sputum_prepare'] = prep; spec.loader.exec_module(prep)
        base = out.parent/(out.name + '_build'); (base/'dataset').mkdir(parents=True, exist_ok=True)
        link = base/'dataset'/'splits.csv'
        if not link.exists(): link.symlink_to(ds['splits_dir']/'splits.csv')
        prep.PREFIX, prep.OUT = ds['prefix'], base
        prep.prefix()
        out.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            if not (out/split).exists(): (base/'stream'/'prefix'/split).replace(out/split)
    for split in SPLITS: check(out/split, ds['old']/split, ds['prefix'])


def check(new, old, width):
    """The long cache holds the same reads as the main-run cache: equal metadata, equal first 9000 samples."""
    with np.load(new/'metadata.npz', allow_pickle=False) as zn, np.load(old/'metadata.npz', allow_pickle=False) as zo:
        keys = tuple(zo.files)
        for k in keys:
            if not np.array_equal(zn[k], zo[k]): raise ValueError(f'{new}: metadata {k} differs from {old}')
    an, ao = np.load(new/'adc.npy', mmap_mode='r'), np.load(old/'adc.npy', mmap_mode='r')
    if an.shape[1] != width or len(an) != len(ao): raise ValueError(f'{new}: shape {an.shape}')
    for s in range(0, len(ao), 4096):
        if not np.array_equal(an[s:s + 4096, :ao.shape[1]], ao[s:s + 4096]): raise ValueError(f'{new}: rows {s}+ differ')
    (new/'equality_check.json').write_text(json.dumps(dict(
        checked=time.strftime('%Y-%m-%d %H:%M:%S'), against=str(old), metadata=list(keys), samples=int(ao.shape[1]),
        reads=int(len(ao)), result='equal'), indent=2) + '\n')
    print(time.strftime('%H:%M:%S'), new, 'equal to', old, flush=True)


def register(a):
    ds = DATASETS[a.dataset]
    out = a.out or HERE/('registration.json' if a.dataset == 'r10_run01' else f'registration_{a.dataset}.json')
    if out.exists(): raise FileExistsError(out)
    from poreloom.stream_training import R0_DEFAULTS
    budgets = ds['budgets']
    manifests = {s: json.loads((ds['local']/s/'manifest.json').read_text())['files'] for s in SPLITS}
    out.write_text(json.dumps(dict(
        written=time.strftime('%Y-%m-%d %H:%M:%S') + ', before any budget-increase training',
        request='decision-budget increase on one dataset: B in {6000, 9000, 12000, 15000, 18000}',
        design=__doc__.split('    budget_up.py prefix')[0].strip(),
        dataset=dict(name=a.dataset, fs=ds['fs'], prefix_samples=ds['prefix'], files=manifests, equality_with=str(ds['old'])),
        budgets_signal=list(budgets), forced_raw=[SKIP + b for b in budgets], replicates=list(REPS),
        n_buckets={b: n_buckets(b) for b in budgets}, dev_lengths={b: list(dev_lengths(b)) for b in budgets},
        seeds=dict(stage_a='STAGEA seed + r', s1=f"{R0_DEFAULTS['seed']} + r", head_train=f"{H.HEAD['train_seed']} + r",
                   test_phase=H.HEAD['phase_seed']),
        reference=dict(main_run=str(a.main), note=(
            'B = 6000 r0 is rerun here through this script (code-path check against the main run, same recipe); every '
            'comparison uses this script\'s runs' if a.dataset == 'r10_run01' else
            'B = 6000 r0 = the main run: its S1 and head copied unchanged, replayed on TEST by this script')),
        primary=dict(metric='TEST command ACC and host mean pore seconds, 0.4 s, nominal service, max DEV-B ACC beta',
                     comparison='each B vs B = 6000 of the same replicate: paired bootstrap over TEST reads '
                                f'({BOOT} draws, seed {BOOT_SEED}); summary = mean over the 3 replicates and how many '
                                'replicates have a 95 % CI excluding 0',
                     noise='run-to-run noise of the method is about 0.3 pp ACC (2026-10-14 rerun)'),
        secondary=['target command FRR, host command / effective reject', 'TEST front of every beta (ACC vs host pore s)',
                   'decision diagnostics: signal used at the decision, share decided at the forced look, share of reads '
                   'that end without a command', '0.2 / 1.0 s chunk and zero-service sensitivity'],
    ), indent=2, ensure_ascii=False) + '\n')


# ---------------------------------------------------------------- method at budget B

def looks_capped(z, fs):
    """head.looks with the time inputs clamped at the budget instead of 7500 (same n/7500, log1p(n/4000) scales)."""
    budget = z['budget']
    grid = {k: z[k] for k in ('n_buckets', 'min_samples') if k in z}
    n, t, valid, forced, bucket = look_grid(z['m'], fs=fs, max_samples=budget, **grid)
    valid &= np.isfinite(z['ell'])
    valid = one_look_per_bucket(valid, np.where(valid, bucket, -1))
    bucket = np.where(valid, bucket, -1)
    x = look_inputs(np.nan_to_num(z['ell']), np.minimum(n, budget), valid, None, max_samples=budget)
    x[..., 1] *= budget/SCALE                     # n/budget -> n/7500 with n <= budget
    return Looks(t, valid, forced, bucket, x, np.asarray(z['raw_length'])/fs, z['is_target'])


H.looks = looks_capped    # train_heads / evaluate call head.looks; identical to the original for budget 7500


def extract(net, data, fs, interval, seed, device, budget):
    z = H.extract(net, data, fs, interval, seed, device, budget=SKIP + budget)
    z['n_buckets'] = n_buckets(budget)
    return z


def limited(data, n, seed=20261016):
    """Smoke only: a class-balanced subset of n reads."""
    rng = np.random.default_rng(seed); y = np.asarray(data.is_target, bool)
    rows = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), n//2, replace=False) for c in (False, True)]))
    return D.subset(data, rows)


def run(a):
    from poreloom.stream_training import R0_DEFAULTS, load_stream_r0, train_stream_r0
    P._setup(a.device); log = P.log
    out = a.out; out.mkdir(parents=True, exist_ok=True)
    cap = SKIP + a.budget
    cfg = dict(data=str(a.data.resolve()), stagea=str(a.stagea.resolve()), budget_signal=a.budget, forced_raw=cap,
               replicate=a.rep, fs=a.fs, interval=a.interval, n_buckets=n_buckets(a.budget), limit=a.limit,
               s1=dict(max_samples=cap, dev_lengths=list(dev_lengths(a.budget)), seed=R0_DEFAULTS['seed'] + a.rep))
    if (out/'run.json').exists():
        old = json.loads((out/'run.json').read_text()); old.pop('started', None)
        if old != cfg: raise ValueError(f'{out} belongs to another configuration')
    else: (out/'run.json').write_text(json.dumps(dict(cfg, started=time.strftime('%Y-%m-%d %H:%M:%S')), indent=2) + '\n')

    def split(name):
        d = D.load_split(a.data, name, verify=name == 'test')
        if d.adc.shape[1] < cap + R0_DEFAULTS['jitter_max']: raise ValueError(f'prefix {d.adc.shape[1]} too short for {cap}')
        return limited(d, a.limit) if a.limit else d

    def halves():
        dev_a, dev_b = D.dev_halves(a.data)
        return (limited(dev_a, a.limit//2), limited(dev_b, a.limit//2)) if a.limit else (dev_a, dev_b)

    s1_over = dict(max_samples=cap, dev_lengths=dev_lengths(a.budget), seed=R0_DEFAULTS['seed'] + a.rep)
    head_over = dict(train_seed=H.HEAD['train_seed'] + a.rep, select='max_acc')
    if a.smoke_epochs: s1_over['epochs'] = a.smoke_epochs; head_over['epochs'] = a.smoke_epochs; head_over['betas'] = [.3, 1., 3.]

    if not (out/'stream'/'done.json').exists():
        log('stage stream', cfg['s1'])
        train = split('train'); dev_a, _ = halves()
        train_stream_r0(train, dev_a, a.stagea, out/'stream', device=a.device, log=log, **s1_over)
        best = torch.load(out/'stream'/'best.pt', map_location='cpu')
        peak = torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else 0.
        (out/'stream'/'done.json').write_text(json.dumps(dict(best_epoch=int(best['epoch']), peak_gib=round(peak, 2)), indent=2))
    net = load_stream_r0(out/'stream'/'best.pt', a.device)[0].eval()
    if net.max_samples != cap: raise ValueError('stream checkpoint cap differs from the budget')

    if not (out/'head'/'selection.json').exists():
        log('stage head', n_buckets(a.budget), 'buckets')
        dev_a, dev_b = halves()
        fit = [extract(net, dev_a, a.fs, a.interval, s, a.device, a.budget) for s in (H.HEAD['phase_seed'], H.HEAD['copy_seed'])]
        sel = extract(net, dev_b, a.fs, a.interval, H.HEAD['phase_seed'], a.device, a.budget)
        choice = H.train_heads(fit, sel, a.fs, out/'head', a.device, log=log, **head_over)
        choice['written'] = time.strftime('%Y-%m-%d %H:%M:%S') + ', before any TEST input'
        (out/'head'/'selection.json').write_text(json.dumps(choice, indent=2))
        log('selected beta', choice['beta'], 'SEL ACC', round(choice['sel']['acc'], 4))

    if not (out/'test'/'metrics.json').exists():
        log('stage test')
        choice = json.loads((out/'head'/'selection.json').read_text())
        data = split('test'); (out/'test').mkdir(parents=True, exist_ok=True)
        ctrl = H.load_controller(out/'head'/'controllers.pt', choice['beta'], a.device)
        rows, extra = [], {}
        for interval in [a.interval] + [i for i in (.2, .4, 1.) if i != a.interval]:
            z = extract(net, data, a.fs, interval, H.HEAD['phase_seed'], a.device, a.budget)
            for service, tag in ((H.HEAD['service'], 'nominal'), (0., 'zero')):
                o = H.evaluate(ctrl, z, a.fs, a.device, service=service)
                np.savez(out/'test'/f'outcomes_{interval}_{tag}.npz', read_id=z['read_id'], is_target=z['is_target'],
                         **{k: v for k, v in o.items() if k != 'missing'})
                rows.append(dict(interval=interval, service=tag, group='pooled', **H.metrics(o, z['is_target'])))
                if interval == a.interval and tag == 'nominal': extra['diagnostics'] = diagnostics(z, o, a.fs, a.budget)
            if interval == a.interval:
                saved = torch.load(out/'head'/'controllers.pt', map_location='cpu')
                extra['front'] = [dict(beta=b, **H.metrics(H.evaluate(H.load_controller(out/'head'/'controllers.pt', b, a.device),
                                                                     z, a.fs, a.device), z['is_target'])) for b in saved['betas']]
        (out/'test'/'metrics.json').write_text(json.dumps(dict(beta=choice['beta'], rows=rows, **extra), indent=2))
        for r in rows:
            if r['interval'] == a.interval and r['service'] == 'nominal':
                log('TEST', {k: round(r[k], 4) for k in ('acc', 'host_command_reject', 'target_command_frr', 'host_mean_pore_seconds')})
    (out/'JOB_DONE').touch()


def diagnostics(z, o, fs, budget):
    """Where the locked policy decides: stream samples used at the command, share at the forced look, share without one."""
    lk = looks_capped(z, fs)
    n = look_grid(z['m'], fs=fs, max_samples=SKIP + budget, n_buckets=n_buckets(budget))[0]
    y = np.asarray(z['is_target'], bool); k = np.asarray(o['steps']); has = np.asarray(o['command'], bool) & (k >= 0)
    i = np.flatnonzero(has)
    used = np.full(len(y), np.nan); used[i] = np.minimum(n[i, k[i]], SKIP + budget) - SKIP
    forced = np.zeros(len(y), bool); forced[i] = lk.forced[i, k[i]]
    q = lambda v: [float(x) for x in np.nanpercentile(v, [25, 50, 75, 90])] if np.isfinite(v).any() else None
    out = {}
    for name, m in (('host', ~y), ('target', y)):
        out[name] = dict(no_command=float((~has[m]).mean()), forced=float(forced[m].mean()),
                         used_q25_50_75_90=q(used[m]), used_mean=float(np.nanmean(used[m])),
                         reject_at_forced=float((forced & np.asarray(o['command_reject'], bool))[m].mean()),
                         read_shorter_than_budget=float((np.asarray(z['raw_length'])[m] < SKIP + budget).mean()))
    return out


# ---------------------------------------------------------------- report

def correct(o):
    y = o['is_target'].astype(bool); rej = o['command_reject'].astype(bool)
    return np.where(y, ~rej, rej).astype(float)


def boot(new, old, mask, seed=BOOT_SEED):
    d = (new - old)[mask]; rng = np.random.default_rng(seed)
    draws = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT)])
    return float(d.mean()), [float(x) for x in np.quantile(draws, [.025, .975])]


def report(a):
    root = a.root; table, deltas = {}, {}
    ds = DATASETS[a.dataset]; BUDGETS, FS = ds['budgets'], ds['fs']
    for b in BUDGETS:
        for r in REPS:
            f = root/f'b{b}_r{r}'/'test'/'metrics.json'
            if not f.exists(): continue
            m = json.loads(f.read_text())
            row = next(x for x in m['rows'] if x['interval'] == .4 and x['service'] == 'nominal')
            table[(b, r)] = dict(row, beta=m['beta'], diagnostics=m['diagnostics'], front=m['front'],
                                 sens={f"{x['interval']}_{x['service']}": x['acc'] for x in m['rows']})
    for (b, r) in table:
        ref = root/f'b6000_r{r}'/'test'/'outcomes_0.4_nominal.npz'
        if b == 6000 or not ref.exists(): continue
        new, old = dict(np.load(root/f'b{b}_r{r}'/'test'/'outcomes_0.4_nominal.npz')), dict(np.load(ref))
        assert np.array_equal(new['read_id'], old['read_id'])
        host = ~new['is_target'].astype(bool)
        deltas[(b, r)] = dict(acc=boot(correct(new), correct(old), np.ones(len(host), bool)),
                              host_pore=boot(new['pore_seconds'], old['pore_seconds'], host))
    main = json.loads((a.main/'test'/'metrics.json').read_text())
    main_row = next(x for x in main['rows'] if x['interval'] == .4 and x['service'] == 'nominal' and x['group'] == 'pooled')
    lines = [f'# PoreLoom 决策预算提高（{a.dataset}）', '',
             f'预算 = 跳过前 1500 点之后的信号点数；强制决策在 raw 1500 + 预算（{FS/1000:g} kHz）。每个预算重训 S1 和 C-p，stage A 按重复共用。',
             'TEST 0.4 s、名义服务时间、DEV-B 最大 ACC 锁定的 β。Δ 相对同一重复的预算 6000，配对 bootstrap（2000 次）95% CI。', '',
             f"代码路径检查：本脚本 b6000_r0 ACC {table.get((6000, 0), {}).get('acc', float('nan'))*100:.2f}% vs 主实验 "
             f"{main_row['acc']*100:.2f}%；host 占孔 {table.get((6000, 0), {}).get('host_mean_pore_seconds', float('nan')):.3f} vs "
             f"{main_row['host_mean_pore_seconds']:.3f} s。", '',
             '| 预算 | 强制决策 s | 重复 | ACC % | host 拒绝 % | target 误拒 % | host 占孔 s | ΔACC pp [95% CI] | Δhost 占孔 s [95% CI] |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for b in BUDGETS:
        for r in REPS:
            if (b, r) not in table: continue
            t = table[(b, r)]; d = deltas.get((b, r))
            da = f"{d['acc'][0]*100:+.2f} [{d['acc'][1][0]*100:+.2f}, {d['acc'][1][1]*100:+.2f}]" if d else '—'
            dp = f"{d['host_pore'][0]:+.3f} [{d['host_pore'][1][0]:+.3f}, {d['host_pore'][1][1]:+.3f}]" if d else '—'
            lines.append(f"| {b} | {(SKIP + b)/FS:.2f} | r{r} | {t['acc']*100:.2f} | {t['host_command_reject']*100:.2f} | "
                         f"{t['target_command_frr']*100:.2f} | {t['host_mean_pore_seconds']:.3f} | {da} | {dp} |")
    lines += ['', '## 重复均值', '', '| 预算 | n | ACC % | host 占孔 s | ΔACC pp 均值 | CI 不含 0 的重复（ACC 升 / 降） | 决策时 host 用到的信号中位数 | target 无指令 % | 强制决策 %（host / target） |',
              '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    summary = {}
    for b in BUDGETS:
        rs = [r for r in REPS if (b, r) in table]
        if not rs: continue
        acc = np.mean([table[(b, r)]['acc'] for r in rs]); pore = np.mean([table[(b, r)]['host_mean_pore_seconds'] for r in rs])
        ds = [deltas[(b, r)]['acc'] for r in rs if (b, r) in deltas]
        up = sum(lo > 0 for _, (lo, hi) in ds); down = sum(hi < 0 for _, (lo, hi) in ds)
        dg = [table[(b, r)]['diagnostics'] for r in rs]
        med = np.mean([g['host']['used_q25_50_75_90'][1] for g in dg if g['host']['used_q25_50_75_90']])
        summary[b] = dict(n=len(rs), acc=acc, host_pore=pore, dacc=float(np.mean([d for d, _ in ds])) if ds else None, up=up, down=down)
        lines.append(f"| {b} | {len(rs)} | {acc*100:.2f} | {pore:.3f} | "
                     f"{'—' if not ds else f'{np.mean([d for d, _ in ds])*100:+.2f}'} | {up} / {down} | {med:.0f} | "
                     f"{np.mean([g['target']['no_command'] for g in dg])*100:.1f} | "
                     f"{np.mean([g['host']['forced'] for g in dg])*100:.1f} / {np.mean([g['target']['forced'] for g in dg])*100:.1f} |")
    lines += ['', '## ACC vs host 占孔前沿（TEST 0.4 s，每个 β；重复 r0）', '',
              '同一 host 占孔下的 ACC 才是公平比较：最大 ACC 选点会随预算把 host 占孔一起推高。', '',
              '| 预算 | host 占孔 ≤1.0 s 最高 ACC | ≤1.5 s | ≤2.0 s | ≤3.0 s |', '|---:|---:|---:|---:|---:|']
    for b in BUDGETS:
        if (b, 0) not in table: continue
        f = table[(b, 0)]['front']
        best = lambda c: max([x['acc'] for x in f if x['host_mean_pore_seconds'] <= c], default=float('nan'))
        lines.append(f"| {b} | " + ' | '.join(f'{best(c)*100:.2f}' for c in (1., 1.5, 2., 3.)) + ' |')
    lines += ['', '## chunk 间隔与服务时间敏感性（ACC %，重复均值）', '', '| 预算 | 0.2 s | 0.4 s | 1.0 s | 0.4 s 零服务 |', '|---:|---:|---:|---:|---:|']
    for b in BUDGETS:
        rs = [r for r in REPS if (b, r) in table]
        if not rs: continue
        s = lambda k: np.mean([table[(b, r)]['sens'][k] for r in rs])*100
        lines.append(f"| {b} | {s('0.2_nominal'):.2f} | {s('0.4_nominal'):.2f} | {s('1.0_nominal'):.2f} | {s('0.4_zero'):.2f} |")
    text = '\n'.join(lines) + '\n'
    print(text)
    if a.md: Path(a.md).write_text(text)
    (root/'report.json').write_text(json.dumps(dict(summary={str(k): v for k, v in summary.items()},
                                                    deltas={f'b{b}_r{r}': v for (b, r), v in deltas.items()}), indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prefix'); p.add_argument('--dataset', choices=DATASETS, required=True); p.add_argument('--out', type=Path)
    p = sub.add_parser('register'); p.add_argument('--dataset', choices=DATASETS, required=True)
    p.add_argument('--main', type=Path, required=True); p.add_argument('--out', type=Path)
    p = sub.add_parser('run')
    for k in ('--data', '--stagea', '--out'): p.add_argument(k, type=Path, required=True)
    p.add_argument('--budget', type=int, required=True, choices=BUDGETS); p.add_argument('--rep', type=int, required=True)
    p.add_argument('--fs', type=float, required=True); p.add_argument('--interval', type=float, default=H.HEAD['interval'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--limit', type=int, default=0, help='smoke only: class-balanced subset size per split')
    p.add_argument('--smoke-epochs', type=int, default=0, help='smoke only: S1 / head epochs and a 3-beta grid')
    p = sub.add_parser('report'); p.add_argument('--dataset', choices=DATASETS, default='r10_run01'); p.add_argument('--root', type=Path, required=True); p.add_argument('--main', type=Path, required=True)
    p.add_argument('--md')
    a = ap.parse_args(argv)
    globals()[a.cmd](a)


if __name__ == '__main__':
    main()
