#!/usr/bin/env python3
"""Observation-budget experiment on R10 mixed group 1 and sputum group 1.

R10 mixed uses usable budgets 6000, 9000, 12000, 15000 and 18000; sputum uses
6000, 9000 and 12000. Three training repeats share a stage-A initialization
across budgets within each repeat, then refit the streaming encoder and FACP.
Each policy selects its beta on DEV-B and is evaluated at 0.4-s intervals with
the nominal processing delay. Sputum budget 6000 repeat 0 reuses the main model.
The long signal cache must match all metadata and the first 9000 samples of the
main cache before use. Budgets exclude the initial 1500 skipped samples.

    budget_up.py prefix --dataset D [--out <root>]
    budget_up.py register --dataset D --main <main run> [--out registration[_D].json]
    budget_up.py run --data <cache> --stagea <stage_A.pt> --budget B --rep r --fs F --out <dir>
    budget_up.py report --dataset D --root <runs>/budget_up/D --main <main run dir> [--md summary_zh.md]
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA_PREP = Path(os.environ.get('PORELOOM_DATA_PREP', REPO/'experiments'))
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
    relative = '20261013_stream_real/run.py' if a.dataset == 'r10_run01' else '20261013_stream_sputum/prepare.py'
    reader = DATA_PREP/relative
    if not reader.is_file():
        raise FileNotFoundError(f'{reader} is required to rebuild raw-signal caches; set PORELOOM_DATA_PREP to the external preprocessing scripts or provide an existing cache to run --data')
    if a.dataset == 'r10_run01':
        spec = importlib.util.spec_from_file_location('stream_real_run', reader)
        real = importlib.util.module_from_spec(spec); spec.loader.exec_module(real)
        real.PREFIX = ds['prefix']
        for split in SPLITS: real.build_prefix('run_01', split, dest=out/split)
    else:   # the sputum POD5 reader writes OUT/stream/prefix/<split> from OUT/dataset/splits.csv (worker processes fork)
        spec = importlib.util.spec_from_file_location('sputum_prepare', reader)
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
        request=f'Observation budgets on {a.dataset}: {list(budgets)}',
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
        primary=dict(metric='TEST accuracy, host rejection, target rejection and mean host pore time',
                     comparison='Three training-repeat values and their means for each observation budget',
                     interval_seconds=0.4, service='nominal'),
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
        z = extract(net, data, a.fs, a.interval, H.HEAD['phase_seed'], a.device, a.budget)
        outcome = H.evaluate(ctrl, z, a.fs, a.device, service=H.HEAD['service'])
        np.savez(out/'test'/f'outcomes_{a.interval}_nominal.npz', read_id=z['read_id'], is_target=z['is_target'],
                 **{k: v for k, v in outcome.items() if k != 'missing'})
        row = dict(interval=a.interval, service='nominal', group='pooled', **H.metrics(outcome, z['is_target']))
        (out/'test'/'metrics.json').write_text(json.dumps(dict(beta=choice['beta'], rows=[row]), indent=2))
        log('TEST', {k: round(row[k], 4) for k in ('acc', 'host_command_reject', 'target_command_frr', 'host_mean_pore_seconds')})
    (out/'JOB_DONE').touch()


# ---------------------------------------------------------------- report


def report(a):
    dataset = DATASETS[a.dataset]
    rows = []
    for budget in dataset['budgets']:
        for rep in REPS:
            path = a.root/f'b{budget}_r{rep}'/'test/metrics.json'
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            selected = next(r for r in result['rows']
                            if r['interval'] == .4 and r['service'] == 'nominal' and r['group'] == 'pooled')
            rows.append(dict(budget=budget, replicate=rep, beta=result['beta'], **selected))
    metrics = ('acc', 'host_command_reject', 'target_command_frr', 'host_mean_pore_seconds')
    summary = {}
    lines = [f'# Observation budgets: {a.dataset}', '',
             'TEST at 0.4-s observations; separately refitted encoders and DEV-B-selected policies.', '',
             '| Budget | Repeat | Accuracy (%) | Host rejection (%) | Target rejection (%) | Host pore time (s) |',
             '|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['budget']} | {row['replicate']} | {100*row['acc']:.2f} | "
                     f"{100*row['host_command_reject']:.2f} | {100*row['target_command_frr']:.2f} | "
                     f"{row['host_mean_pore_seconds']:.3f} |")
    for budget in dataset['budgets']:
        values = [r for r in rows if r['budget'] == budget]
        if values:
            summary[str(budget)] = dict(n=len(values), **{k: float(np.mean([r[k] for r in values])) for k in metrics})
    main = json.loads((a.main/'test/metrics.json').read_text())
    reference = next(r for r in main['rows'] if r['interval'] == .4 and r['service'] == 'nominal' and r['group'] == 'pooled')
    text = '\n'.join(lines) + '\n'
    print(text)
    if a.md:
        Path(a.md).write_text(text)
    (a.root/'report.json').write_text(json.dumps(dict(rows=rows, summary=summary, main_reference=reference), indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prefix'); p.add_argument('--dataset', choices=DATASETS, required=True); p.add_argument('--out', type=Path)
    p = sub.add_parser('register'); p.add_argument('--dataset', choices=DATASETS, required=True)
    p.add_argument('--main', type=Path, required=True); p.add_argument('--out', type=Path)
    p = sub.add_parser('run')
    for k in ('--data', '--stagea', '--out'): p.add_argument(k, type=Path, required=True)
    p.add_argument('--budget', type=int, required=True, choices=BUDGETS); p.add_argument('--rep', type=int, required=True)
    p.add_argument('--fs', type=float, required=True); p.add_argument('--interval', type=float, choices=(0.4,), default=0.4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--limit', type=int, default=0, help='smoke only: class-balanced subset size per split')
    p.add_argument('--smoke-epochs', type=int, default=0, help='smoke only: S1 / head epochs and a 3-beta grid')
    p = sub.add_parser('report'); p.add_argument('--dataset', choices=DATASETS, default='r10_run01'); p.add_argument('--root', type=Path, required=True); p.add_argument('--main', type=Path, required=True)
    p.add_argument('--md')
    a = ap.parse_args(argv)
    globals()[a.cmd](a)


if __name__ == '__main__':
    main()
