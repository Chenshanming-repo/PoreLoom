"""Single-window native baselines (NanoDeep, ReadCurrent, SquiggleNet) trained from scratch on one dataset.

    python -m poreloom.method.baselines --tool readcurrent --data <dataset root> --out <run dir> --fs 5000 --device cuda:0

Same dataset layout, DEV halves and TEST phases as the PoreLoom pipeline (poreloom.method). Recipe = the sputum scratch
baselines (experiments/20261013_stream_sputum/bench.py): native architecture from a seeded random init (seed 20261014
+ r), every weight trained on TRAIN's native window (NanoDeep raw[1000:4000], others raw[1500:4500]) with AdamW lr 1e-3,
weight decay 1e-4, batch 128, at most 100 epochs, patience 5 on DEV-A cross entropy (seed 20261012 + r). One decision
per read when the window is complete (next chunk arrival + measured single-read service). Threshold on the host
margin by maximum DEV-B command ACC (also native argmax and OP1 = target FRR <= 0.5%), locked before TEST.
TEST: 0.4 s primary; 0.2 / 1.0 s and zero service as sensitivity. Outputs mirror the PoreLoom run directory:
  fit/best.pt, selection.json (DEV-B rules, measured service), test/metrics.json, test/outcomes_<rule>_<interval>_<service>.npz
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import native, native_policy as policy
from ..baselines.preprocessing import CONTRACTS

TOOLS = ('nanodeep', 'readcurrent', 'squigglenet')
INIT_SEED = 20261014
TRAINING = dict(seed=20261012, epochs=100, patience=5, batch_size=128, eval_batch=128, backbone_lr=1e-3, head_lr=1e-3,
                weight_decay=1e-4, clip=1.)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# NanoDeep authors' training settings (third_party/NanoDeep: nanodeep_trainmodel.py defaults, read_deep/rt.py train()):
# SGD lr 1e-3 momentum 0.9, no weight decay or clipping, MultiLabelSoftMarginLoss on one-hot labels, batch 50, 30 epochs,
# keep the epoch with the best validation accuracy (--save_best). Validation = DEV-A here; window/normalization stay
# the online single-window contract (raw[1000:4000], (x - mean) / var of the window), as in the README 3000-sample example.
UPSTREAM = dict(nanodeep=dict(seed=20261012, epochs=30, batch_size=50, eval_batch=128, lr=1e-3, momentum=.9))


def fit_upstream(tool, init, train, dev_a, output, device, cfg):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(cfg['seed']); rng = np.random.default_rng(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    train_rows, dev_rows = native._check_data(tool, train), native._check_data(tool, dev_a)
    model = native._load(tool, init, device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])
    loss_fn = torch.nn.MultiLabelSoftMarginLoss()
    yt, yd = native.network_labels(tool, train['is_target']), native.network_labels(tool, dev_a['is_target'])
    (output/'config.json').write_text(json.dumps(dict(tool=tool, recipe='upstream', config=cfg, init=str(init),
                                                      eligible_train=len(train_rows), eligible_dev_a=len(dev_rows)), indent=2))
    best, history = -1., []
    for epoch in range(1, cfg['epochs'] + 1):
        started = time.perf_counter(); model.train(); total = 0.
        order = rng.permutation(train_rows)
        for s in range(0, len(order), cfg['batch_size']):
            part = order[s:s + cfg['batch_size']]
            logits = native._forward(model, train['x'][part], tool, device)
            loss = loss_fn(logits, torch.nn.functional.one_hot(torch.as_tensor(yt[part], device=device), 2).float())
            opt.zero_grad(); loss.backward(); opt.step(); total += float(loss.detach()) * len(part)
        model.eval(); correct = 0
        with torch.inference_mode():
            for s in range(0, len(dev_rows), cfg['eval_batch']):
                part = dev_rows[s:s + cfg['eval_batch']]
                correct += int((native._forward(model, dev_a['x'][part], tool, device).argmax(1).cpu().numpy() == yd[part]).sum())
        acc = correct/len(dev_rows)
        entry = dict(epoch=epoch, train_loss=total/len(train_rows), dev_accuracy=acc, improved=acc > best, seconds=time.perf_counter() - started)
        history.append(entry); print(json.dumps(dict(tool=tool, **entry)), flush=True)
        if acc > best:
            best = acc; torch.save(dict(model_state=model.state_dict(), epoch=epoch, dev_accuracy=acc, recipe='upstream'), output/'best.pt')
    (output/'history.json').write_text(json.dumps(history, indent=2))
    (output/'done.json').write_text(json.dumps(dict(status='completed', epochs_completed=cfg['epochs'],
                                                    best_epoch=max(history, key=lambda h: h['dev_accuracy'])['epoch'], best_dev_accuracy=best), indent=2))
    return output/'best.pt'


def _meta(prefix, split):
    return dict(read_id=prefix.read_id, is_target=np.asarray(prefix.is_target, bool), raw_length=prefix.raw_length,
                role=prefix.role, split=split)


def fit(a):
    best = a.out/'fit'/'best.pt'
    if (a.out/'fit'/'done.json').exists(): return best
    from ..baselines.models import build_model
    init = a.out/'random_init.pt'
    torch.manual_seed(INIT_SEED + a.replicate)
    torch.save(dict(model_state=build_model(a.tool).state_dict(), init='random', seed=INIT_SEED + a.replicate), init)
    train = D.load_split(a.data, 'train'); dev_a, _ = D.dev_halves(a.data)
    train.role, dev_a.role = np.full(len(train), 'train'), np.full(len(dev_a), 'dev')   # role names differ across caches
    if (a.out/'fit').exists():                                   # interrupted fit: start over
        for p in sorted((a.out/'fit').iterdir()): p.unlink()
        (a.out/'fit').rmdir()
    if a.recipe == 'upstream':
        cfg = dict(UPSTREAM[a.tool], seed=UPSTREAM[a.tool]['seed'] + a.replicate)
        return fit_upstream(a.tool, init, native.prepare(a.tool, train), native.prepare(a.tool, dev_a), a.out/'fit', a.device, cfg)
    cfg = dict(TRAINING, seed=TRAINING['seed'] + a.replicate)
    return native.fit(a.tool, init, native.prepare(a.tool, train), native.prepare(a.tool, dev_a), a.out/'fit', a.device, cfg)


def select(a):
    if (a.out/'selection.json').exists(): return
    dev = D.load_split(a.data, 'dev'); half = D.dev_half(dev.read_id)
    rows_a, rows_b = np.flatnonzero(half == 0), np.flatnonzero(half == 1)
    dev.role = np.full(len(dev), 'dev')
    service = native.measure_service(a.tool, a.out/'fit'/'best.pt', dev, rows_a, a.device, repeats=64)
    seconds = float(service['p50_seconds'])
    b = D.subset(dev, rows_b); meta = _meta(b, 'dev_b')
    scores = native.score(a.tool, a.out/'fit'/'best.pt', native.prepare(a.tool, b), a.device)
    rules = policy.select_native(scores, meta, seconds, CONTRACTS[a.tool]['raw_end'])
    doc = dict(tool=a.tool, service=service, service_seconds=seconds, rules=rules,
               written=time.strftime('%Y-%m-%d %H:%M:%S') + ', before any TEST input')
    (a.out/'selection.json').write_text(json.dumps(doc, indent=2))
    log('DEV-B max_acc', rules['max_acc']['threshold'], round(rules['max_acc']['dev_metrics']['acc'], 4), 'service', round(seconds, 5))


def test(a):
    if (a.out/'test'/'metrics.json').exists(): return
    sel = json.loads((a.out/'selection.json').read_text())
    prefix = D.load_split(a.data, 'test', verify=True); meta = _meta(prefix, 'test')
    with np.load(Path(a.data)/'test'/'metadata.npz', allow_pickle=False) as z:
        repeat = z['repeat'] if 'repeat' in z.files else np.ones(len(prefix), int)
    scores = native.score(a.tool, a.out/'fit'/'best.pt', native.prepare(a.tool, prefix), a.device)
    (a.out/'test').mkdir(parents=True, exist_ok=True)
    np.savez(a.out/'test'/'scores.npz', **scores)
    groups = {'pooled': np.ones(len(prefix), bool), **{f'repeat_{r}': repeat == r for r in np.unique(repeat)}}
    rows = []
    for name, rule in sel['rules'].items():
        for interval, tag, service in ((.4, 'measured', sel['service_seconds']), (.2, 'measured', sel['service_seconds']),
                                       (1., 'measured', sel['service_seconds']), (.4, 'zero', 0.)):
            o = policy.native_policy(scores, meta, threshold=rule['threshold'], service=service, delta=interval,
                                     raw_end=CONTRACTS[a.tool]['raw_end'])
            np.savez(a.out/'test'/f'outcomes_{name}_{interval}_{tag}.npz', read_id=meta['read_id'], is_target=meta['is_target'],
                     repeat=repeat, **o)
            for g, mask in groups.items():
                rows.append(dict(rule=name, interval=interval, service=tag, group=g, **policy.metrics(o, meta, rows=mask)))
    (a.out/'test'/'metrics.json').write_text(json.dumps(dict(threshold=sel['rules']['max_acc']['threshold'], rows=rows), indent=2))
    for r in rows:
        if r['rule'] == 'max_acc' and r['interval'] == .4 and r['service'] == 'measured':
            log(r['group'], {k: round(r[k], 4) for k in ('acc', 'host_command_reject_rate', 'target_command_frr', 'host_mean_pore_seconds')})


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m poreloom.method.baselines', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tool', choices=TOOLS, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--fs', type=float, required=True, help='sampling rate of the data in Hz (replay times = samples / fs)')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--replicate', type=int, default=0, help='training replicate r: init and training seeds + r')
    ap.add_argument('--recipe', choices=('project', 'upstream'), default='project',
                    help="project: shared scratch recipe (default); upstream: the tool authors' training settings (NanoDeep only)")
    a = ap.parse_args(argv)
    if a.recipe == 'upstream' and a.tool not in UPSTREAM: ap.error(f'no upstream recipe for {a.tool}')
    policy.FS = a.fs
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if str(a.device).startswith('cuda'): torch.cuda.set_device(a.device)
    run = dict(tool=a.tool, data=str(a.data.resolve()), fs=a.fs, replicate=a.replicate, recipe=a.recipe)
    training = UPSTREAM[a.tool] if a.recipe == 'upstream' else TRAINING
    if not (a.out/'run.json').exists(): (a.out/'run.json').write_text(json.dumps(dict(run, training=training, init_seed=INIT_SEED), indent=2))
    elif any(json.loads((a.out/'run.json').read_text()).get(k) != v for k, v in run.items()):
        raise ValueError('run directory belongs to another tool, dataset, sampling rate or replicate')
    for st in (fit, select, test):
        log('stage', st.__name__); st(a)


if __name__ == '__main__':
    main()
