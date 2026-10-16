"""Copied from experiments/20261013_nanodeep_baselines/native.py (logic unchanged).

Native baseline windows, full-checkpoint R9 adaptation, and single-read service.

This module performs no work at import time. The experiment runner owns registration,
quality gates, DEV A/B routing, threshold locking and TEST access. Adaptation is the
new fixed-window 20-epoch recipe, not the original full-read tiling training recipe.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from ..baselines.models import build_model, checkpoint_state, torch_load
from ..baselines import preprocessing as _pp
from ..baselines.preprocessing import CONTRACTS, prepare_windows
from ..prefix_data import load_prefix_cache

DEFAULTS = dict(seed=20261012, epochs=20, patience=3, batch_size=128, eval_batch=128,
                backbone_lr=1e-4, head_lr=3e-4, weight_decay=1e-4, clip=1.)
HEADS = dict(nanodeep='liner', readcurrent='fc_layers', cafes='fc_layers', squigglenet='fc')


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _write(path, value):
    with Path(path).open('x') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _prefix(value):
    return load_prefix_cache(value, verify=True) if isinstance(value, (str, Path)) else value


def _tool(tool):
    if tool not in CONTRACTS:
        raise ValueError(f'unknown native baseline {tool}')
    return CONTRACTS[tool]


def _rows(prefix, rows):
    rows = np.arange(len(prefix)) if rows is None else np.asarray(rows)
    if rows.ndim != 1 or rows.dtype.kind not in 'iu' or np.any(rows < 0) or np.any(rows >= len(prefix)):
        raise ValueError('rows must be valid one-dimensional integer indices')
    return rows.astype(np.int64, copy=False)


def prepare(tool, prefix, rows=None):
    """Return native normalized windows [N,C,3000], keeping all requested parents.

    Geometry-invalid or normalization-invalid parents retain a zero window and a
    false eligibility flag. CAFES is processed in small batches so temporary feature
    arrays do not multiply the final [N,6,3000] allocation; callers can also chunk rows.
    """
    contract = _tool(tool); prefix = _prefix(prefix); rows = _rows(prefix, rows)
    lo, hi = contract['raw_start'], contract['raw_end']
    if prefix.adc.shape[1] < hi:
        raise ValueError(f'prefix cache must retain at least {hi} ADC samples')
    channels = 6 if tool == 'cafes' else 1
    x = np.zeros((len(rows), channels, _pp.WINDOW), np.float32)
    eligible = np.zeros(len(rows), bool)
    candidates = np.flatnonzero(prefix.raw_length[rows] >= hi)
    batch = 32 if tool == 'cafes' else 256
    for start in range(0, len(candidates), batch):
        part = candidates[start:start + batch]
        raw = (prefix.pa(rows[part], lo, hi) if contract['units'] == 'pA'
               else np.asarray(prefix.adc[rows[part], lo:hi], dtype=np.int16))
        try:
            values, valid = prepare_windows(raw, tool)
        except FloatingPointError:
            # CAFES upstream features can be undefined for a constant input. Isolate
            # such reads without discarding the other valid parents in their batch.
            for local, destination in enumerate(part):
                try:
                    values, valid = prepare_windows(raw[local:local + 1], tool)
                except FloatingPointError:
                    continue
                if values.ndim == 2:
                    values = values[:, None]
                if bool(valid[0]) and np.isfinite(values).all():
                    x[destination] = values[0]; eligible[destination] = True
            continue
        if values.ndim == 2:
            values = values[:, None]
        valid = np.asarray(valid, bool) & np.isfinite(values).all(axis=(1, 2))
        x[part[valid]] = values[valid]
        eligible[part[valid]] = True
    return dict(x=x, eligible=eligible, is_target=np.asarray(prefix.is_target[rows], bool),
                raw_length=np.asarray(prefix.raw_length[rows]), read_id=np.asarray(prefix.read_id[rows]),
                role=np.asarray(prefix.role[rows]))


def network_labels(tool, is_target):
    contract = _tool(tool)
    return np.where(np.asarray(is_target, bool), contract['microbe_index'], contract['host_index']).astype(np.int64)


def _check_data(tool, data):
    _tool(tool)
    n = len(data['read_id']); channels = 6 if tool == 'cafes' else 1
    x = data['x']
    if x.dtype != np.float32 or x.shape != (n, channels, _pp.WINDOW):
        raise ValueError(f'prepared inputs must be float32 [N,C,{_pp.WINDOW}]')
    for key in ('is_target', 'raw_length', 'eligible'):
        if np.shape(data[key]) != (n,):
            raise ValueError(f'prepared {key} must have one value per parent')
    if np.asarray(data['eligible']).dtype != np.bool_:
        raise ValueError('prepared eligibility must be boolean')
    if len(set(map(str, data['read_id']))) != n:
        raise ValueError('duplicate prepared read_id')
    rows = np.flatnonzero(data['eligible'])
    if np.any(data['raw_length'][rows] < CONTRACTS[tool]['raw_end']):
        raise ValueError('eligible prepared row is too short for native window')
    # Validate incrementally to avoid a full-size temporary boolean CAFES tensor.
    for start in range(0, len(rows), 128):
        if not np.isfinite(x[rows[start:start + 128]]).all():
            raise ValueError('nonfinite eligible prepared inputs')
    return rows


def _load(tool, checkpoint, device):
    model = build_model(tool)
    # Only pinned trusted local checkpoints reach this module. The shared loader
    # handles archived trainer dictionaries (which may also contain optimizer RNGs).
    model.load_state_dict(checkpoint_state(torch_load(checkpoint)), strict=True)
    return model.to(device).eval()


def _forward(model, x, tool, device):
    x = torch.as_tensor(np.ascontiguousarray(x), device=device)
    if tool in ('readcurrent', 'squigglenet'):
        x = x[:, 0]  # Their native forward() inserts the channel itself.
    logits = model(x)
    if logits.shape != (len(x), 2) or not torch.isfinite(logits).all():
        raise FloatingPointError('native model produced invalid two-class logits')
    return logits


@torch.inference_mode()
def _evaluate(model, tool, data, rows, device, batch_size):
    model.eval()  # DEV never updates BatchNorm or activates dropout.
    labels = network_labels(tool, data['is_target'])
    total, correct = 0., 0
    for start in range(0, len(rows), batch_size):
        part = rows[start:start + batch_size]
        logits = _forward(model, data['x'][part], tool, device)
        y = torch.as_tensor(labels[part], device=device)
        total += float(F.cross_entropy(logits, y, reduction='sum').cpu())
        correct += int((logits.argmax(1) == y).sum().cpu())
    return dict(ce=total / len(rows), accuracy=correct / len(rows), eligible_reads=len(rows))


def fit(tool, checkpoint, train, dev_a, output, device='cuda:0', config=None):
    """Fine-tune every full native model weight; choose by DEV-A CE, epoch 0 included.

    Role and read-ID hash checks protect the early-stopping source. The outer
    registration additionally pins the source arrays and their split membership.
    """
    supplied = dict(config or {})
    if 'max_epochs' in supplied:
        if 'epochs' in supplied:
            raise ValueError('provide only epochs or max_epochs')
        supplied['epochs'] = supplied.pop('max_epochs')
    if set(supplied) - set(DEFAULTS):
        raise ValueError(f'unknown adaptation config keys: {sorted(set(supplied) - set(DEFAULTS))}')
    cfg = {**DEFAULTS, **supplied}
    if (cfg['epochs'] < 0 or cfg['patience'] < 1 or cfg['batch_size'] < 1 or cfg['eval_batch'] < 1
            or cfg['backbone_lr'] <= 0 or cfg['head_lr'] <= 0 or cfg['clip'] <= 0 or cfg['weight_decay'] < 0):
        raise ValueError('invalid adaptation config')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    train_rows, dev_rows = _check_data(tool, train), _check_data(tool, dev_a)
    if set(map(str, train['read_id'])) & set(map(str, dev_a['read_id'])):
        raise ValueError('TRAIN and DEV A read_id overlap')
    for data, allowed, label in ((train, {'train'}, 'TRAIN'), (dev_a, {'dev', 'dev_a'}, 'DEV A')):
        if 'role' in data and not set(map(str, data['role'])) <= allowed:
            raise ValueError(f'{label} input has disallowed roles')
    if any(int(hashlib.sha256(str(r).encode()).hexdigest()[:8], 16) % 2 for r in dev_a['read_id']):
        raise ValueError('early stopping accepts only DEV A read hashes')
    for data, rows in ((train, train_rows), (dev_a, dev_rows)):
        if set(np.asarray(data['is_target'])[rows].tolist()) != {False, True}:
            raise ValueError('both classes required among eligible TRAIN and DEV A reads')
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed']); rng = np.random.default_rng(cfg['seed'])
    if torch.device(device).type == 'cuda':
        torch.cuda.set_device(device); torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = _load(tool, checkpoint, device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    head = list(getattr(model, HEADS[tool]).parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    if not head or not backbone:
        raise ValueError('head/backbone optimizer groups must both contain parameters')
    optimizer = torch.optim.AdamW([dict(params=backbone, lr=cfg['backbone_lr']),
                                   dict(params=head, lr=cfg['head_lr'])], weight_decay=cfg['weight_decay'])
    output.mkdir(parents=True, exist_ok=False)
    provenance = dict(tool=tool, config=cfg, init_checkpoint=str(Path(checkpoint).resolve()),
                      init_checkpoint_sha256=_sha(checkpoint), full_checkpoint_strict=True,
                      training_recipe='R9 native first-window full-model fine-tuning; all weights; BN updates TRAIN only',
                      selection='minimum read-weighted DEV-A cross entropy; epoch 0 is a candidate',
                      contract=dict(CONTRACTS[tool]), head_module=HEADS[tool],
                      optimizer_groups=dict(backbone_lr=cfg['backbone_lr'], head_lr=cfg['head_lr']),
                      train_reads=len(train['read_id']), eligible_train=len(train_rows),
                      dev_a_reads=len(dev_a['read_id']), eligible_dev_a=len(dev_rows),
                      train_read_ids_sha256=hashlib.sha256('\n'.join(map(str, train['read_id'])).encode()).hexdigest(),
                      dev_a_read_ids_sha256=hashlib.sha256('\n'.join(map(str, dev_a['read_id'])).encode()).hexdigest())
    _write(output / 'config.json', provenance)
    labels = network_labels(tool, train['is_target'])
    best, stale, history = float('inf'), 0, []
    with (output / 'history.jsonl').open('x') as stream:
        for epoch in range(cfg['epochs'] + 1):
            started = time.perf_counter(); loss_sum = 0.; count = 0
            if epoch:
                model.train()
                order = rng.permutation(train_rows)
                for start in range(0, len(order), cfg['batch_size']):
                    part = order[start:start + cfg['batch_size']]
                    optimizer.zero_grad(set_to_none=True)
                    logits = _forward(model, train['x'][part], tool, device)
                    y = torch.as_tensor(labels[part], device=device)
                    loss = F.cross_entropy(logits, y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite TRAIN cross entropy')
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip']); optimizer.step()
                    loss_sum += float(loss.detach().cpu()) * len(part); count += len(part)
            evaluation = _evaluate(model, tool, dev_a, dev_rows, device, cfg['eval_batch'])
            improved = evaluation['ce'] < best
            entry = dict(epoch=epoch, train_ce=loss_sum/count if count else None, dev_ce=evaluation['ce'],
                         dev_accuracy=evaluation['accuracy'], improved=improved, seconds=time.perf_counter()-started)
            history.append(entry)
            stream.write(json.dumps(entry, allow_nan=False) + '\n'); stream.flush()
            print(json.dumps(dict(tool=tool, **entry), allow_nan=False), flush=True)
            if improved:
                best, stale = evaluation['ce'], 0
                temporary = output / 'best.pt.partial'
                torch.save(dict(model_state=model.state_dict(), epoch=epoch, dev_ce=best,
                                dev_accuracy=evaluation['accuracy'], **provenance), temporary)
                temporary.replace(output / 'best.pt')
            else:
                stale += 1
                if stale >= cfg['patience']:
                    break
    _write(output / 'history.json', history)
    _write(output / 'done.json', dict(status='completed', epochs_completed=history[-1]['epoch'],
                                    best_epoch=min(history, key=lambda x: x['dev_ce'])['epoch'],
                                    best_dev_ce=best, checkpoint_sha256=_sha(output / 'best.pt'),
                                    stop_reason='patience' if stale >= cfg['patience'] else 'max_epochs'))
    return output / 'best.pt'


@torch.inference_mode()
def score(tool, checkpoint, data, device='cuda:0', batch_size=128):
    """Return host margin and true native argmax decisions, retaining NaN no-calls."""
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    rows = _check_data(tool, data); model = _load(tool, checkpoint, device)
    margin = np.full(len(data['read_id']), np.nan, np.float32)
    reject = np.zeros(len(margin), bool)
    host = CONTRACTS[tool]['host_index']; target = CONTRACTS[tool]['microbe_index']
    for start in range(0, len(rows), batch_size):
        part = rows[start:start + batch_size]
        logits = _forward(model, data['x'][part], tool, device)
        margin[part] = (logits[:, host] - logits[:, target]).cpu().numpy()
        reject[part] = (logits.argmax(1) == host).cpu().numpy()  # ties are class 0, per torch argmax
    return dict(margin=margin, eligible=np.asarray(data['eligible']).copy(), native_reject=reject,
                read_id=np.asarray(data['read_id']).copy())


def measure_service(tool, checkpoint, prefix, rows, device, repeats=64):
    """Warm model, one read at a time: preprocessing + upload + forward + sync.

    Source must be DEV hash half A. Model loading and initial warmup are excluded.
    Samples cycle deterministically through the supplied valid DEV-A rows.
    """
    if repeats < 1:
        raise ValueError('repeats must be positive')
    prefix = _prefix(prefix); rows = _rows(prefix, rows)
    if not len(rows):
        raise ValueError('no DEV A rows for service measurement')
    if (not set(map(str, prefix.role[rows])) <= {'dev', 'dev_a'}
            or any(int(hashlib.sha256(str(r).encode()).hexdigest()[:8], 16) % 2 for r in prefix.read_id[rows])):
        raise ValueError('service measurement accepts only DEV A rows')
    # Eligibility screening is outside timing; each measured call repeats its own
    # native preprocessing, including the CPU CAFES feature construction.
    prepared = prepare(tool, prefix, rows)
    rows = rows[prepared['eligible']]
    del prepared
    if not len(rows):
        raise ValueError('no eligible DEV A rows for service measurement')
    model = _load(tool, checkpoint, device)
    cuda = torch.device(device).type == 'cuda'
    def sync():
        if cuda:
            torch.cuda.synchronize(device)
    def one(row):
        data = prepare(tool, prefix, np.array([row]))
        if not data['eligible'][0]:
            raise ValueError('service measurement eligibility changed')
        _forward(model, data['x'], tool, device)
    durations = []
    with torch.inference_mode():
        for i in range(min(8, len(rows))):
            one(rows[i])
        sync()
        for i in range(repeats):
            sync(); start = time.perf_counter()
            one(rows[i % len(rows)])
            sync(); durations.append(time.perf_counter() - start)
    return dict(tool=tool, repeats=repeats, available_reads=len(rows), warmup=min(8, len(rows)), device=str(device),
                scope='DEV A native preprocessing + CPU-to-device transfer + single-read forward + synchronization; model resident',
                p50_seconds=float(np.quantile(durations, .5)), p95_seconds=float(np.quantile(durations, .95)),
                mean_seconds=float(np.mean(durations)), seconds=durations)
