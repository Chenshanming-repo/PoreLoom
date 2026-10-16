"""Training recipes of the four native baselines (seed 20260929, TRAIN / model_DEV).

- NanoDeep: project fixed-window controller. Adam 1e-3, batch 32,
  ReduceLROnPlateau(0.5, patience 5), early stopping on DEV CE (30-300 epochs,
  patience 25, min_delta 1e-4), deterministic algorithms.
- ReadCurrent / CAFES: patched upstream ReadCurrent trainer on class-balanced
  tiles (``tiles.prepare_tiles``). Adam 1e-3, validation every 200 steps on the
  full DEV set, LR halved after 10 stale validations (up to 4 times), then stop;
  at most 300 positive-class epochs. Batch 1024 (ReadCurrent) / 512 (CAFES).
- SquiggleNet: official loop. Adam 1e-3, batch 1000, 20 epochs; before every
  update the full DEV set is scored in train mode and the best mean batch
  accuracy is checkpointed.

Output files per fit: ``best.pt`` (selected weights; trainer checkpoints keep them
under ``model_state``), ``config.json``, ``metrics.json`` and a training log.
"""
from copy import deepcopy
import csv
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .models import build_model, cafes_preprocessor, readcurrent_trainer
from .native_source import NativeSource
from .preprocessing import cafes_features, nanodeep_inputs, squigglenet_inputs

SEED = 20260929


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def seed_everything(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.device(device).type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def deterministic(device):
    if torch.device(device).type == 'cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def runtime(device):
    device = torch.device(device)
    return dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, device=str(device),
                gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))


def _new_output(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    return output


# ----------------------------------------------------------------------------- NanoDeep

@dataclass
class EarlyStopping:
    """DEV-loss checkpoint selection with patience; never stops before ``min_epochs``."""
    min_epochs: int = 30
    max_epochs: int = 300
    patience: int = 25
    min_delta: float = 1e-4
    epoch: int = 0
    best_loss: float = math.inf
    best_epoch: int = 0
    anchor: float = math.inf
    stale: int = 0

    def update(self, loss):
        """Record one epoch; return whether it is the best so far."""
        loss = float(loss)
        if not math.isfinite(loss) or loss < 0:
            raise ValueError('validation loss must be finite and nonnegative')
        self.epoch += 1
        improved = loss < self.best_loss
        if improved:
            self.best_loss, self.best_epoch = loss, self.epoch
        if loss < self.anchor - self.min_delta:
            self.anchor, self.stale = loss, 0
        else:
            self.stale += 1
        return improved

    @property
    def stop_reason(self):
        if self.epoch >= self.min_epochs and self.stale >= self.patience:
            return 'validation_patience'
        if self.epoch >= self.max_epochs:
            return 'max_epochs'
        return None


def nanodeep_data(native, limit=None):
    """Normalized raw[1000:4000] inputs and eligible train/dev rows."""
    source = NativeSource(native)
    adc = source.windows('adc_1000_4000')
    rows = {split: source.rows(split)[:limit] for split in ('train', 'dev')}
    inputs = np.zeros(adc.shape, np.float32)
    eligible = np.zeros(source.n, bool)
    for split_rows in rows.values():
        candidates = split_rows[source.eligible('adc_1000_4000')[split_rows]]
        for start in range(0, len(candidates), 4096):
            selected = candidates[start:start + 4096]
            inputs[selected], eligible[selected] = nanodeep_inputs(np.asarray(adc[selected]))
    rows = {split: split_rows[eligible[split_rows]] for split, split_rows in rows.items()}
    for split, split_rows in rows.items():
        if set(source.labels[split_rows]) != {0, 1}:
            raise ValueError(f'both classes required among eligible {split} rows')
    return inputs, source.labels, rows


@torch.inference_mode()
def _nanodeep_dev(model, inputs, labels, rows, device, batch_size):
    model.eval()
    loss = 0.
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start + batch_size]
        x = torch.as_tensor(inputs[selected], device=device)[:, None, :]
        y = torch.as_tensor(labels[selected], dtype=torch.long, device=device)
        logits = model(x)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('nonfinite dev logits')
        loss += float(F.cross_entropy(logits, y, reduction='sum').cpu())
    return loss / len(rows)


def train_nanodeep(native, output, *, device='cuda', seed=SEED, lr=1e-3, batch_size=32,
                   min_epochs=30, max_epochs=300, limit=None):
    output = _new_output(output)
    inputs, labels, rows = nanodeep_data(native, limit)
    torch.set_num_threads(2)
    seed_everything(seed, device)
    deterministic(device)
    model = build_model('nanodeep').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=5)
    stopper = EarlyStopping(min_epochs=min_epochs, max_epochs=max_epochs)
    config = dict(tool='nanodeep', native_source=str(Path(native).resolve()), seed=seed, lr=lr, batch_size=batch_size,
                  input='ADC raw[1000:4000], (x-mean)/var', labels='microbe0 host1',
                  early_stopping=dict(min_epochs=min_epochs, max_epochs=max_epochs, patience=25, min_delta=1e-4),
                  scheduler=dict(name='ReduceLROnPlateau', factor=.5, patience=5),
                  eligible_train=len(rows['train']), eligible_dev=len(rows['dev']), limit=limit,
                  parameters=sum(p.numel() for p in model.parameters()), runtime=runtime(device))
    write_json(output / 'config.json', config)
    fields = ['epoch', 'train_ce', 'dev_ce', 'lr', 'next_lr', 'best_epoch', 'stale', 'epoch_seconds']
    started = time.monotonic()
    with (output / 'training.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        while stopper.stop_reason is None:
            tick = time.monotonic()
            model.train()
            order = np.random.permutation(rows['train'])
            total = 0.
            lr_now = optimizer.param_groups[0]['lr']
            for start in range(0, len(order), batch_size):
                selected = order[start:start + batch_size]
                x = torch.as_tensor(inputs[selected], device=device)[:, None, :]
                y = torch.as_tensor(labels[selected], dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite training loss')
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(selected)
            dev_ce = _nanodeep_dev(model, inputs, labels, rows['dev'], device, batch_size)
            if stopper.update(dev_ce):
                torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, output / 'best.pt')
            scheduler.step(dev_ce)
            row = dict(epoch=stopper.epoch, train_ce=total / len(order), dev_ce=dev_ce, lr=lr_now,
                       next_lr=optimizer.param_groups[0]['lr'], best_epoch=stopper.best_epoch, stale=stopper.stale,
                       epoch_seconds=time.monotonic() - tick)
            writer.writerow(row)
            stream.flush()
            print(json.dumps(row), flush=True)
    model.load_state_dict(torch.load(output / 'best.pt', map_location=device))
    metrics = dict(tool='nanodeep', epochs_completed=stopper.epoch, best_epoch=stopper.best_epoch,
                   best_dev_ce=stopper.best_loss, stop_reason=stopper.stop_reason,
                   final_dev_ce=_nanodeep_dev(model, inputs, labels, rows['dev'], device, batch_size),
                   elapsed_seconds=time.monotonic() - started, early_stopping=asdict(stopper))
    write_json(output / 'metrics.json', metrics)
    return metrics


# ----------------------------------------------------------------------------- ReadCurrent / CAFES

def train_tiles(tool, tiles, output, *, device='cuda', seed=SEED, batch_size=None, epochs=300, lr=1e-3,
                tolerance=10, smoke_steps=None):
    """Fit ReadCurrent or CAFES on ``tiles/<tool>/data/{pos,neg}`` with the patched trainer."""
    if tool not in ('readcurrent', 'cafes'):
        raise ValueError('tile training is for readcurrent or cafes')
    batch_size = batch_size or (1024 if tool == 'readcurrent' else 512)
    data = Path(tiles) / tool / 'data'
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(2)
    trainer = readcurrent_trainer()
    if tool == 'cafes':
        preprocessor = cafes_preprocessor()

        def tensor_batch(array, indices, label, target):
            x = cafes_features(np.asarray(array[indices]).copy(), preprocessor).to(target)
            return x, torch.full((len(indices),), label, dtype=torch.long, device=target)
        trainer.tensor_batch = tensor_batch
    trainer.seed_everything(seed)
    model = build_model(tool)
    history = trainer.fit(model, data / 'pos', data / 'neg', Path(output), device=device, seed=seed,
                          batch_size=batch_size, epochs=epochs, learning_rate=lr, tolerance=tolerance,
                          smoke_steps=smoke_steps, full_validation=True)
    return dict(tool=tool, step=history['step'], stop_reason=history['stop_reason'])


# ----------------------------------------------------------------------------- SquiggleNet

class _Rows(Dataset):
    def __init__(self, inputs, labels, rows):
        self.inputs, self.labels, self.rows = inputs, labels, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = int(self.rows[index])
        return torch.from_numpy(self.inputs[row]), int(self.labels[row])


def squigglenet_data(native, limit=None):
    """Official-normalized pA raw[1500:4500]; per split, microbe rows then host rows."""
    source = NativeSource(native)
    pa = source.windows('pa_1500_4500')
    inputs = np.zeros(pa.shape, np.float32)
    eligible = np.zeros(source.n, bool)
    rows = {}
    for split in ('train', 'dev'):
        candidates = source.rows(split)[:limit]
        candidates = candidates[source.eligible('pa_1500_4500')[candidates]]
        for start in range(0, len(candidates), 4096):
            selected = candidates[start:start + 4096]
            inputs[selected], eligible[selected] = squigglenet_inputs(np.asarray(pa[selected]))
        kept = candidates[eligible[candidates]]
        # Upstream Dataset concatenates the target (microbe) tensor, then host.
        rows[split] = np.concatenate([kept[source.labels[kept] == label] for label in (0, 1)])
        if any(not np.any(source.labels[kept] == label) for label in (0, 1)):
            raise ValueError(f'both classes required among eligible {split} rows')
    return inputs, source.labels, rows


def _full_dev_accuracy(model, loader, device):
    """Official DEV pass: train mode (BN statistics update), no gradients, mean batch accuracy."""
    accuracies = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('nonfinite dev logits')
        accuracies.append(100. * (y == logits.max(dim=1).indices).float().mean().item())
    return sum(accuracies) / len(accuracies)


def train_squigglenet(native, output, *, device='cuda', seed=SEED, epochs=20, batch_size=1000, lr=1e-3, limit=None):
    output = _new_output(output)
    inputs, labels, rows = squigglenet_data(native, limit)
    seed_everything(seed, device)
    deterministic(device)
    model = build_model('squigglenet').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)
    train_loader = DataLoader(_Rows(inputs, labels, rows['train']), batch_size=batch_size, shuffle=True, num_workers=0)
    dev_loader = DataLoader(_Rows(inputs, labels, rows['dev']), batch_size=batch_size, shuffle=True, num_workers=0)
    config = dict(tool='squigglenet', native_source=str(Path(native).resolve()), seed=seed, lr=lr, epochs=epochs,
                  batch_size=batch_size, input="pA raw[1500:4500], official normalization", labels='microbe0 host1',
                  selection='mean batch DEV accuracy, train mode, before each update (strictly greater)',
                  eligible_train=len(rows['train']), eligible_dev=len(rows['dev']), limit=limit,
                  parameters=sum(p.numel() for p in model.parameters()), runtime=runtime(device))
    write_json(output / 'config.json', config)
    model.train()
    best, selected_step, step = 0., None, 0
    started = time.monotonic()
    with (output / 'updates.jsonl').open('w') as log:
        for epoch in range(1, epochs + 1):
            for x, y in train_loader:
                step += 1
                x, y = x.to(device), y.to(torch.long).to(device)
                logits = model(x)
                loss = criterion(logits, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite training loss')
                with torch.no_grad():
                    accuracy = _full_dev_accuracy(model, dev_loader, device)
                    if accuracy > best:
                        best, selected_step = accuracy, step
                        torch.save(dict(model_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                        optimizer_state=deepcopy(optimizer.state_dict()), selection_step=step,
                                        epoch=epoch), output / 'best.pt')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                log.write(json.dumps(dict(epoch=epoch, step=step, train_loss=float(loss.detach()),
                                          dev_accuracy=accuracy, best_dev_accuracy=best)) + '\n')
            log.flush()
            print(json.dumps(dict(epoch=epoch, step=step, best_dev_accuracy=best, selection_step=selected_step)), flush=True)
    if selected_step is None:
        raise ValueError('no checkpoint selected: DEV accuracy never exceeded zero')
    torch.save(dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(), step=step), output / 'final.pt')
    metrics = dict(tool='squigglenet', epochs_completed=epochs, updates_completed=step, selection_step=selected_step,
                   best_dev_accuracy=best, elapsed_seconds=time.monotonic() - started)
    write_json(output / 'metrics.json', metrics)
    return metrics
