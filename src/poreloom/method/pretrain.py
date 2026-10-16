"""PoreLoom window encoder (stage A), the first training stage of the method.

Stage A  PoreLoom window encoder = CNN (method.backbone, ReadCurrent architecture) + projection (poreloom.model) with a
         temporary linear head, from a seeded random initialization, trained on one random valid 1500-sample window per
         read and epoch (windows from raw[1500:], each MAD-normalized): AdamW lr 1e-3 for every weight, weight decay
         1e-4, batch 128, 30 epochs; the epoch with the lowest DEV-A read-normalized window cross entropy is kept
         (stage_A.pt: model_state, temporary_head).
This is the recipe chosen by the 2026-10-14 ablation (variant a1, experiments/20261014_method_ablation): an earlier
stage 0 window-classifier pretraining with a short stage A (lr 1e-4, 5 epochs) was 2.85 pp ACC worse on the sputum TEST.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import mad_normalize

STAGEA = dict(seed=20261009, window_start=1500, window=1500, max_windows=8, epochs=30, batch_size=128, encoder_lr=1e-3,
              new_lr=1e-3, weight_decay=1e-4, clip=1.)
HOST, TARGET = 0, 1                       # network class indices (class_order host, target)


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def encoder_windows(data, start=1500, window=1500, max_windows=8, path=None):
    """MAD-normalized windows [N, K, window] (memmap at `path` if given) and validity: window k is valid when the read
    and the prefix cache hold it, its MAD is positive, and every earlier window is valid."""
    n = len(data)
    x = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(n, max_windows, window)) if path else \
        np.zeros((n, max_windows, window), np.float32)
    valid = np.zeros((n, max_windows), bool)
    held = np.minimum(np.asarray(data.raw_length), data.adc.shape[1])
    for k in range(max_windows):
        lo, hi = start + k*window, start + (k + 1)*window
        rows = np.flatnonzero(held >= hi)
        for s in range(0, len(rows), 4096):
            part = rows[s:s + 4096]
            z, ok = mad_normalize(np.asarray(data.adc[part, lo:hi]))
            x[part[ok], k] = z[ok]; valid[part[ok], k] = True
    valid = np.logical_and.accumulate(valid, axis=1)
    if path: x.flush()
    return x, valid


def train_window_encoder(train, dev_a, output, device='cuda', log=print, **overrides):
    """Stage A from random initialization. Returns the path of stage_A.pt (PoreLoom model_state and temporary_head)."""
    from ..model import build_model
    cfg = {**STAGEA, **overrides}
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    xt, vt = encoder_windows(train, cfg['window_start'], cfg['window'], cfg['max_windows'], output/'train_windows.npy')
    xd, vd = encoder_windows(dev_a, cfg['window_start'], cfg['window'], cfg['max_windows'])
    yt = np.where(train.is_target, TARGET, HOST); yd = np.where(dev_a.is_target, TARGET, HOST)
    model = build_model(None, max_windows=cfg['max_windows']).to(device); temporary = nn.Linear(128, 2).to(device)
    params = list(model.encoder.parameters()) + list(model.projection.parameters()) + list(temporary.parameters())
    opt = torch.optim.AdamW([dict(params=model.encoder.parameters(), lr=cfg['encoder_lr']),
                             dict(params=list(model.projection.parameters()) + list(temporary.parameters()), lr=cfg['new_lr'])],
                            weight_decay=cfg['weight_decay'])
    rng = np.random.default_rng(cfg['seed'])
    active, dev_active = np.flatnonzero(vt.any(1)), np.flatnonzero(vd.any(1))
    _write(output/'stageA_config.json', dict(stage='A: window encoder', config=cfg, train_active=len(active),
                                             dev_a_active=len(dev_active), valid_windows_train=vt.sum(0).tolist(), init='random'))

    @torch.inference_mode()
    def validate():
        model.eval(); temporary.eval()
        total = np.zeros(len(vd)); counts = vd.sum(1); pairs = np.argwhere(vd)
        for s in range(0, len(pairs), 128):
            ids = pairs[s:s + 128]
            x = torch.as_tensor(xd[ids[:, 0], ids[:, 1]], device=device)[:, None]
            loss = F.cross_entropy(temporary(model.encode(x)), torch.as_tensor(yd[ids[:, 0]], device=device), reduction='none')
            np.add.at(total, ids[:, 0], loss.cpu().numpy())
        return float((total[dev_active]/counts[dev_active]).mean())

    best, history, started = float('inf'), [], time.time()
    for epoch in range(1, cfg['epochs'] + 1):
        model.train(); temporary.train(); order = rng.permutation(active); total = 0.
        for s in range(0, len(order), cfg['batch_size']):
            ids = order[s:s + cfg['batch_size']]
            ks = np.array([rng.integers(vt[i].sum()) for i in ids])
            x = torch.as_tensor(np.asarray(xt[ids, ks]), device=device)[:, None]
            opt.zero_grad()
            loss = F.cross_entropy(temporary(model.encode(x)), torch.as_tensor(yt[ids], device=device))
            loss.backward(); nn.utils.clip_grad_norm_(params, cfg['clip']); opt.step()
            total += loss.item() * len(ids)
        val = validate()
        entry = dict(epoch=epoch, train_loss=total/len(active), dev_loss=val, elapsed_seconds=time.time() - started)
        history.append(entry); log(json.dumps(dict(stage='A', **entry)))
        if val < best:
            best = val
            torch.save(dict(model_state=model.state_dict(), temporary_head=temporary.state_dict(), epoch=epoch, dev_loss=val,
                            config=cfg), output/'stage_A.pt')
    _write(output/'stageA_history.json', history)
    (output/'train_windows.npy').unlink()
    return output/'stage_A.pt'
