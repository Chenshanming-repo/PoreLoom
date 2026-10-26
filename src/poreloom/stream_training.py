"""Train streaming-prefix encoding with cross-entropy at sampled lengths per read."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import file_sha256
from torch.nn import functional as F

from .stream_encoder import build_stream_r0
from .stream_config import DEFAULT_FS, DEFAULT_MAX_SAMPLES, sampling_rate
from .stream_evaluation import frr_at_host_reject
from .training_common import balanced_indices


def normalizer_prior(data, rows, skip, max_samples, batch=2048):
    """Median over reads of each read's pA median and MAD on [skip, min(raw_length, max_samples))."""
    medians, mads = [], []
    for start in range(0, len(rows), batch):
        part = rows[start:start + batch]
        x = data.pa(part, skip, max_samples).astype(np.float64)
        x[np.arange(skip, max_samples) >= data.raw_length[part, None]] = np.nan
        med = np.nanmedian(x, 1); medians.append(med); mads.append(np.nanmedian(np.abs(x - med[:, None]), 1))
    medians, mads = np.concatenate(medians), np.concatenate(mads)
    ok = np.isfinite(medians) & (mads > 0)
    return float(np.median(medians[ok])), float(np.median(mads[ok]))


def batch_signal(data, rows, config, rng=None):
    """pA [B, max_samples] from read start and usable lengths; with rng: start jitter and amplitude augmentation."""
    width = config['max_samples']
    shift = np.zeros(len(rows), np.int64)
    if rng is not None:
        jitter = rng.random(len(rows)) < config['jitter_p']
        shift[jitter] = rng.integers(0, config['jitter_max'] + 1, int(jitter.sum()))
    shift = np.minimum(shift, data.adc.shape[1] - width)
    full = data.pa(rows, 0, data.adc.shape[1])
    x = np.take_along_axis(full, shift[:, None] + np.arange(width), 1)
    lengths = np.clip(data.raw_length[rows] - shift, 0, width)
    if rng is not None:
        gain = rng.uniform(*config['gain'], len(rows)).astype(np.float32)
        offset = rng.uniform(-config['shift_pa'], config['shift_pa'], len(rows)).astype(np.float32)
        x = np.where(np.arange(width) < lengths[:, None], x*gain[:, None] + offset[:, None], 0).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(lengths)


R0_DEFAULTS = dict(skip=1500, max_samples=DEFAULT_MAX_SAMPLES, fs=DEFAULT_FS, min_samples=400, prior_weight=400., lengths_per_read=4, epochs=20,
                   patience=3, batch_size=128, backbone_lr=1e-4, lr=3e-4, weight_decay=1e-4, clip=1., jitter_p=.5,
                   jitter_max=1500, gain=(.9, 1.1), shift_pa=3., eval_batch=128, seed=20261012,
                   dev_lengths=(432, 768, 1200, 1488, 1968, 2496, 2976, 3456, 4464, 6000), dev_target_frr=.005)


def sample_lengths(lengths, model, k, rng):
    """k absolute decision sample counts per read, skip + L' with L' a uniform multiple of 48 in
    [min_samples, usable stream]; -1 where the read is too short."""
    top = model.used(np.clip(np.asarray(lengths) - model.skip, 0, model.max_samples - model.skip))
    ok = top >= model.min_samples
    steps = np.maximum((top - model.min_samples)//model.stride + 1, 1)
    used = model.min_samples + model.stride*rng.integers(0, steps[:, None], (len(top), k))
    return np.where(ok[:, None], model.skip + used, -1)


def r0_loss(logits, labels, valid):
    """CE averaged over each read's valid lengths, then over reads with any."""
    ce = F.cross_entropy(torch.where(valid[..., None], logits, 0.).flatten(0, 1),
                         labels[:, None].expand_as(valid).flatten(), reduction='none').view(valid.shape)
    count = valid.sum(1); active = count > 0
    return ((ce*valid).sum(1)[active]/count[active]).mean()


@torch.no_grad()
def score_r0(model, data, rows, config, device, lengths):
    """p(target) [N, L] at stream lengths `lengths` (nan where the read is shorter) and the mean CE over valid entries."""
    model.eval(); out = []; ce_sum = 0.; ce_n = 0
    lengths = np.asarray(lengths)
    for start in range(0, len(rows), config['eval_batch']):
        part = rows[start:start + config['eval_batch']]
        pa, usable = batch_signal(data, part, config)
        n = np.where(data.raw_length[part, None] >= config['skip'] + lengths, config['skip'] + lengths, -1)
        logits, valid = model(pa.to(device), usable.to(device), torch.as_tensor(n, device=device))
        labels = torch.as_tensor(data.is_target[part].astype(np.int64), device=device)
        if valid.any():
            ce = F.cross_entropy(logits[valid], labels[:, None].expand_as(valid)[valid], reduction='sum')
            ce_sum += ce.item(); ce_n += int(valid.sum())
        out.append(torch.softmax(logits, -1)[..., 1].cpu().numpy())
    return np.concatenate(out), ce_sum/max(ce_n, 1)


def host_reject_at_target_frr(p, is_target, frr):
    """Host reject rate when the threshold falsely rejects `frr` of targets (reject if p < threshold)."""
    return float(np.mean(p[~is_target] < np.quantile(p[is_target], frr)))


def train_stream_r0(train, model_dev, init_checkpoint, output, device='cuda', log=print, normalizer_factory=None, **overrides):
    config = {**R0_DEFAULTS, **overrides}
    config['fs'] = sampling_rate(config['fs'])
    for name, data in (('TRAIN', train), ('DEV', model_dev)):
        if data.adc.shape[1] < config['max_samples']:
            raise ValueError(f"{name} prefix cache has {data.adc.shape[1]} samples; decision endpoint requires {config['max_samples']}")
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config['seed']); rng = np.random.default_rng(config['seed'])
    median, mad = normalizer_prior(train, np.arange(len(train)), config['skip'], config['max_samples'])
    model = build_stream_r0(init_checkpoint, median=median, mad=mad, prior_weight=config['prior_weight'],
                            skip=config['skip'], max_samples=config['max_samples'], min_samples=config['min_samples'],
                            time_features=True, fs=config['fs'])
    if config['max_samples'] != R0_DEFAULTS['max_samples'] and 'dev_lengths' not in overrides:
        top = model.used(config['max_samples'] - config['skip'])
        base = [n for n in R0_DEFAULTS['dev_lengths'] if model.min_samples <= n <= top]
        step = model.used(1488)
        extra = range(R0_DEFAULTS['dev_lengths'][-1] + step, top, step)
        config['dev_lengths'] = tuple(sorted(set([*base, *extra, top])))
    if normalizer_factory is not None:
        if not config.get('normalization'):
            raise ValueError('custom normalizer requires a recorded normalization setting')
        model.normalizer = normalizer_factory(model.normalizer)
    model = model.to(device)
    backbone = list(model.backbone.parameters()); ids = {id(p) for p in backbone}
    rest = [p for p in model.parameters() if id(p) not in ids]
    opt = torch.optim.AdamW([dict(params=backbone, lr=config['backbone_lr']), dict(params=rest, lr=config['lr'])],
                            weight_decay=config['weight_decay'])
    labels = torch.as_tensor(train.is_target.astype(np.int64))
    dev_rows = np.arange(len(model_dev))
    provenance = dict(init_checkpoint=str(init_checkpoint), init_checkpoint_sha256=file_sha256(init_checkpoint),
                      normalizer_prior=dict(median_pa=median, mad_pa=mad, weight=config['prior_weight']),
                      model='StreamR0 with time_features', config=config)
    (output/'training_registration.json').write_text(json.dumps(provenance, indent=2) + '\n')

    def evaluate(epoch, train_ce, seconds):
        p, dev_ce = score_r0(model, model_dev, dev_rows, config, device, config['dev_lengths'])
        host, frr95 = {}, {}
        for j, length in enumerate(config['dev_lengths']):
            alive = np.isfinite(p[:, j]); y = model_dev.is_target[alive].astype(bool)
            host[str(length)] = host_reject_at_target_frr(p[alive, j], y, config['dev_target_frr'])
            frr95[str(length)] = frr_at_host_reject(p[alive, j], y)
        return dict(epoch=epoch, train_ce=train_ce, dev_ce=dev_ce, dev_host_reject_at_target_frr=host,
                    dev_frr_host95=frr95, seconds=seconds)

    history = [evaluate(0, None, 0.)]; log(json.dumps(history[-1]))
    best, stale = history[0]['dev_ce'], 0
    torch.save(dict(model_state=model.state_dict(), epoch=0, dev_ce=best, **provenance), output/'best.pt')
    steps = len(train)//config['batch_size']
    for epoch in range(1, config['epochs'] + 1):
        model.train(); started = time.time(); total = 0.
        for _ in range(steps):
            rows = balanced_indices(train.is_target, config['batch_size'], rng)
            pa, usable = batch_signal(train, rows, config, rng)
            n = sample_lengths(usable.numpy(), model, config['lengths_per_read'], rng)
            logits, valid = model(pa.to(device), usable.to(device), torch.as_tensor(n, device=device))
            loss = r0_loss(logits, labels[rows].to(device), valid)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip']); opt.step(); total += loss.item()
        history.append(evaluate(epoch, total/steps, round(time.time() - started, 1)))
        log(json.dumps(history[-1]))
        (output/'training_history.json').write_text(json.dumps(history, indent=2) + '\n')
        if history[-1]['dev_ce'] < best:
            best, stale = history[-1]['dev_ce'], 0
            torch.save(dict(model_state=model.state_dict(), epoch=epoch, dev_ce=best, **provenance), output/'best.pt')
        else:
            stale += 1
            if stale >= config['patience']: break
    return output/'best.pt'


def load_stream_r0(path, device='cpu', normalizer_factory=None):
    payload = torch.load(path, map_location='cpu')
    config, prior = payload['config'], payload['normalizer_prior']
    if config.get('normalization') and normalizer_factory is None:
        raise ValueError('custom normalization checkpoint requires its normalizer_factory')
    model = build_stream_r0(median=prior['median_pa'], mad=prior['mad_pa'], prior_weight=prior['weight'],
                            skip=config['skip'], max_samples=config['max_samples'], min_samples=config['min_samples'],
                            time_features=True, fs=config.get('fs', DEFAULT_FS))
    if normalizer_factory is not None:
        model.normalizer = normalizer_factory(model.normalizer)
    model.load_state_dict(payload['model_state'])
    return model.to(device).eval(), payload
