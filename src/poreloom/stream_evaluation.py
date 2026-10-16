"""V1: encoder-only discrimination at absolute sample counts n (plan §7), plus the window and R0 references.

Scores are p(target). At each n only reads still in the pore (raw_length >= n) count. A method with no decision
yet at n (e.g. PoreLoom before its first complete window) has NaN there.
"""
import numpy as np
import torch
from scipy.stats import rankdata

from .model import build_model
from .streaming import CausalNormalizer
from .windowing import START, WINDOW, normalize_windows

HOST_REJECT = (.90, .95, .99)


def auroc(score, is_target):
    ranks = rankdata(score); n_t = int(is_target.sum()); n_h = len(score) - n_t
    return float((ranks[is_target].sum() - n_t*(n_t + 1)/2)/(n_t*n_h))


def threshold_at_host_reject(p_target, is_target, host_reject):
    return np.quantile(p_target[~is_target], host_reject)


def frr_at_host_reject(p_target, is_target, host_reject=.95):
    """Target false-reject rate when the threshold rejects `host_reject` of hosts (reject if p_target < t)."""
    return float(np.mean(p_target[is_target] < threshold_at_host_reject(p_target, is_target, host_reject)))


def ece(p_target, is_target, bins=10):
    which = np.minimum((p_target*bins).astype(int), bins - 1)
    return float(sum(abs(p_target[which == b].mean() - is_target[which == b].mean())*np.mean(which == b)
                     for b in range(bins) if np.any(which == b)))


def summarize(p_target, is_target):
    ok = np.isfinite(p_target)
    p, y = p_target[ok], is_target[ok]
    out = dict(reads=int(len(p)), targets=int(y.sum()), auroc=auroc(p, y), ece=ece(p, y))
    out.update({f'frr_host{int(100*r)}': frr_at_host_reject(p, y, r) for r in HOST_REJECT})
    return out


def paired_frr_difference(p_new, p_old, is_target, host_reject=.95, reps=2000, seed=20261010):
    """FRR(new) - FRR(old) at the same host-reject rate; class-stratified paired bootstrap, thresholds refit."""
    rng = np.random.default_rng(seed); target, host = np.flatnonzero(is_target), np.flatnonzero(~is_target)
    point = frr_at_host_reject(p_new, is_target, host_reject) - frr_at_host_reject(p_old, is_target, host_reject)
    draws = []
    for _ in range(reps):
        rows = np.concatenate([rng.choice(target, len(target)), rng.choice(host, len(host))])
        y = is_target[rows]
        draws.append(frr_at_host_reject(p_new[rows], y, host_reject) - frr_at_host_reject(p_old[rows], y, host_reject))
    lo, hi = np.quantile(draws, [.025, .975])
    return dict(delta=float(point), ci95=[float(lo), float(hi)], host_reject=host_reject, reps=reps, seed=seed)


def _load_poreloom(checkpoint, device):
    model = build_model(max_windows=8)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model_state'])
    return model.to(device).eval()


@torch.no_grad()
def poreloom_window_scores(checkpoint, data, rows, points, device, windows=4, batch=512):
    """Existing PoreLoom: ADC windows [1500 + 1500k, 3000 + 1500k), window-local normalization; at n the last
    complete valid window's head output. [N, P] p(target)."""
    model = _load_poreloom(checkpoint, device)
    last = np.clip((np.asarray(points) - START)//WINDOW, 0, windows)
    out = np.full((len(rows), len(points)), np.nan)
    for start in range(0, len(rows), batch):
        part = rows[start:start + batch]
        raw = np.asarray(data.adc[part, START:START + windows*WINDOW], np.float64).reshape(len(part), windows, WINDOW)
        z, ok = normalize_windows(raw.reshape(-1, WINDOW))
        geometry = data.raw_length[part, None] >= START + WINDOW*np.arange(1, windows + 1)
        valid = np.logical_and.accumulate(geometry & ok.reshape(len(part), windows), 1)
        z = torch.as_tensor(z, device=device)[:, None]
        e = model.encode(z).reshape(len(part), windows, -1)
        p = torch.softmax(model.forward_features(e, torch.as_tensor(valid, device=device)), -1)[..., 1].cpu().numpy()
        for j, k in enumerate(last):
            if k: out[start:start + len(part), j] = np.where(valid[:, k - 1], p[:, k - 1], np.nan)
    return out


@torch.no_grad()
def r0_scores(stage_a_checkpoint, data, rows, points, device, skip=START, batch=256, normalizer=None):
    """R0: stage A non-causal backbone re-encodes [skip, n) as a whole, mean-pools all frames, then stage A
    projection + temporary head. No retraining. [N, P] p(target).

    Normalization: median/MAD of the whole prefix [skip, n), outliers hold the last inlier; or, with `normalizer`
    (a CausalNormalizer), the streaming blockwise one, which makes every frame except the last few (those that
    see the right zero padding at n) fixed once computed."""
    payload = torch.load(stage_a_checkpoint, map_location='cpu')
    model = build_model(max_windows=8); model.load_state_dict(payload['model_state']); model.to(device).eval()
    head = torch.nn.Linear(128, 2).to(device); head.load_state_dict(payload['temporary_head'])
    hold = CausalNormalizer(0., 1.).hold
    out = np.full((len(rows), len(points)), np.nan)
    for j, n in enumerate(points):
        for start in range(0, len(rows), batch):
            part = rows[start:start + batch]
            if normalizer is not None:
                z = normalizer.to(device)(torch.as_tensor(data.pa(part, skip, n), device=device))
            else:
                x = torch.as_tensor(data.pa(part, skip, n), device=device, dtype=torch.float64)
                med = x.median(1, keepdim=True).values; mad = (x - med).abs().median(1, keepdim=True).values
                z = hold(((x - med)/(1.4826*mad.clamp(min=1e-8))).float(), x.new_zeros(len(x)).float())
            logits = head(model.projection(model.encoder(z[:, None]).mean(-1)))
            out[start:start + len(part), j] = torch.softmax(logits, -1)[:, 1].cpu().numpy()
    return out


def native_scores(npz_path, rows, points, decision_sample):
    """Fixed-window native baseline: p(target) = sigmoid(-host margin), available once n >= decision_sample."""
    with np.load(npz_path, allow_pickle=False) as z:
        margin = z['margin'][rows]
    p = 1/(1 + np.exp(margin))
    return np.stack([p if n >= decision_sample else np.full(len(rows), np.nan) for n in points], 1)
