"""Factorized action-cost heads over streaming-prefix scores, trained backward over time buckets.

A look is a chunk arrival at which R0 can act (L' >= min_samples). Its input comes from the frozen encoder; the
labels of every action come from chunk_replay.replay, so training and evaluation share one outcome model.

Buckets split the used stream length L' into equal ranges (`bucket` samples each) plus one forced head (the first
arrival holding >= max_samples, accept or reject only). With chunks of at least `bucket` + 47 samples, consecutive
looks of a read always fall in strictly later buckets, so heads are trained from the last bucket back to the first:
the label of DEFER at a look is the outcome of the already trained later heads. The factorized costs are
u = host pore seconds / reference and r = target command reject; the action score is u + beta * r.
"""
import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .chunk_replay import NO_DECISION, replay
from .contracts import Action
from .stream_config import DEFAULT_FS, DEFAULT_MAX_SAMPLES, bucket_count, sampling_rate

ACCEPT, REJECT, DEFER = int(Action.ACCEPT), int(Action.REJECT), int(Action.DEFER)


@dataclass
class Looks:
    """Per-read decision opportunities; rows are reads (or read copies under different chunk phases).

    t [N,J]: arrival time in s (nan: no update); valid [N,J]: R0 can act; forced [N,J]; bucket [N,J] (-1: cannot act,
    n_buckets: forced); x [N,J,D] inputs; t_end [N] natural end in s; is_target [N]."""
    t: np.ndarray
    valid: np.ndarray
    forced: np.ndarray
    bucket: np.ndarray
    x: np.ndarray
    t_end: np.ndarray
    is_target: np.ndarray

    def __len__(self): return len(self.t)

    def subset(self, rows):
        return Looks(*(getattr(self, f)[rows] for f in ('t', 'valid', 'forced', 'bucket', 'x', 't_end', 'is_target')))


def look_grid(m, skip=1500, max_samples=DEFAULT_MAX_SAMPLES, min_samples=432, stride=48, fs=DEFAULT_FS, bucket=768, n_buckets=None):
    """Arrival sample counts m [N,J] (nan past the last) -> (n, t, valid, forced, bucket index). Same conventions as
    the threshold replay: an arrival after skip is an update; it can act once L' >= min_samples; forced once it
    holds >= max_samples."""
    fs = sampling_rate(fs)
    if n_buckets is None: n_buckets = bucket_count(max_samples, skip, min_samples, bucket)
    n = np.where(np.isfinite(m), np.floor(m), -1).astype(np.int64)
    update = n > skip
    t = np.where(update, m/fs, np.nan)
    forced = update & (n >= max_samples)
    used = (np.clip(n, skip, max_samples) - skip)//stride*stride
    valid = update & (used >= min_samples)
    index = np.where(forced, n_buckets, np.where(valid, np.minimum(n_buckets - 1, (used - min_samples)//bucket), -1))
    return n, t, valid, forced, index


def next_look_used(used, min_samples=432, bucket=768, stride=48, max_used=6000):
    """Smallest used stream length L' at which a read may act again after acting at `used`: one look per bucket
    (backward training never saw two looks of a read in one bucket, which chunks < `bucket` samples would cause),
    never past the forced look at max_used. Chunks >= `bucket` samples are unaffected."""
    used = np.asarray(used, np.int64)
    nxt = min_samples + bucket*((used - min_samples)//bucket + 1)
    return np.minimum(max_used, np.maximum(used + stride, nxt))


def one_look_per_bucket(valid, bucket):
    """valid [N,J] restricted to the first look of each bucket in every read: the offline form of next_look_used."""
    keep = np.zeros_like(valid)
    last = np.full(len(valid), -1)
    for j in range(valid.shape[1]):
        keep[:, j] = valid[:, j] & (bucket[:, j] > last)
        last = np.where(keep[:, j], bucket[:, j], last)
    return keep


def trajectory(ell, valid):
    """Per valid look: logit margin at the previous valid look (itself at the first), running minimum, look count."""
    prev, low, count = (np.zeros(ell.shape, np.float32) for _ in range(3))
    last = np.full(len(ell), np.nan); running = np.full(len(ell), np.inf); seen = np.zeros(len(ell))
    for j in range(ell.shape[1]):
        v = valid[:, j]
        running = np.where(v, np.minimum(running, ell[:, j]), running); seen = seen + v
        prev[:, j] = np.where(v, np.where(np.isnan(last), ell[:, j], last), 0)
        low[:, j] = np.where(v, running, 0); count[:, j] = np.where(v, seen, 0)
        last = np.where(v, ell[:, j], last)
    return prev, low, count


def look_inputs(ell, n, valid, h=None, max_samples=DEFAULT_MAX_SAMPLES, fs=DEFAULT_FS):
    """Controller inputs [N,J,D]: (h,) ell, n/max_samples, log1p(n/fs), previous ell, minimum ell, look count;
    zeros where the look cannot act."""
    fs = sampling_rate(fs)
    prev, low, count = trajectory(ell, valid)
    nf = np.minimum(n, max_samples).astype(np.float32)
    parts = [np.where(valid, ell, 0), np.where(valid, nf/max_samples, 0), np.where(valid, np.log1p(nf/fs), 0), prev, low, count]
    x = np.stack(parts, -1).astype(np.float32)
    if h is not None: x = np.concatenate([np.where(valid[..., None], h, 0).astype(np.float32), x], -1)
    return x


@torch.no_grad()
def encode_looks(model, data, n, device, batch=128):
    """Frozen StreamR0 at absolute sample counts n [N,P] (-1: none) -> (projection output h [N,P,C], logit margin
    ell [N,P]); nan where R0 has no decision."""
    captured = {}
    hook = model.projection.register_forward_hook(lambda module, args, out: captured.__setitem__('h', out))
    try:
        h = np.full(n.shape + (model.head.in_features,), np.nan, np.float32); ell = np.full(n.shape, np.nan, np.float32)
        for start in range(0, len(n), batch):
            rows = np.arange(start, min(start + batch, len(n)))
            pa = torch.as_tensor(data.pa(rows, 0, model.max_samples), device=device)
            logits, valid = model(pa, torch.as_tensor(data.raw_length[rows], device=device), torch.as_tensor(n[rows], device=device))
            ok = valid.cpu().numpy()
            h[rows] = np.where(ok[..., None], captured['h'].float().cpu().numpy(), np.nan)
            ell[rows] = (logits[..., 1] - logits[..., 0]).float().cpu().numpy()
    finally:
        hook.remove()
    return h, ell


def _head(input_dim, actions, hidden):
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, 2*actions))


class BucketController(nn.Module):
    """One factorized head per time bucket plus a forced head; inputs are standardized with stored statistics."""
    def __init__(self, input_dim, n_buckets=8, hidden=64, beta=1., mean=None, std=None):
        super().__init__()
        self.n_buckets, self.beta = int(n_buckets), float(beta)
        self.heads = nn.ModuleList([_head(input_dim, 3, hidden) for _ in range(n_buckets)] + [_head(input_dim, 2, hidden)])
        self.register_buffer('mean', torch.zeros(input_dim) if mean is None else torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.ones(input_dim) if std is None else torch.as_tensor(std, dtype=torch.float32))

    def raw(self, x, b):
        """(u logits, r logits), each [..., A]; ACCEPT never rejects, so its r logit is ignored."""
        out = self.heads[b]((x - self.mean)/self.std)
        a = out.shape[-1]//2
        return out[..., :a], out[..., a:]

    def scores(self, x, b):
        u, r = self.raw(x, b)
        r = torch.sigmoid(r)*torch.tensor([0.] + [1.]*(r.shape[-1] - 1), device=r.device)
        return F.softplus(u) + self.beta*r

    @torch.no_grad()
    def actions(self, x, b):
        return self.scores(x, b).argmin(-1)


def policy_actions(controller, looks, device, buckets=None, out=None):
    """Actions [N,J] of the controller at every look (NO_DECISION where it cannot act); only `buckets` are filled
    into `out` when given."""
    out = np.full(looks.t.shape, NO_DECISION) if out is None else out
    for b in range(controller.n_buckets + 1) if buckets is None else buckets:
        rows, cols = np.nonzero(looks.bucket == b)
        if not len(rows): continue
        x = torch.as_tensor(looks.x[rows, cols], device=device)
        out[rows, cols] = controller.actions(x, b).cpu().numpy()
    return out


def outcome(looks, actions, service, eject):
    return replay(looks.t + service, actions, looks.t, service, looks.t_end, eject)


def node_labels(looks, policy, b, service, eject, ref, forced=False):
    """Training nodes of bucket b (looks where the read is still in the pore when the decision completes) and the
    components of every legal action there: (rows, cols, u [K,A], r [K,A]). DEFER follows `policy` afterwards; the
    forced head has ACCEPT and REJECT only."""
    alive = looks.t + service < looks.t_end[:, None]
    rows, cols = np.nonzero((looks.bucket == b) & alive)
    if len(np.unique(rows)) != len(rows): raise ValueError('two looks of one read share a bucket (chunks too short)')
    sub = looks.subset(rows)
    base = policy[rows].copy()
    later = np.arange(base.shape[1]) > cols[:, None]
    base[~later] = NO_DECISION   # every earlier look deferred to reach the node
    legal = (ACCEPT, REJECT) if forced else (ACCEPT, REJECT, DEFER)
    u, r = [], []
    for a in legal:
        act = base.copy(); act[np.arange(len(rows)), cols] = a
        o = outcome(sub, act, service, eject)
        u.append(np.where(sub.is_target, 0., o['pore_seconds']/ref)); r.append((sub.is_target & o['command_reject']).astype(float))
    return rows, cols, np.stack(u, 1).astype(np.float32), np.stack(r, 1).astype(np.float32)


def component_loss(controller, x, u, r, w, b):
    u_raw, r_raw = controller.raw(x, b)
    loss = (F.softplus(u_raw) - u).square()
    risk = F.binary_cross_entropy_with_logits(r_raw, r, reduction='none')
    loss = loss + risk*torch.tensor([0.] + [1.]*(r.shape[-1] - 1), device=r.device)
    return (loss.mean(-1)*w).sum()/w.sum()


def class_weights(is_target):
    q = float(np.mean(is_target))
    return np.where(is_target, .5/max(q, 1e-12), .5/max(1 - q, 1e-12)).astype(np.float32)


def train_backward(fit, sel, beta, ref, service, eject, device, n_buckets=8, hidden=64, lr=1e-3, weight_decay=1e-4,
                   batch=512, epochs=30, patience=3, clip=1., seed=20261013, log=None):
    """One controller for a fixed beta: the forced head, then buckets from last to first, each early-stopped on the
    SEL component loss. Inputs are standardized with FIT statistics over looks that can act.
    Returns (controller, FIT policy, SEL policy, history)."""
    torch.manual_seed(seed); generator = torch.Generator(device='cpu').manual_seed(seed)
    acting = fit.x[fit.bucket >= 0]
    mean, std = acting.mean(0), acting.std(0)
    std = np.where(std > 1e-6, std, 1.)
    controller = BucketController(fit.x.shape[-1], n_buckets, hidden, beta, mean, std).to(device)
    policies = [np.full(fit.t.shape, NO_DECISION), np.full(sel.t.shape, NO_DECISION)]
    history = []
    for b in [n_buckets] + list(range(n_buckets - 1, -1, -1)):
        sets = []
        for looks, policy in zip((fit, sel), policies):
            rows, cols, u, r = node_labels(looks, policy, b, service, eject, ref, forced=b == n_buckets)
            if not len(rows): raise ValueError(f'no training nodes in bucket {b}')
            w = class_weights(looks.is_target[rows])
            sets.append([torch.as_tensor(v, device=device) for v in (looks.x[rows, cols], u, r, w)])
        head = controller.heads[b]
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
        best, best_state, stale = np.inf, copy.deepcopy(head.state_dict()), 0
        x, u, r, w = sets[0]
        for epoch in range(epochs):
            head.train()
            for idx in torch.randperm(len(x), generator=generator).to(device).split(batch):
                loss = component_loss(controller, x[idx], u[idx], r[idx], w[idx], b)
                optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), clip); optimizer.step()
            head.eval()
            with torch.no_grad():
                val = float(component_loss(controller, *sets[1], b))
            history.append(dict(bucket=b, epoch=epoch + 1, nodes_fit=len(x), nodes_sel=len(sets[1][0]), sel_loss=val))
            if val < best: best, best_state, stale = val, copy.deepcopy(head.state_dict()), 0
            else: stale += 1
            if stale >= patience: break
        head.load_state_dict(best_state); head.eval()
        for looks, policy in zip((fit, sel), policies): policy_actions(controller, looks, device, [b], policy)
        if log: log('bucket', b, 'nodes', len(x), 'sel loss', round(best, 6))
    return controller, policies[0], policies[1], history
