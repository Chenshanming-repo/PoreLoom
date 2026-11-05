"""A trained PoreLoom bundle (``<run>/poreloom_sh_cp.pt`` from ``python -m poreloom.method``) as a live policy.

The computation is the one used for the paper's Icarust runs (experiments/20261014_final/icarust_live.py,
``PoreLoomPolicy``): the StreamR0 encoder keeps per-read state in a device-resident bank and is advanced by each
newly arrived chunk; the C-p head looks once per decision bucket and returns accept, reject or defer, with a forced
accept/reject at the decision endpoint. A read whose input is unusable is accepted (sequenced normally).
"""
from collections import deque
from pathlib import Path
import time

import numpy as np


def picoamperes(adc, c):
    return ((np.asarray(adc, np.float64) + c['offset'])*c['range']/c['digitisation']).astype(np.float32)


def resolve_encoder(bundle, bundle_path, encoder=None):
    """Explicit path, else the path stored in the bundle, else <bundle dir>/stream/best.pt."""
    if encoder is not None:
        return Path(encoder)
    stored = bundle.get('encoder')
    if stored and Path(stored).is_file():
        return Path(stored)
    local = Path(bundle_path).parent/'stream'/'best.pt'
    if local.is_file():
        return local
    raise FileNotFoundError(f'encoder checkpoint not found (bundle stores {stored!r}); pass encoder=')


class PoreLoomLivePolicy:
    """calibration: the int16 ADC contract of the buffered samples (see signal.adc_mapping)."""
    name = 'PoreLoom'

    def __init__(self, bundle_path, device, calibration, capacity=2048, encoder=None):
        import torch
        from ..data import file_sha256
        from ..method.head import look_config
        from ..stream_bank import StreamR0Bank
        from ..stream_config import sampling_rate
        from ..stream_controller import BucketController
        from ..stream_training import load_stream_r0
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        bundle_path = Path(bundle_path)
        bundle = torch.load(bundle_path, map_location='cpu')
        self.torch, self.device, self.cal = torch, device, dict(calibration)
        encoder_path = resolve_encoder(bundle, bundle_path, encoder)
        self.model, _ = load_stream_r0(encoder_path, device); self.model.eval()
        self.skip, self.max_samples = self.model.skip, self.model.max_samples
        self.fs = sampling_rate(bundle['fs'])
        h = bundle['head']
        cfg = look_config(h.get('look_config') or dict(budget=self.max_samples, skip=self.skip,
                          stride=self.model.stride, min_samples=self.model.min_samples, n_buckets=h['n_buckets']))
        if (cfg['budget'] != self.max_samples or cfg['skip'] != self.skip or cfg['stride'] != self.model.stride
                or cfg['min_samples'] != self.model.min_samples or cfg['n_buckets'] != h['n_buckets']
                or cfg['feature_fs'] != self.model.fs or bundle.get('max_samples', self.max_samples) != self.max_samples):
            raise ValueError('bundle head configuration differs from the encoder sampling features or decision endpoint')
        if h['input_dim'] != 6:
            raise ValueError(f"expected a 6-feature C-p head, got input_dim={h['input_dim']}")
        self.feature_fs, self.feature_max_samples = cfg['feature_fs'], cfg['feature_max_samples']
        self.controller = BucketController(h['input_dim'], h['n_buckets'], h['hidden'], bundle['beta'])
        self.controller.load_state_dict(bundle['head_state']); self.controller.to(device).eval()
        self.n_buckets, self.bucket, self.min_samples = h['n_buckets'], cfg['bucket_samples'], self.model.min_samples
        self.bank = StreamR0Bank(self.model, capacity, device)
        cap = self.bank.capacity
        self.last, self.low, self.count = np.full(cap, np.nan), np.full(cap, np.inf), np.zeros(cap)
        self.last_bucket = np.full(cap, -1)
        self.pending = deque()
        self.pins = dict(bundle=dict(path=str(bundle_path.resolve()), sha256=file_sha256(bundle_path)),
                         encoder=dict(path=str(encoder_path.resolve()), sha256=file_sha256(encoder_path)),
                         beta=bundle['beta'], fs=self.fs, max_samples=self.max_samples,
                         interval_trained=bundle.get('interval_trained'))
        self.stats = dict(steps=0, reads_stepped=0, step_seconds=0., max_step_seconds=0., fallbacks=0, peak_slots=0)

    def needed(self, s):
        from ..stream_controller import next_look_used
        slot = s.get('h')
        if slot is None: return self.skip + self.min_samples
        if self.bank.failed[slot] or self.bank.received[slot] >= self.max_samples: return None
        return int(self.skip + next_look_used(self.bank.used[slot], self.min_samples, self.bucket, self.model.stride,
                                              max_used=self.max_samples - self.skip))

    def release(self, s):
        slot = s.pop('h', None)
        if slot is not None: self.pending.append(slot)

    def decide(self, jobs):
        from ..contracts import Action
        torch, bank = self.torch, self.bank
        while self.pending: bank.release(self.pending.popleft())
        slots = []
        for s in jobs:
            slot = s.get('h')
            if slot is None:
                slot = s['h'] = bank.allocate()
                self.last[slot], self.low[slot], self.count[slot] = np.nan, np.inf, 0
                self.last_bucket[slot] = -1
            slots.append(slot)
        slots = np.array(slots, np.int64)
        arrival = np.array([min(int(s['received']), len(s['buf'])) for s in jobs])   # snapshot: buffers may grow meanwhile
        starts = bank.received[slots].copy()
        signals = [None if min(a, self.max_samples) <= r else picoamperes(s['buf'][r:min(a, self.max_samples)], self.cal)
                   for s, r, a in zip(jobs, starts, arrival)]
        t0 = time.perf_counter()
        with torch.inference_mode():
            bank.step(slots, starts, signals)
            logits = bank.readout(slots)
            ell = (logits[:, 1] - logits[:, 0]).float().cpu().numpy()
            used, failed = bank.used[slots], bank.failed[slots]
            n = np.minimum(arrival, self.max_samples).astype(np.float64)
            valid = np.isfinite(ell) & (used >= self.min_samples)
            forced = n >= self.max_samples
            bucket = np.where(forced, self.n_buckets, np.minimum(self.n_buckets - 1, (used - self.min_samples)//self.bucket))
            valid &= bucket > self.last_bucket[slots]
            act = np.full(len(jobs), int(Action.DEFER))
            rows = np.flatnonzero(valid)
            if len(rows):
                sl = slots[rows]; e = ell[rows]
                prev = np.where(np.isnan(self.last[sl]), e, self.last[sl])
                self.low[sl] = np.minimum(self.low[sl], e); self.count[sl] += 1; self.last[sl] = e
                self.last_bucket[sl] = bucket[rows]
                nf = np.minimum(n[rows], self.feature_max_samples)
                x = torch.as_tensor(np.stack([e, nf/self.feature_max_samples, np.log1p(nf/self.feature_fs), prev, self.low[sl], self.count[sl]], -1)
                                    .astype(np.float32), device=logits.device)
                for b in np.unique(bucket[rows]):
                    k = np.flatnonzero(bucket[rows] == b)
                    act[rows[k]] = self.controller.actions(x[torch.as_tensor(k, device=x.device)], int(b)).cpu().numpy()
        dt = time.perf_counter() - t0
        st = self.stats
        st['steps'] += 1; st['reads_stepped'] += len(jobs); st['step_seconds'] += dt
        st['max_step_seconds'] = max(st['max_step_seconds'], dt); st['peak_slots'] = max(st['peak_slots'], bank.capacity - len(bank.free))
        out = []
        for s, a, e, u, bad in zip(jobs, act, ell, used, failed):
            s['k'] = s.get('k', 0) + 1
            if bad:
                st['fallbacks'] += 1; action, info = 'accept', dict(reason='input_fallback')
            else:
                action = 'reject' if a == Action.REJECT else 'accept' if a == Action.ACCEPT else 'defer'
                info = dict(margin=float(e), used=int(u))
            if action != 'defer': self.release(s)
            out.append((s, action, info))
        return out
