#!/usr/bin/env python3
"""One live Icarust run for the final closed-loop comparison (sim g1 models, all trained from scratch on sim g1 TRAIN).

Unchanged simloop client and simulator launch (experiments/20261011_simloop/closed_loop.py: 512 channels, R10 4 kHz
Icarust with the TEST-role genomic references, unblock 0.1 s, stop_further_data on accept). Strategies:
  No-AS        accept every read
  PoreLoom     poreloom.method bundle (StreamR0 encoder in a device-resident state bank + C-p head at the locked beta);
               one look per decision bucket, with sampling features and endpoint loaded from the trained bundle
  readcurrent / nanodeep / squigglenet
               poreloom.method.baselines single native window (NanoDeep raw[1000:4000], others raw[1500:4500]); one
               decision when the window is complete; reject when host margin >= the locked DEV-B max-ACC threshold
               (NanoDeep: authors' training recipe)
Model paths: icarust_models.json next to this file. Icarust ADC is mapped to the sim g1 training ADC contract
(digitisation 8192, range 1536.598389 pA, offset = mean sim g1 DEV offset) identically for every strategy.
"""
from collections import deque
from pathlib import Path
import importlib.util
import json
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO/'src'))
_spec = importlib.util.spec_from_file_location('simloop_closed_loop', REPO/'experiments/20261011_simloop/closed_loop.py')
cl = importlib.util.module_from_spec(_spec); sys.modules[_spec.name] = cl; _spec.loader.exec_module(cl)

MODELS = json.loads((HERE/'icarust_models.json').read_text())
cl.TRAIN_CAL = dict(MODELS['calibration'])
TOOLS = ('readcurrent', 'nanodeep', 'squigglenet')
NAMES = ['No-AS', 'PoreLoom', *TOOLS, 'PoreLoom-head0.2']
# --break-ms (chunk time of the simulated sequencer; ablation 2026-10-15) is taken out before simloop parses argv;
# default 400 = the simloop profile unchanged. PoreLoom-head0.2 = the same encoder with the C-p head trained at 0.2 s.
BREAK_MS = 400
if '--break-ms' in sys.argv:
    i = sys.argv.index('--break-ms'); BREAK_MS = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
_profile_text = cl.profile_text


def _profile(*args):
    text = _profile_text(*args)
    assert text.count('break_read_ms = 400') == 1
    return text.replace('break_read_ms = 400', f'break_read_ms = {BREAK_MS}')


cl.profile_text = _profile


def pa(adc, c):
    return ((np.asarray(adc, np.float64) + c['offset'])*c['range']/c['digitisation']).astype(np.float32)


class PoreLoomPolicy(cl.Strategy):
    name = 'PoreLoom'

    def __init__(self, device, calibration, capacity=2048, bundle_path=None):
        import torch
        from poreloom.method.head import look_config
        from poreloom.stream_bank import StreamR0Bank
        from poreloom.stream_config import sampling_rate
        from poreloom.stream_controller import BucketController
        from poreloom.stream_training import load_stream_r0
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        run = Path(MODELS['PoreLoom'])
        bundle_path = Path(bundle_path or run/'poreloom_sh_cp.pt')
        bundle = torch.load(bundle_path, map_location='cpu')
        self.torch, self.device, self.cal = torch, device, calibration
        encoder_path = Path(bundle.get('encoder', run/'stream'/'best.pt'))
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
        self.feature_fs, self.feature_max_samples = cfg['feature_fs'], cfg['feature_max_samples']
        self.controller = BucketController(h['input_dim'], h['n_buckets'], h['hidden'], bundle['beta'])
        self.controller.load_state_dict(bundle['head_state']); self.controller.to(device).eval()
        assert h['input_dim'] == 6
        self.n_buckets, self.bucket, self.min_samples = h['n_buckets'], cfg['bucket_samples'], self.model.min_samples
        self.bank = StreamR0Bank(self.model, capacity, device)
        cap = self.bank.capacity
        self.last, self.low, self.count = np.full(cap, np.nan), np.full(cap, np.inf), np.zeros(cap)
        self.last_bucket = np.full(cap, -1)
        self.pending = deque()
        self.pins = dict(bundle=cl.pin(bundle_path), interval_trained=bundle.get('interval_trained'), encoder=cl.pin(encoder_path), beta=bundle['beta'],
                         fs=self.fs, max_samples=self.max_samples,
                         wrapper=cl.pin(__file__))
        self.stats = dict(steps=0, reads_stepped=0, step_seconds=0., max_step_seconds=0., fallbacks=0, peak_slots=0)

    def needed(self, s):
        from poreloom.stream_controller import next_look_used
        slot = s.get('h')
        if slot is None: return self.skip + self.min_samples
        if self.bank.failed[slot] or self.bank.received[slot] >= self.max_samples: return None
        return int(self.skip + next_look_used(self.bank.used[slot], self.min_samples, self.bucket, self.model.stride,
                                              max_used=self.max_samples - self.skip))

    def release(self, s):
        slot = s.pop('h', None)
        if slot is not None: self.pending.append(slot)

    def decide(self, jobs):
        from poreloom.contracts import Action
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
        signals = [None if min(a, self.max_samples) <= r else pa(s['buf'][r:min(a, self.max_samples)], self.cal)
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


class NativePolicy(cl.Strategy):
    def __init__(self, tool, device, calibration):
        import torch
        from poreloom.method import native
        from poreloom.baselines.preprocessing import CONTRACTS
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        run = Path(MODELS[tool])
        sel = json.loads((run/'selection.json').read_text())
        self.name, self.tool, self.native, self.device, self.cal = tool, tool, native, device, calibration
        self.model = native._load(tool, run/'fit'/'best.pt', device)
        self.threshold = sel['rules']['max_acc']['threshold']          # float, or 'all_keep'
        self.contract = CONTRACTS[tool]; self.max_samples = self.contract['raw_end']
        self.pins = dict(checkpoint=cl.pin(run/'fit'/'best.pt'), selection=cl.pin(run/'selection.json'), threshold=self.threshold,
                         wrapper=cl.pin(__file__))
        self.stats = dict(steps=0, reads=0, fallbacks=0)

    def needed(self, s):
        return self.max_samples if s.get('k', 0) == 0 else None

    def margins(self, jobs):
        import torch
        from poreloom.prefix_data import PrefixSet
        n = len(jobs)
        raw = np.stack([s['buf'][:self.max_samples] for s in jobs])
        cal = np.tile([self.cal['offset'], self.cal['range'], self.cal['digitisation']], (n, 1))
        data = PrefixSet(raw, np.zeros(n, bool), np.full(n, self.max_samples), cal, np.array([f'live{i}' for i in range(n)]),
                         np.full(n, 'live'))
        prepared = self.native.prepare(self.tool, data)
        ok = np.flatnonzero(prepared['eligible']); margin = np.full(n, np.nan)
        if len(ok):
            with torch.inference_mode():
                logits = self.native._forward(self.model, prepared['x'][ok], self.tool, self.device)
            margin[ok] = (logits[:, self.contract['host_index']] - logits[:, self.contract['microbe_index']]).cpu().numpy()
        return margin

    def decide(self, jobs):
        margin = self.margins(jobs)
        self.stats['steps'] += 1; self.stats['reads'] += len(jobs)
        out = []
        for s, m in zip(jobs, margin):
            s['k'] = 1
            if not np.isfinite(m):
                self.stats['fallbacks'] += 1; out.append((s, 'accept', dict(reason='input_fallback')))
            else:
                reject = self.threshold != 'all_keep' and m >= float(self.threshold)
                out.append((s, 'reject' if reject else 'accept', dict(margin=float(m))))
        return out


def build(name, device, calibration):
    if name == 'No-AS': return cl.NoAS()
    if name == 'PoreLoom': return PoreLoomPolicy(device, calibration)
    if name == 'PoreLoom-head0.2': return PoreLoomPolicy(device, calibration, bundle_path=MODELS['PoreLoom-head0.2'])
    return NativePolicy(name, device, calibration)


cl.STRATEGIES = NAMES
cl.build_strategy = build
_close = cl.Client.close_read


def _close_read(self, s, reason):
    release = getattr(self.strategy, 'release', None)
    if release is not None: release(s)
    _close(self, s, reason)


cl.Client.close_read = _close_read
_run = cl.Client.run


def _run_pinned(self):
    info = _run(self)
    info.update(final_wrapper=cl.pin(__file__), models=MODELS, break_read_ms=BREAK_MS, policy_stats=getattr(self.strategy, 'stats', None))
    return info


cl.Client.run = _run_pinned

if __name__ == '__main__':
    cl.main()
