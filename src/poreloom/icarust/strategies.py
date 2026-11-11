"""Strategies of the closed-loop comparison.

  No-AS              accept every read (stop_further_data once 7500 samples have arrived)
  PoreLoom           a trained bundle (poreloom_sh_cp.pt from ``python -m poreloom.method``) through the packaged live
                     policy (poreloom.minknow.policy.PoreLoomLivePolicy): streaming encoder state per read, one look
                     per decision interval, forced decision at the 7500-sample endpoint
  readcurrent, nanodeep, squigglenet
                     one native 3000-sample window (NanoDeep raw[1000:4000], the others raw[1500:4500]); a single
                     decision when the window is complete, with each model's native two-class decision
                     (probability 0.5)
  <tool>_w6000       the same with a 6000-sample window raw[1500:7500] (models trained on that window)

Every strategy reads the simulator signal on one ADC grid (MODEL_CALIBRATION, the simulated-data training contract);
a read whose input cannot be scored is accepted.
"""
from pathlib import Path
import json

import numpy as np

TOOLS = ('readcurrent', 'nanodeep', 'squigglenet')
STRATEGIES = ('No-AS', 'PoreLoom', *TOOLS, *(f'{t}_w6000' for t in TOOLS))
NAMES = dict(readcurrent='ReadCurrent', nanodeep='NanoDeep', squigglenet='SquiggleNet')
# Simulated R10 4 kHz training ADC contract: digitisation 8192, range 1536.598389 pA, offset = mean group-1 validation offset.
MODEL_CALIBRATION = dict(offset=13.464453946637898, range=1536.598389, digitisation=8192.)
NO_AS_SAMPLES = 7500
CAPACITY = 2048                 # encoder state slots: 512 channels plus slots released at the next step


def label(name):
    """Display name and observation of a strategy."""
    if name == 'No-AS': return 'Nonselective', '-'
    if name == 'PoreLoom': return 'PoreLoom', 'Budget 6000'
    tool, window = parse(name)
    return NAMES[tool], f'Window {window}'


def parse(name):
    """Baseline strategy -> (tool, window)."""
    tool = name[:-len('_w6000')] if name.endswith('_w6000') else name
    if tool not in TOOLS:
        raise ValueError(f'unknown strategy {name}; choose from {", ".join(STRATEGIES)}')
    return tool, 6000 if name.endswith('_w6000') else 3000


class NoAS:
    name = 'No-AS'
    max_samples = NO_AS_SAMPLES
    pins = None

    def __init__(self):
        self.stats = dict(steps=0, reads=0)

    def needed(self, s):
        return NO_AS_SAMPLES if s.get('k', 0) == 0 else None

    def decide(self, jobs):
        self.stats['steps'] += 1; self.stats['reads'] += len(jobs)
        for s in jobs:
            s['k'] = 1
        return [(s, 'accept', {}) for s in jobs]


def poreloom_bundle(path, workdir):
    """A PoreLoom model as a bundle file: poreloom_sh_cp.pt itself, a run directory holding it, or a directory with
    encoder.pt and head.pt (the layout of example/models/poreloom), which is written to workdir/poreloom_bundle.pt."""
    path = Path(path)
    if path.is_file():
        return path
    if (path/'poreloom_sh_cp.pt').is_file():
        return path/'poreloom_sh_cp.pt'
    if not ((path/'encoder.pt').is_file() and (path/'head.pt').is_file()):
        raise FileNotFoundError(f'{path}: expected poreloom_sh_cp.pt, or encoder.pt and head.pt')
    import torch
    from ..stream_training import load_stream_r0
    saved = torch.load(path/'head.pt', map_location='cpu')
    selection = json.loads((path/'selection.json').read_text()) if (path/'selection.json').is_file() else {}
    beta = float(selection.get('beta', 1.0))
    state = saved['states'][f'{beta:.6g}']
    model = load_stream_r0(path/'encoder.pt', 'cpu')[0]
    bundle = dict(method='PoreLoom-SH-ACC C-p', encoder=str((path/'encoder.pt').resolve()), head_state=state,
                  head=dict(n_buckets=saved['n_buckets'], hidden=saved['hidden'], input_dim=int(state['mean'].shape[0]),
                            look_config=saved.get('look_config')),
                  beta=beta, fs=float(saved['fs']), max_samples=model.max_samples, t_norm=saved.get('t_norm'),
                  source=str(path.resolve()))
    out = Path(workdir)/'poreloom_bundle.pt'
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out)
    return out


def poreloom(path, device, calibration, workdir):
    from ..minknow.policy import PoreLoomLivePolicy
    return PoreLoomLivePolicy(poreloom_bundle(path, workdir), device, calibration, capacity=CAPACITY)


_W3000 = None


def set_window(window):
    """Native window of every baseline in this process: 3000 (the trained defaults) or 6000 (raw[1500:7500]).
    The window contract is module state of poreloom.baselines.preprocessing, as in the paper's 6000-sample runs."""
    global _W3000
    from ..baselines import preprocessing as pp
    if _W3000 is None:
        _W3000 = {t: dict(c) for t, c in pp.CONTRACTS.items()}
    for tool, contract in pp.CONTRACTS.items():
        contract.update(_W3000[tool])
        if window == 6000:
            contract.update(raw_start=1500, raw_end=7500)
        elif window != 3000:
            raise ValueError(f'unsupported window {window}')
    pp.WINDOW = window


def checkpoint(path):
    """A baseline model: a checkpoint file, a run directory of python -m poreloom.method.baselines (fit/best.pt), or a
    directory holding best.pt."""
    path = Path(path)
    for p in (path, path/'fit'/'best.pt', path/'best.pt'):
        if p.is_file():
            return p
    raise FileNotFoundError(f'{path}: no baseline checkpoint (fit/best.pt or best.pt)')


def native_reject(margin, host_index):
    """The model's two-class argmax as a host-margin rule (torch argmax picks class 0 on exact ties)."""
    return margin >= 0 if host_index == 0 else margin > 0


class Native:
    def __init__(self, name, path, device, calibration):
        import torch
        from ..data import file_sha256
        from ..method import native
        from ..baselines.preprocessing import CONTRACTS
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        self.tool, self.window = parse(name)
        set_window(self.window)
        ckpt = checkpoint(path)
        stored = next((p for p in (ckpt.parent/'window.json', ckpt.parent.parent/'window.json') if p.is_file()), None)
        if stored is not None:
            window = json.loads(stored.read_text()).get('window')
            if window != self.window:
                raise ValueError(f'{ckpt} was trained on a {window}-sample window, not {self.window} ({name})')
        self.name, self.native, self.device, self.cal = name, native, device, dict(calibration)
        self.model = native._load(self.tool, ckpt, device)
        self.contract = dict(CONTRACTS[self.tool]); self.max_samples = self.contract['raw_end']
        self.pins = dict(checkpoint=dict(path=str(ckpt.resolve()), sha256=file_sha256(ckpt)), window=self.window,
                         raw_start=self.contract['raw_start'], raw_end=self.contract['raw_end'],
                         rule='native argmax (probability 0.5)', host_index=self.contract['host_index'])
        self.stats = dict(steps=0, reads=0, fallbacks=0)

    def needed(self, s):
        return self.max_samples if s.get('k', 0) == 0 else None

    def margins(self, jobs):
        import torch
        from ..prefix_data import PrefixSet
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
                out.append((s, 'reject' if native_reject(m, self.contract['host_index']) else 'accept', dict(margin=float(m))))
        return out


def build(name, model, device, calibration=MODEL_CALIBRATION, workdir='.'):
    """name: one of STRATEGIES; model: path of its trained model (unused for No-AS)."""
    if name == 'No-AS':
        return NoAS()
    if model is None:
        raise ValueError(f'{name} needs a model path (--model or --models)')
    if name == 'PoreLoom':
        return poreloom(model, device, calibration, workdir)
    return Native(name, model, device, calibration)


def models_config(path):
    """--models JSON: {strategy: model path}; relative paths are relative to the JSON file."""
    path = Path(path)
    doc = json.loads(path.read_text())
    unknown = sorted(set(doc) - set(STRATEGIES) - {'note'})
    if unknown:
        raise ValueError(f'{path}: unknown strategies {unknown}; expected keys from {", ".join(STRATEGIES)}')
    return {k: str((path.parent/v).resolve()) if not Path(v).is_absolute() else v
            for k, v in doc.items() if k in STRATEGIES and v}
