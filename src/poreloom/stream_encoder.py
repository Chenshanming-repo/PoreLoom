"""Streaming ReadCurrent backbone and frame encoder for any chunk length.

No layer pads on the right, so a frame exists only once every sample it depends on has arrived and never changes
afterwards. Two left-padding choices:
  delayed  each layer keeps its original left padding: frames are exactly the original (non-causal) ReadCurrent
           frames, emitted 529 samples late (frame t depends on [48t-524, 48t+529]); transferred weights stay valid.
  causal   every layer pads k - s on the left: frame t depends on [48t-1006, 48t+47], no delay, but the residual
           branches no longer line up, which breaks transferred weights (2026-10-12 diagnostic).
Online updates recompute "context + new samples" (overlap-save) and keep only the new frames, which makes `step`
equal to the offline `forward` however the signal is chunked.

StreamR0 (plan §4.6) streams R0 exactly: the stage A backbone over the whole stream, mean-pooled over all frames.
Its frames are the delayed frames plus a tail of 11 frames that reach the right zero padding; only the tail is
recomputed at each update.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import build_model
from .streaming import FS, CausalNormalizer, fresh_samples
from .stream_config import DEFAULT_MAX_SAMPLES, sampling_rate


def _is_block(module):
    return all(hasattr(module, name) for name in ('conv1', 'bn1', 'conv2', 'bn2', 'relu', 'shortcut', 'downsample'))


BOTTLENECKS = (('conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3'),   # SquiggleNet Bottleneck
               ('c1', 'b1', 'c2', 'b2', 'c3', 'b3'))               # NanoDeep ResidualBlock1D


def _bottleneck(module):
    """[(conv, bn) x 3] of a three-conv residual block (relu after the first two, then + identity or downsample(x),
    relu), or None. Used by the backbone-replacement experiment (experiments/20261015_backbone)."""
    if _is_block(module) or not all(hasattr(module, n) for n in ('relu', 'downsample')): return None
    for names in BOTTLENECKS:
        if all(hasattr(module, n) for n in names):
            return [(getattr(module, names[i]), getattr(module, names[i + 1])) for i in (0, 2, 4)]
    return None


ELEMENTWISE = (nn.BatchNorm1d, nn.ReLU, nn.Dropout)


class Residual(nn.Module):
    """x -> body(x) + x, then ReLU if `relu` (body: stride 1, centered padding, so output j stays aligned with x[j])."""
    def __init__(self, body, relu):
        super().__init__()
        self.body, self.relu = body, bool(relu)

    def forward(self, x):
        out = self.body(x) + x
        return F.relu(out) if self.relu else out


class CausalSE(nn.Module):
    """Squeeze-excitation with a causal squeeze: r = body(x); out[t] = r[t]*fc(mean(r[0..t])) + x[t]. Frame t depends on
    the whole prefix, so a stream containing it recomputes from the stream start (_StreamBase sets the context to
    the full budget); frames stay final once they exist."""
    def __init__(self, body, fc):
        super().__init__()
        self.body, self.fc = body, fc

    def gate(self, r):
        mean = r.cumsum(-1)/torch.arange(1, r.shape[-1] + 1, device=r.device, dtype=r.dtype)
        return self.fc(mean.transpose(1, 2)).transpose(1, 2)

    def forward(self, x):
        r = self.body(x)
        return r*self.gate(r) + x


ALIGNMENTS = ('delayed', 'causal')


def _layer(module, alignment):
    """(kernel, stride, left padding) of a conv or max-pool layer."""
    if isinstance(module, nn.Conv1d):
        k, s, p, d = module.kernel_size[0], module.stride[0], module.padding[0], module.dilation[0]
    elif isinstance(module, nn.AvgPool1d):
        (k,), (s,), (p,), d = module.kernel_size, module.stride, module.padding, 1
    else:
        k, s, p, d = module.kernel_size, module.stride, module.padding, module.dilation
    if d != 1 or (k < s and k != 1): raise ValueError('streaming layers need dilation 1 and kernel >= stride (or kernel 1)')
    return k, s, (p if alignment == 'delayed' else k - s)


def _stream(module, x, alignment):
    if isinstance(module, nn.Conv1d):
        k, s, p = _layer(module, alignment)
        return F.conv1d(F.pad(x, (p, 0)), module.weight, module.bias, s)
    if isinstance(module, nn.MaxPool1d):
        k, s, p = _layer(module, alignment)
        return F.max_pool1d(F.pad(x, (p, 0), value=-math.inf), k, s)
    if isinstance(module, nn.AvgPool1d):
        k, s, p = _layer(module, alignment)
        return F.avg_pool1d(F.pad(x, (p, 0)), k, s)
    if isinstance(module, CausalSE):
        r = _stream(module.body, x, alignment)
        return r*module.gate(r) + x[..., :r.shape[-1]]
    if isinstance(module, Residual):
        out = _stream(module.body, x, alignment)
        out = out + x[..., :out.shape[-1]]
        return F.relu(out) if module.relu else out
    if _bottleneck(module):
        (c1, b1), (c2, b2), (c3, b3) = _bottleneck(module)
        out = b3(_stream(c3, module.relu(b2(_stream(c2, module.relu(b1(_stream(c1, x, alignment))), alignment))), alignment))
        identity = x if module.downsample is None else _stream(module.downsample, x, alignment)
        return module.relu(out + identity[..., :out.shape[-1]])
    if _is_block(module):
        out = module.bn2(_stream(module.conv2, module.relu(module.bn1(_stream(module.conv1, x, alignment))), alignment))
        if module.shortcut:  # without right padding the main branch is shorter; both start at the same position
            out = out + (x if module.downsample is None else _stream(module.downsample, x, alignment))[..., :out.shape[-1]]
        return module.relu(out)
    if isinstance(module, nn.Sequential):
        for child in module: x = _stream(child, x, alignment)
        return x
    if isinstance(module, ELEMENTWISE): return module(x)
    raise TypeError(f'no streaming form for {type(module).__name__}')


def _ragged(module, x, lengths):
    """Original (two-sided padding) forward of rows of different lengths in one zero-padded batch: x [B,C,L],
    lengths [B] (on x's device) -> (out, out lengths). Each conv / max-pool input is masked past its row's length
    (0, or -inf for max-pool), which is exactly that layer's own padding, so every row's valid outputs equal its own
    forward. Positions past a row's length hold garbage (possibly non-finite)."""
    def masked(x, n, value):
        return x.masked_fill(torch.arange(x.shape[-1], device=x.device) >= n[:, None, None], value)
    if isinstance(module, (nn.Conv1d, nn.MaxPool1d)):
        conv = isinstance(module, nn.Conv1d)
        k, s, p = (module.kernel_size[0], module.stride[0], module.padding[0]) if conv else \
            (module.kernel_size, module.stride, module.padding)
        if (module.dilation[0] if conv else module.dilation) != 1 or (not conv and module.ceil_mode):
            raise ValueError('ragged layers need dilation 1 and floor mode')
        return module(masked(x, lengths, 0. if conv else -math.inf)), (lengths + 2*p - k)//s + 1
    if isinstance(module, nn.AvgPool1d):
        (k,), (s,), (p,) = module.kernel_size, module.stride, module.padding
        if module.ceil_mode or not module.count_include_pad: raise ValueError('ragged avg-pool needs floor mode, zero pad')
        return module(masked(x, lengths, 0.)), (lengths + 2*p - k)//s + 1
    if isinstance(module, CausalSE):
        r, n = _ragged(module.body, x, lengths)
        return r*module.gate(r) + x, n
    if isinstance(module, Residual):
        out, n = _ragged(module.body, x, lengths); out = out + x
        return (F.relu(out) if module.relu else out), n
    if _bottleneck(module):
        (c1, b1), (c2, b2), (c3, b3) = _bottleneck(module)
        out, n = _ragged(c1, x, lengths)
        out, n = _ragged(c2, module.relu(b1(out)), n)
        out, n = _ragged(c3, module.relu(b2(out)), n)
        identity = x if module.downsample is None else _ragged(module.downsample, x, lengths)[0]
        return module.relu(b3(out) + identity), n
    if _is_block(module):
        out, n = _ragged(module.conv1, x, lengths)
        out, n = _ragged(module.conv2, module.relu(module.bn1(out)), n)
        out = module.bn2(out)
        if module.shortcut:
            out = out + (x if module.downsample is None else _ragged(module.downsample, x, lengths)[0])
        return module.relu(out), n
    if isinstance(module, nn.Sequential):
        for child in module: x, lengths = _ragged(child, x, lengths)
        return x, lengths
    if isinstance(module, ELEMENTWISE): return module(x), lengths
    raise TypeError(f'no ragged form for {type(module).__name__}')


def ragged_lengths(module, lengths):
    """Output lengths of _ragged (the original layers) for input lengths (numpy), computed on the host."""
    if isinstance(module, (nn.Conv1d, nn.MaxPool1d)):
        conv = isinstance(module, nn.Conv1d)
        k, s, p = (module.kernel_size[0], module.stride[0], module.padding[0]) if conv else \
            (module.kernel_size, module.stride, module.padding)
        return (np.asarray(lengths) + 2*p - k)//s + 1
    if isinstance(module, nn.AvgPool1d):
        (k,), (s,), (p,) = module.kernel_size, module.stride, module.padding
        return (np.asarray(lengths) + 2*p - k)//s + 1
    if isinstance(module, (Residual, CausalSE)): return ragged_lengths(module.body, lengths)
    if _bottleneck(module):
        for conv, _ in _bottleneck(module): lengths = ragged_lengths(conv, lengths)
        return np.asarray(lengths)
    if _is_block(module): return ragged_lengths(module.conv2, ragged_lengths(module.conv1, lengths))
    if isinstance(module, nn.Sequential):
        for child in module: lengths = ragged_lengths(child, lengths)
    return np.asarray(lengths)


def _geometry(module, alignment, left=0, right=0, stride=1):
    """(left extent, right extent, cumulative stride): unit j depends on samples [stride*j - left, stride*j + right]."""
    if isinstance(module, (nn.Conv1d, nn.MaxPool1d, nn.AvgPool1d)):
        k, s, p = _layer(module, alignment)
        return left + stride*p, right + stride*(k - 1 - p), stride*s
    if isinstance(module, (Residual, CausalSE)): return _geometry(module.body, alignment, left, right, stride)
    if _bottleneck(module):
        for conv, _ in _bottleneck(module): left, right, stride = _geometry(conv, alignment, left, right, stride)
        return left, right, stride
    if _is_block(module):
        for conv in (module.conv1, module.conv2): left, right, stride = _geometry(conv, alignment, left, right, stride)
        return left, right, stride
    if isinstance(module, nn.Sequential):
        for child in module: left, right, stride = _geometry(child, alignment, left, right, stride)
    return left, right, stride


class StreamBackbone(nn.Module):
    """ReadCurrent `layers[:-1]` (same modules, same parameter names) run without right padding."""
    def __init__(self, layers, alignment='delayed'):
        super().__init__()
        if alignment not in ALIGNMENTS: raise ValueError(f'alignment must be one of {ALIGNMENTS}')
        self.layers, self.alignment = layers, alignment
        self.left, self.right, self.stride = _geometry(layers, alignment)
        self.receptive_field = self.left + self.right + 1
        self.channels = [m for m in layers.modules() if isinstance(m, nn.Conv1d)][-1].out_channels

    def frames(self, length):
        """Frames whose samples [stride*t - left, stride*t + right] all lie in [0, length)."""
        return max(0, (length - self.right - 1)//self.stride + 1)

    def forward(self, z):
        """z [B,L] -> [B,C,frames(L)]."""
        if self.frames(z.shape[1]) == 0: return z.new_zeros((len(z), self.channels, 0))
        return _stream(self.layers, z[:, None], self.alignment)


@dataclass
class StreamState:
    """One read. `received` counts absolute samples accepted. raw/z (CPU, preallocated) hold the first `length`
    stream samples (from `skip`, at most max_samples - skip); stats holds per-block (m, s); h/e_sum live on the
    model device."""
    raw: torch.Tensor
    z: torch.Tensor
    h: torch.Tensor
    e_sum: torch.Tensor
    stats: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 2))
    received: int = 0
    length: int = 0
    frames: int = 0
    last: float = 0.
    failed: bool = False


@dataclass
class R0State:
    """One read for StreamR0: like StreamState, with the sum of final frames (`f_sum`), the sum and count of the
    current tail frames, and `used`, the stream samples (a multiple of 48) the tail was computed for."""
    raw: torch.Tensor
    z: torch.Tensor
    f_sum: torch.Tensor
    tail: torch.Tensor
    stats: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 2))
    received: int = 0
    length: int = 0
    frames: int = 0
    tail_frames: int = 0
    used: int = 0
    last: float = 0.
    failed: bool = False


class _StreamBase(nn.Module):
    """Intake, causal normalization and final (delayed or causal) frames of the stream [skip, max_samples)."""
    def __init__(self, layers, normalizer, skip, max_samples, fs, alignment):
        super().__init__()
        if not 0 <= skip < max_samples: raise ValueError('need 0 <= skip < max_samples')
        self.backbone, self.normalizer = StreamBackbone(layers, alignment), normalizer
        self.skip, self.max_samples, self.fs = int(skip), int(max_samples), sampling_rate(fs)
        self.stride = self.backbone.stride
        self.context = -(-self.backbone.left // self.stride)*self.stride
        if any(isinstance(m, CausalSE) for m in layers.modules()):   # prefix state: recompute from the stream start
            self.context = -(-(self.max_samples - self.skip) // self.stride)*self.stride

    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d): module.eval()  # frozen statistics: a fixed per-channel affine map
        return self

    def frames_at(self, n):
        """Final frames once absolute sample count n has arrived (tensor or int)."""
        stream = (n - self.skip).clamp(min=0) if torch.is_tensor(n) else max(0, n - self.skip)
        count = (stream - self.backbone.right - 1)//self.stride + 1
        return count.clamp(min=0) if torch.is_tensor(n) else max(0, count)

    def _intake(self, state, offset, signal):
        """Account for the chunk's in-budget fresh samples; return (previous stream length, samples) for the caller
        to store in state.raw, or None if the stream did not grow."""
        try:
            fresh = fresh_samples(state.received, offset, signal)
        except ValueError:
            state.failed = True
            return None
        start, end = state.received, state.received + len(fresh)
        lo, hi = max(start, self.skip), min(end, self.max_samples)
        state.received = end
        if hi <= lo: return None
        old = state.length
        state.length += hi - lo
        return old, np.array(fresh[lo - start:hi - start], np.float32)

    def _ingest(self, states, chunks):
        """Take in one chunk per state (None: nothing new) and normalize the new samples; return the states whose
        stream grew. A gap, invalid chunk or non-finite sample marks the read failed (callers fail open) and it is
        skipped after."""
        if len(states) != len(chunks): raise ValueError('one chunk per state')
        grown, device, host = [], [], []
        for state, chunk in zip(states, chunks):
            if state.failed or chunk is None: continue
            got = self._intake(state, *chunk)
            if got is None: continue
            grown.append((state, got[0]))
            (host if state.raw.device.type == 'cpu' else device).append((state, *got))
        if not grown: return []
        for state, old, x in host: state.raw[old:old + len(x)] = torch.from_numpy(x)
        if device:  # one host-to-device copy for the batch
            flat, at = torch.from_numpy(np.concatenate([x for *_, x in device])).to(device[0][0].raw.device), 0
            for state, old, x in device:
                state.raw[old:old + len(x)] = flat[at:at + len(x)]; at += len(x)
        z, stats = self.normalizer.extend([s.raw[:s.length] for s, _ in grown], [old for _, old in grown],
                                          [s.stats for s, _ in grown], [s.last for s, _ in grown])
        lasts = torch.stack([new[-1] for new in z]).tolist()  # one host sync per batch
        for (state, old), new, table, last in zip(grown, z, stats, lasts):
            state.z[old:state.length], state.stats, state.last = new, table, last
        return [state for state, _ in grown]

    def _new_frames(self, states, totals, embed):
        """Final frames [state.frames, total) of each state via overlap-save: (states that got any, embed(frames)
        [J,T,D] zero-padded, counts [J]). Does not advance state.frames."""
        jobs = []
        for state, total in zip(states, totals):
            if total <= state.frames: continue
            begin = max(0, state.frames*self.stride - self.context)
            jobs.append((state, begin, state.frames - begin//self.stride, total - state.frames))
        if not jobs: return [], None, None
        device = self.normalizer.prior.device
        windows = [state.z[begin:state.length] for state, begin, *_ in jobs]
        e = embed(nn.utils.rnn.pad_sequence(windows, batch_first=True).to(device))
        first, count = (torch.tensor(column, device=device) for column in list(zip(*jobs))[2:])
        offsets = torch.arange(int(count.max()), device=device)
        index = (first[:, None] + offsets).clamp(max=e.shape[1] - 1)
        new = e.gather(1, index[..., None].expand(-1, -1, e.shape[-1]))*(offsets < count[:, None])[..., None]
        return [state for state, *_ in jobs], new, count


class StreamEncoder(_StreamBase):
    """Normalizer -> streaming backbone -> per-frame projection -> GRU; s(n) = [h, mean e, n/max_samples, log1p(n/fs)].

    The stream starts at absolute sample `skip`; frame t is final ("ends") at skip + 48*t + right + 1. At most
    `max_samples` samples are used.
    """
    feature_dim = 258

    def __init__(self, layers, projection, normalizer, skip=1500, max_samples=7500, fs=FS, alignment='delayed'):
        super().__init__(layers, normalizer, skip, max_samples, fs, alignment)
        self.projection = projection
        self.gru = nn.GRU(128, 128, batch_first=True)

    def frame_ends(self, frames):
        return self.skip + self.stride*np.arange(frames) + self.backbone.right + 1

    def time_features(self, n):
        n = torch.as_tensor(n, dtype=torch.float32, device=self.normalizer.prior.device).clamp(max=self.max_samples)
        return torch.stack([n/self.max_samples, torch.log1p(n/self.fs)], -1)

    def _embed(self, z):
        return self.projection(self.backbone(z).transpose(1, 2))

    def forward(self, pa, lengths):
        """Offline: pa [B,L] from read start, lengths [B] -> memory [B,F,256] = [h, mean e] per frame, valid [B,F]."""
        lengths = torch.as_tensor(lengths, device=pa.device).clamp(max=self.max_samples)
        x = pa[:, self.skip:self.max_samples].float()
        stream = (lengths - self.skip).clamp(min=0)
        x = x.masked_fill(torch.arange(x.shape[1], device=x.device) >= stream[:, None], 0)
        if self.backbone.frames(x.shape[1]) == 0:
            return x.new_zeros((len(x), 0, 256)), torch.zeros((len(x), 0), dtype=torch.bool, device=x.device)
        e = self._embed(self.normalizer(x))
        h, _ = self.gru(e)
        count = torch.arange(1, e.shape[1] + 1, device=e.device, dtype=e.dtype)[:, None]
        valid = torch.arange(e.shape[1], device=e.device) < self.frames_at(stream + self.skip)[:, None]
        return torch.cat([h, e.cumsum(1)/count], -1), valid

    def state_at(self, memory, n):
        """Decision state s(n) [B,258] from offline memory: last frame ending at or before n."""
        n = torch.as_tensor(n, device=memory.device).clamp(max=self.max_samples)
        frames = self.frames_at(n).clamp(max=memory.shape[1])
        picked = memory[torch.arange(len(memory), device=memory.device), (frames - 1).clamp(min=0)]
        picked = torch.where((frames > 0)[:, None], picked, torch.zeros_like(picked))
        return torch.cat([picked, self.time_features(n)], -1)

    def init_state(self, signal_device='cpu'):
        """signal_device: where raw/z and the normalization run (the model device avoids host-side medians)."""
        device = self.normalizer.prior.device
        size = self.max_samples - self.skip
        return StreamState(raw=torch.zeros(size, device=signal_device), z=torch.zeros(size, device=signal_device),
                           h=torch.zeros(128, device=device), e_sum=torch.zeros(128, device=device),
                           stats=torch.zeros(0, 2, device=signal_device))

    @torch.no_grad()
    def step(self, states, chunks):
        """Online: chunks[i] = (absolute offset, pA samples) or None for states[i]; updates states in place."""
        grown = self._ingest(states, chunks)
        jobs, new, count = self._new_frames(grown, [self.frames_at(self.skip + s.length) for s in grown], self._embed)
        if not jobs: return
        device = new.device
        out, _ = self.gru(new, torch.stack([state.h for state in jobs])[None])
        h = out[torch.arange(len(jobs), device=device), count - 1]
        e_sum = torch.stack([state.e_sum for state in jobs]) + new.sum(1)
        for i, state in enumerate(jobs):
            state.h, state.e_sum, state.frames = h[i], e_sum[i], state.frames + int(count[i])

    def readout(self, states):
        """Current decision states s(n) [B,258]; n = samples received (capped at max_samples)."""
        memory = torch.stack([torch.cat([s.h, s.e_sum/max(s.frames, 1)]) for s in states])
        return torch.cat([memory, self.time_features([s.received for s in states])], -1)


class StreamClassifier(nn.Module):
    """StreamEncoder + auxiliary head p(target | s(n)) (258 -> 64 -> 2, same shape as PoreLoom's head)."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Linear(encoder.feature_dim, 64), nn.GELU(), nn.Dropout(.1), nn.Linear(64, 2))

    def frame_logits(self, memory):
        """Logits [B,F,2] with s taken at each frame end."""
        ends = torch.as_tensor(self.encoder.frame_ends(memory.shape[1]), device=memory.device)
        time = self.encoder.time_features(ends).to(memory.dtype).expand(len(memory), -1, -1)
        return self.head(torch.cat([memory, time], -1))

    def forward(self, pa, lengths):
        """Per-frame logits [B,F,2], valid [B,F]."""
        memory, valid = self.encoder(pa, lengths)
        return self.frame_logits(memory), valid

    def logits_at(self, memory, n):
        return self.head(self.encoder.state_at(memory, n))


class StreamR0(_StreamBase):
    """R0 streamed exactly (plan §4.6): at absolute sample count n, the stage A backbone (original two-sided zero
    padding) over the normalized stream [skip, skip + L'), L' = (min(n, max_samples) - skip) rounded down to a
    multiple of 48, mean-pooled over all frames -> projection -> head. No decision before L' >= min_samples.

    The first (L' - 530)//48 + 1 frames do not reach the right padding and equal the delayed frames, so their sum
    is cached; the remaining 11 (the tail) are recomputed from the window [48F - 528, L'), 1056 samples once
    L' >= 1056, so every read's tail goes through the network in one batch.

    time_features=True adds Linear(phi(n)) to the logits, phi(n) = [n/max_samples, log1p(n/fs)] at n = skip + L',
    zero-initialized so training starts from R0 (plan §10 M2.7).
    """
    def __init__(self, layers, projection, head, normalizer, skip=1500, max_samples=DEFAULT_MAX_SAMPLES, fs=FS, min_samples=400,
                 time_features=False):
        super().__init__(layers, normalizer, skip, max_samples, fs, 'delayed')
        self.projection, self.head = projection, head
        self.min_samples = -(-int(min_samples)//self.stride)*self.stride
        self.channels = self.backbone.channels
        self.time_head = None
        if time_features:
            self.time_head = nn.Linear(2, head.out_features)
            nn.init.zeros_(self.time_head.weight); nn.init.zeros_(self.time_head.bias)

    def used(self, stream):
        """Stream samples R0 uses out of `stream` received after skip (tensor or int)."""
        return stream//self.stride*self.stride

    def _final(self, used):
        return self.backbone.frames(used) if not torch.is_tensor(used) else \
            ((used - self.backbone.right - 1)//self.stride + 1).clamp(min=0)

    def _tails(self, windows, firsts):
        """windows: list of z windows [begin, used); firsts: index of the first tail frame in each. Returns tail
        sums [N,C] and counts [N]; windows of equal length share one forward pass."""
        device = self.normalizer.prior.device
        sums = torch.zeros(len(windows), self.channels, device=device)
        counts = torch.zeros(len(windows), dtype=torch.long, device=device)
        groups = {}
        for i, (window, first) in enumerate(zip(windows, firsts)): groups.setdefault((len(window), first), []).append(i)
        for (_, first), rows in groups.items():
            x = torch.stack([windows[i] for i in rows]).to(device)
            f = self.backbone.layers(x[:, None])[..., first:]
            sums[rows], counts[rows] = f.sum(-1), f.shape[-1]
        return sums, counts

    def _tail_window(self, used, final):
        begin = max(0, final*self.stride - self.context)
        return begin, final - begin//self.stride

    def logits(self, pooled, used=None):
        """Logits from mean-pooled backbone features; `used` (stream samples, tensor) feeds the time term."""
        out = self.head(self.projection(pooled))
        if self.time_head is None: return out
        n = (self.skip + used).to(pooled.dtype)
        return out + self.time_head(torch.stack([n/self.max_samples, torch.log1p(n/self.fs)], -1))

    def forward(self, pa, lengths, n):
        """Offline: pa [B,L] from read start, lengths [B], absolute decision sample counts n [B,P] (capped at the
        read length) -> logits [B,P,2] (NaN where there is no decision yet), valid [B,P]."""
        lengths = torch.as_tensor(lengths, device=pa.device).clamp(max=self.max_samples)
        n = torch.minimum(torch.as_tensor(n, device=pa.device), lengths[:, None])
        x = pa[:, self.skip:self.max_samples].float()
        stream = (lengths - self.skip).clamp(min=0)
        z = self.normalizer(x.masked_fill(torch.arange(x.shape[1], device=x.device) >= stream[:, None], 0))
        used = self.used((n - self.skip).clamp(min=0))
        valid = used >= self.min_samples
        final = self._final(used)
        f = self.backbone(z)
        csum = torch.cat([f.new_zeros(len(f), f.shape[1], 1), f.cumsum(-1)], -1)
        f_sum = csum.gather(-1, final.clamp(max=f.shape[-1])[:, None, :].expand(-1, f.shape[1], -1)).transpose(1, 2)
        rows, cols = valid.nonzero(as_tuple=True)
        windows, firsts = [], []
        for b, j in zip(rows.tolist(), cols.tolist()):
            begin, first = self._tail_window(int(used[b, j]), int(final[b, j]))
            windows.append(z[b, begin:int(used[b, j])]); firsts.append(first)
        pooled = f_sum.clone()
        if windows:
            sums, counts = self._tails(windows, firsts)
            pooled[rows, cols] = (f_sum[rows, cols] + sums)/(final[rows, cols] + counts)[:, None]
        logits = self.logits(pooled, used)
        return logits.masked_fill(~valid[..., None], math.nan), valid

    def init_state(self, signal_device='cpu'):
        """signal_device: where raw/z and the normalization run (the model device avoids host-side medians)."""
        device = self.normalizer.prior.device
        size = self.max_samples - self.skip
        return R0State(raw=torch.zeros(size, device=signal_device), z=torch.zeros(size, device=signal_device),
                       f_sum=torch.zeros(self.channels, device=device), tail=torch.zeros(self.channels, device=device),
                       stats=torch.zeros(0, 2, device=signal_device))

    @torch.no_grad()
    def step(self, states, chunks):
        """Online: chunks[i] = (absolute offset, pA samples) or None for states[i]; updates states in place."""
        grown = [s for s in self._ingest(states, chunks) if self.used(s.length) > s.used]
        jobs, new, count = self._new_frames(grown, [self._final(self.used(s.length)) for s in grown],
                                            lambda z: self.backbone(z).transpose(1, 2))
        if jobs:
            f_sum = torch.stack([s.f_sum for s in jobs]) + new.sum(1)
            for s, f, c in zip(jobs, f_sum, count.tolist()): s.f_sum, s.frames = f, s.frames + c
        ready = [s for s in grown if self.used(s.length) >= self.min_samples]
        windows, firsts = [], []
        for s in grown: s.used = self.used(s.length)
        for s in ready:
            begin, first = self._tail_window(s.used, s.frames)
            windows.append(s.z[begin:s.used]); firsts.append(first)
        if not ready: return
        sums, counts = self._tails(windows, firsts)
        for s, total, c in zip(ready, sums, counts.tolist()): s.tail, s.tail_frames = total, c

    def readout(self, states):
        """Logits [B,2] for the samples received so far; NaN rows before the first decision."""
        f_sum, tail = torch.stack([s.f_sum for s in states]), torch.stack([s.tail for s in states])
        count = torch.tensor([max(s.frames + s.tail_frames, 1) for s in states], dtype=f_sum.dtype, device=f_sum.device)
        pooled = (f_sum + tail)/count[:, None]
        ready = torch.tensor([s.used >= self.min_samples for s in states], device=pooled.device)
        used = torch.tensor([s.used for s in states], device=pooled.device)
        return self.logits(pooled, used).masked_fill(~ready[:, None], math.nan)


def build_stream_encoder(checkpoint=None, *, median, mad, prior_weight=400., skip=1500, max_samples=7500,
                         alignment='delayed'):
    """Causal encoder with the backbone and projection of a PoreLoom checkpoint (stage A or later), or random init."""
    base = build_model()
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location='cpu')
        base.load_state_dict(state.get('model_state', state), strict=True)
    return StreamEncoder(base.encoder, base.projection, CausalNormalizer(median, mad, prior_weight),
                         skip=skip, max_samples=max_samples, alignment=alignment).train(False)


def build_stream_r0(checkpoint=None, *, median, mad, prior_weight=400., skip=1500, max_samples=DEFAULT_MAX_SAMPLES, min_samples=400,
                    time_features=False, fs=FS):
    """StreamR0 with the backbone, projection and temporary head of a stage A checkpoint, or random init."""
    base, head = build_model(), nn.Linear(128, 2)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location='cpu')
        base.load_state_dict(payload['model_state'], strict=True)
        head.load_state_dict(payload['temporary_head'])
    return StreamR0(base.encoder, base.projection, head, CausalNormalizer(median, mad, prior_weight), skip=skip,
                    max_samples=max_samples, min_samples=min_samples, time_features=time_features, fs=fs).train(False)
