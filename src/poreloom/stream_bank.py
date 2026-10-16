"""Device-resident state bank for StreamR0 (plan §10 M3): many reads in preallocated tensors, batched updates.

`StreamR0.step` keeps one R0State per read and loops over reads in Python; in the live client that per-read work
dominates the update time. The bank keeps the same quantities for up to `capacity` reads in a few [capacity, ...]
tensors (signal, normalized signal, block statistics, frame sums) plus numpy counters, so one update is a fixed
number of batched tensor operations whatever the batch size. It computes the same function as StreamR0.step and
readout (equal up to float summation order).
"""
import numpy as np
import torch

from .stream_encoder import _ragged, ragged_lengths
from .streaming import fresh_samples


def _ranges(begin, count):
    """Concatenated aranges [begin_i, begin_i + count_i) and the owner index of each element (numpy)."""
    count = np.asarray(count, np.int64)
    owner = np.repeat(np.arange(len(count)), count)
    start = np.cumsum(count) - count
    return np.arange(int(count.sum())) - start[owner] + np.asarray(begin, np.int64)[owner], owner


class StreamR0Bank:
    """States of up to `capacity` reads for one StreamR0 on `device`. A read holds a slot from `allocate()` until
    `release(slot)`; `step` takes chunks for distinct slots, `readout` gives logits for slots."""
    def __init__(self, model, capacity, device=None):
        self.model = model
        device = torch.device(device) if device is not None else model.normalizer.prior.device
        size = model.max_samples - model.skip
        norm = model.normalizer
        self.block, self.blocks = norm.block, -(-size//norm.block)
        self.raw = torch.zeros(capacity, size, device=device)
        self.z = torch.zeros(capacity, size, device=device)
        self.stats = torch.zeros(capacity, self.blocks, 2, device=device)
        self.last = torch.zeros(capacity, device=device)
        self.f_sum = torch.zeros(capacity, model.channels, device=device)
        self.tail = torch.zeros(capacity, model.channels, device=device)
        self.received, self.length, self.frames, self.tail_frames, self.used, self.done_blocks = \
            (np.zeros(capacity, np.int64) for _ in range(6))
        self.failed = np.zeros(capacity, bool)
        self.free = list(range(capacity - 1, -1, -1))
        self.device, self.capacity = device, capacity

    def allocate(self):
        if not self.free: raise RuntimeError('state bank full')
        slot = self.free.pop()
        for a in (self.received, self.length, self.frames, self.tail_frames, self.used, self.done_blocks): a[slot] = 0
        self.failed[slot] = False
        self.last[slot] = 0; self.f_sum[slot] = 0; self.tail[slot] = 0
        return slot

    def release(self, slot):
        self.free.append(int(slot))

    def _upload(self, arrays, dtype=np.int64):
        """Several 1-D arrays -> device tensors with one (pinned, asynchronous) copy, so no update syncs midway."""
        sizes = [len(a) for a in arrays]
        flat = torch.from_numpy(np.concatenate([np.asarray(a, dtype) for a in arrays]))
        if self.device.type == 'cuda': flat = flat.pin_memory().to(self.device, non_blocking=True)
        return torch.split(flat, sizes)

    @torch.no_grad()
    def step(self, slots, starts, signals):
        """signals[i]: pA samples at absolute offset starts[i] for slots[i] (None: nothing new). Same contract as
        StreamR0.step: repeated samples are dropped; a gap or non-finite sample marks the slot failed."""
        slots = np.asarray(slots, np.int64)
        if len(np.unique(slots)) != len(slots): raise ValueError('one chunk per slot')
        m = self.model
        grown, parts = [], []
        for slot, start, x in zip(slots, starts, signals):
            if x is None or self.failed[slot]: continue
            try:
                fresh = fresh_samples(int(self.received[slot]), int(start), x)
            except ValueError:
                self.failed[slot] = True
                continue
            a = int(self.received[slot]); e = a + len(fresh)
            lo, hi = max(a, m.skip), min(e, m.max_samples)
            self.received[slot] = e
            if hi > lo: grown.append(slot); parts.append(np.asarray(fresh[lo - a:hi - a], np.float32))
        if not grown: return
        # every index the update needs follows from the counters: build them all, then one upload
        g = np.array(grown, np.int64)
        n = np.array([len(p) for p in parts], np.int64)
        old = self.length[g].copy(); self.length[g] = old + n
        col, owner = _ranges(old, n)
        need = (old + n - 1)//self.block + 1
        b, pair = _ranges(self.done_blocks[g], np.maximum(need - self.done_blocks[g], 0))
        self.done_blocks[g] = np.maximum(self.done_blocks[g], need)
        used = self.length[g]//m.stride*m.stride
        up = used > self.used[g]
        f = dict(g=g[up], used=used[up])
        f['begin'] = np.maximum(0, self.frames[f['g']]*m.stride - m.context)
        f['first'] = self.frames[f['g']] - f['begin']//m.stride
        f['total'] = np.maximum(0, (f['used'] - m.backbone.right - 1)//m.stride + 1)
        f['split'] = f['total'] - f['begin']//m.stride                  # window index of the first tail frame
        f['width'] = f['used'] - f['begin']
        f['count'] = ragged_lengths(m.backbone.layers, f['width'])       # window frames
        f['ready'] = f['used'] >= m.min_samples                          # tails only once R0 can decide (as StreamR0)
        names = ['row', 'col', 'g', 'last', 'offset', 'owner', 'b', 'stat_row', 'fg', 'begin', 'first', 'split', 'width',
                 'fused', 'ready']
        dev = dict(zip(names, self._upload([g[owner], col, g, n - 1, col - old[owner], owner, b, g[pair], f['g'], f['begin'],
                                            f['first'], f['split'], f['width'], f['used'], f['ready']])))
        (x,) = self._upload([np.concatenate(parts)], np.float32)
        self._normalize(dev, x, len(g), int(n.max()), int(b.max())*self.block if len(b) else 0)
        if len(f['g']):
            self._frames(dev, int(f['width'].max()))
            self.frames[f['g']] = f['total']
            self.used[f['g']] = f['used']
            ready = f['ready']
            self.tail_frames[f['g'][ready]] = (f['count'] - f['split'])[ready]

    def _normalize(self, d, x, rows, width, prefix_width):
        """Store new samples, compute missing block statistics, normalize and hold outliers (CausalNormalizer)."""
        norm = self.model.normalizer
        dev = x.device
        self.raw[d['row'], d['col']] = x
        if len(d['b']):    # statistics of block b come from the stream prefix [0, block*b)
            prefix = (d['b']*self.block).float()
            m0, s0 = norm.prior.to(dev)
            if prefix_width:
                pre = self.raw[d['stat_row'], :prefix_width]
                pre = pre.masked_fill(torch.arange(prefix_width, device=dev) >= prefix[:, None], float('nan'))
                med = pre.nanmedian(1).values
                mad = (pre - med[:, None]).abs().nanmedian(1).values
                w = norm.prior_weight
                mean = torch.where(prefix > 0, (prefix*med + w*m0)/(prefix + w), m0)
                spread = torch.where(prefix > 0, (prefix*mad + w*s0)/(prefix + w), s0)
            else:
                mean, spread = m0.expand(len(prefix)), s0.expand(len(prefix))
            self.stats[d['stat_row'], d['b']] = torch.stack([mean, spread], 1)
        block = d['col']//self.block
        z = norm.scale(x, self.stats[d['row'], block, 0], self.stats[d['row'], block, 1])
        # hold: an outlier takes the last inlier on its left within this update, else the slot's last value
        padded = torch.full((rows, width), float('nan'), device=dev)
        padded[d['owner'], d['offset']] = z
        position = torch.arange(width, device=dev).expand_as(padded)
        last = torch.where(padded.abs() <= norm.clip, position, -1).cummax(1).values
        held = torch.where(last >= 0, padded.gather(1, last.clamp(min=0)), self.last[d['g']][:, None])
        self.z[d['row'], d['col']] = held[d['owner'], d['offset']]
        self.last[d['g']] = held[torch.arange(rows, device=dev), d['last']]

    def _frames(self, d, width):
        """One ragged pass of the original layers over [48F - context, used) per slot (F: final frames so far). Frames
        whose receptive field lies inside the window are the new final (delayed) frames, [F, final(used)), and go into
        f_sum; the rest reach the window's right edge, i.e. R0's right zero padding at `used`, and are the tail."""
        layers, rows = self.model.backbone.layers, d['fg']
        index = d['begin'][:, None] + torch.arange(width, device=rows.device)
        window = self.z[rows[:, None], index.clamp(max=self.z.shape[1] - 1)]*(index < d['fused'][:, None])
        f, count = _ragged(layers, window[:, None], d['width'])
        t = torch.arange(f.shape[-1], device=rows.device)
        lo, mid = d['first'][:, None], d['split'][:, None]
        self.f_sum[rows] = self.f_sum[rows] + torch.where(((t >= lo) & (t < mid))[:, None], f, 0.).sum(-1)
        tail = torch.where(((t >= mid) & (t < count[:, None]))[:, None], f, 0.).sum(-1)
        self.tail[rows] = torch.where(d['ready'].bool()[:, None], tail, self.tail[rows])

    @torch.no_grad()
    def readout(self, slots):
        """Logits [B,2] for the samples each slot has so far; NaN rows before its first decision."""
        m = self.model
        slots = np.asarray(slots, np.int64)
        st, count, used = self._upload([slots, np.maximum(self.frames[slots] + self.tail_frames[slots], 1), self.used[slots]])
        pooled = (self.f_sum[st] + self.tail[st])/count.float()[:, None]
        return m.logits(pooled, used).masked_fill((used < m.min_samples)[:, None], float('nan'))
