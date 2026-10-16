"""Causal sample-level preprocessing for the streaming encoder: calibration, chunk intake, blockwise normalization.

Every normalized sample depends only on earlier samples, so the result does not depend on how the stream is chunked.
"""
import numpy as np
import torch
from torch import nn

from .stream_config import DEFAULT_FS

FS = DEFAULT_FS


def to_pa(adc, offset, range_, digitisation):
    """ADC -> pA with per-read calibration (BLOW5 / read_until / Icarust get_calibration)."""
    return ((np.asarray(adc, np.float64) + offset) * (range_ / digitisation)).astype(np.float32)


def fresh_samples(received, offset, signal):
    """New samples of a chunk at absolute `offset`, given `received` samples already accepted.

    Same contract as windowing.SignalBuffer: repeated samples are dropped, gaps and invalid chunks raise ValueError
    (callers fail open).
    """
    signal = np.asarray(signal)
    if signal.ndim != 1 or offset < 0: raise ValueError('invalid signal chunk')
    if offset > received: raise ValueError('noncontiguous signal offset')
    fresh = signal[min(len(signal), received - offset):]
    if not np.isfinite(fresh).all(): raise ValueError('non-finite signal')
    return fresh


class CausalNormalizer(nn.Module):
    """Block b (samples [block*b, block*(b+1)) of the stream) is scaled by the median/MAD of samples [0, block*b),
    shrunk toward a TRAIN prior: m = (n*med + w*m0)/(n + w), same for MAD. Samples with |z| > clip take the last
    inlier value (0 before the first inlier). Median is torch's lower median."""
    def __init__(self, median, mad, prior_weight=400., block=480, clip=3.5):
        super().__init__()
        if not mad > 0 or prior_weight < 0 or block < 1: raise ValueError('invalid normalizer prior')
        self.register_buffer('prior', torch.tensor([float(median), float(mad)]))
        self.prior_weight, self.block, self.clip = float(prior_weight), int(block), float(clip)

    def stats(self, prefix):
        """prefix [B,n] -> shrunk (median, MAD), each [B,1]."""
        n = prefix.shape[1]
        m0, s0 = (p.expand(len(prefix), 1).to(prefix) for p in self.prior[:, None])
        if n == 0: return m0, s0
        med = prefix.median(1, keepdim=True).values
        mad = (prefix - med).abs().median(1, keepdim=True).values
        w = self.prior_weight
        return (n*med + w*m0)/(n + w), (n*mad + w*s0)/(n + w)

    @staticmethod
    def scale(x, m, s):
        return (x - m)/(1.4826*s)

    def hold(self, z, init):
        """Replace outliers by the last inlier on the left; `init` [B] where there is none."""
        position = torch.arange(z.shape[1], device=z.device).expand_as(z)
        last = torch.where(z.abs() <= self.clip, position, -1).cummax(1).values
        held = z.gather(1, last.clamp(min=0))
        return torch.where(last >= 0, held, init[:, None].to(z))

    def forward(self, x):
        """Offline: x [B,L] stream pA -> z [B,L]. Positions past a row's own length never affect earlier ones."""
        z = torch.empty_like(x)
        for lo in range(0, x.shape[1], self.block):
            z[:, lo:lo + self.block] = self.scale(x[:, lo:lo + self.block], *self.stats(x[:, :lo]))
        return self.hold(z, x.new_zeros(len(x)))

    def extend(self, streams, starts, stats, inits):
        """Online, several reads: z for streams[i][starts[i]:], given each whole stream so far.

        stats[i]: [n_i,2] per-block (m, s) already computed; inits[i]: last normalized value before starts[i] (0 at
        stream start). Missing block statistics are computed one block index at a time across reads.
        Returns (z rows, extended stats).
        """
        stats, blocks = list(stats), [-(-len(stream) // self.block) for stream in streams]
        for b in range(min(map(len, stats)), max(blocks)):
            rows = [i for i, table in enumerate(stats) if len(table) == b < blocks[i]]
            if not rows: continue
            new = torch.cat(self.stats(torch.stack([streams[i][:b*self.block] for i in rows])), 1)
            for i, row in zip(rows, new): stats[i] = torch.cat([stats[i], row[None]])
        table = nn.utils.rnn.pad_sequence(stats, batch_first=True, padding_value=1.)
        sizes = [len(stream) - start for stream, start in zip(streams, starts)]
        device = table.device
        position = torch.tensor(starts, device=device)[:, None] + torch.arange(max(sizes), device=device)
        ms = table.gather(1, (position // self.block).clamp(max=table.shape[1] - 1)[..., None].expand(-1, -1, 2))
        x = nn.utils.rnn.pad_sequence([stream[start:] for stream, start in zip(streams, starts)], batch_first=True)
        z = self.hold(self.scale(x, ms[..., 0], ms[..., 1]), torch.tensor(inits, dtype=x.dtype, device=device))
        return [row[:size] for row, size in zip(z, sizes)], stats
