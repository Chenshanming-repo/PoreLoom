"""Score native windows with a fitted baseline: one host margin per parent read."""
import numpy as np
import torch

from .models import load_model
from .native_source import NativeSource
from .preprocessing import CONTRACTS, host_margin, prepare_windows


@torch.inference_mode()
def score_windows(model, windows, tool, *, device='cpu', batch_size=256):
    """Host margins for raw native windows [N, 3000]; NaN where the input is invalid."""
    windows = np.asarray(windows)
    margins = np.full(len(windows), np.nan, np.float32)
    for start in range(0, len(windows), batch_size):
        x, valid = prepare_windows(windows[start:start + batch_size], tool)
        if valid.any():
            logits = model(torch.from_numpy(np.ascontiguousarray(x[valid])).to(device)).cpu().numpy()
            margins[start + np.flatnonzero(valid)] = host_margin(logits, tool)
    return margins


def score_source(tool, checkpoint, native, *, rows=None, device='cpu', batch_size=256):
    """Score every eligible parent (or ``rows``) of a native export.

    Returns ``margin`` (NaN where no valid window), ``labels``, ``rows``.
    """
    source = NativeSource(native)
    key = CONTRACTS[tool]['key']
    windows = source.windows(key)
    rows = np.arange(source.n) if rows is None else np.asarray(rows)
    eligible = source.eligible(key)[rows]
    model = load_model(tool, checkpoint, device)
    margin = np.full(len(rows), np.nan, np.float32)
    margin[eligible] = score_windows(model, windows[rows[eligible]], tool, device=device, batch_size=batch_size)
    return dict(margin=margin, labels=source.labels[rows], rows=rows)
