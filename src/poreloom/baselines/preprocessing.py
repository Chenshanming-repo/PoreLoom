"""Per-tool window contracts, input normalization and host margins.

Every tool sees one fixed 3000-sample window of the read's native signal:

- NanoDeep: ADC raw[1000:4000], ``(x - mean) / var`` of the observed window.
- ReadCurrent: ADC raw[1500:4500], ``(x - median) / (1.4826 * MAD)``, then each
  original |z| > 3.5 index is replaced in ascending order by its neighbors' mean.
- CAFES: ADC raw[1500:4500], the six upstream features computed at batch time.
- SquiggleNet: calibrated pA raw[1500:4500], the official ``normalization``.

Project labels are microbe 0 / host 1. ReadCurrent and CAFES were trained with
network labels flipped (host 0), so the host margin indexes logits per tool.
"""
import ast

import numpy as np
from scipy import stats
import torch

from .models import SQUIGGLENET_ROOT, cafes_preprocessor

CONTRACTS = {
    'nanodeep': dict(units='ADC', key='adc_1000_4000', raw_start=1000, raw_end=4000, host_index=1, microbe_index=0),
    'cafes': dict(units='ADC', key='adc_1500_4500', raw_start=1500, raw_end=4500, host_index=0, microbe_index=1),
    'readcurrent': dict(units='ADC', key='adc_1500_4500', raw_start=1500, raw_end=4500, host_index=0, microbe_index=1),
    'squigglenet': dict(units='pA', key='pa_1500_4500', raw_start=1500, raw_end=4500, host_index=1, microbe_index=0),
}
WINDOW = 3000


def _windows(values, dtype):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != WINDOW or values.dtype != np.dtype(dtype):
        raise ValueError(f'expected {np.dtype(dtype)} windows [N, {WINDOW}], got {values.dtype} {values.shape}')
    if not np.isfinite(values).all():
        raise ValueError('nonfinite source window')
    return values


def nanodeep_inputs(adc):
    x = _windows(adc, np.int16).astype(np.float64)
    variance = x.var(axis=1)
    valid = variance > 1e-8
    out = np.zeros(x.shape, np.float32)
    out[valid] = ((x[valid] - x[valid].mean(axis=1, keepdims=True)) / variance[valid, None]).astype(np.float32)
    return out, valid


def readcurrent_inputs(adc):
    x = _windows(adc, np.int16).astype(np.float64)
    median = np.median(x, axis=1)
    deviation = x - median[:, None]
    mad = np.median(np.abs(deviation), axis=1)
    valid = np.isfinite(mad) & (mad > 1e-8)
    z = np.zeros_like(x)
    z[valid] = deviation[valid] / (1.4826 * mad[valid, None])
    for row in np.flatnonzero(valid):
        # Outlier indices are fixed first, then replaced in increasing index order.
        for i in np.flatnonzero(np.abs(z[row]) > 3.5):
            if i == 0:
                z[row, i] = z[row, 1]
            elif i == WINDOW - 1:
                z[row, i] = z[row, WINDOW - 2]
            else:
                z[row, i] = (z[row, i - 1] + z[row, i + 1]) / 2
    return z.astype(np.float32), valid


def cafes_features(adc, preprocessor=None):
    """Six-channel CAFES features as a float32 tensor [N, 6, 3000]."""
    raw = np.asarray(adc)
    if raw.ndim != 2 or raw.shape[1] != WINDOW or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError('CAFES expects raw integer ADC windows [N, 3000]')
    preprocessor = cafes_preprocessor() if preprocessor is None else preprocessor
    features = preprocessor.add_features(raw, norm=True, s_len=WINDOW, w_len=3).to(torch.float32)
    if features.shape != (len(raw), 6, WINDOW) or not torch.isfinite(features).all():
        raise FloatingPointError('invalid CAFES features')
    return features


def cafes_inputs(adc):
    adc = _windows(adc, np.int16)
    return cafes_features(adc).numpy(), np.ones(len(adc), bool)


_official_normalization = None


def squigglenet_normalization():
    """The official ``normalization`` function, extracted from the vendored inference.py.

    inference.py imports ont_fast5_api and click at module level, so only this
    function is compiled. Note the official MAD is already scaled by 1.4826 and is
    multiplied by 1.4826 again.
    """
    global _official_normalization
    if _official_normalization is None:
        path = SQUIGGLENET_ROOT / 'inference.py'
        tree = ast.parse(path.read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'normalization']
        if len(functions) != 1:
            raise ValueError('official normalization function missing or duplicated')
        namespace = {'np': np, 'stats': stats, 'torch': torch, 'print': lambda *args, **kwargs: None}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
        _official_normalization = namespace['normalization']
    return _official_normalization


def _squigglenet_normalization_any(data):
    """Upstream SquiggleNet normalization (third_party/SquiggleNet/inference.py) with its hardcoded last index 2999
    generalized to the window length; same arithmetic and sequential outlier replacement order."""
    mad = stats.median_abs_deviation(data, axis=1, scale='normal')
    data = ((data - np.median(data, axis=1)[:, None])*1.0) / (1.4826 * mad[:, None])
    last = data.shape[1] - 1
    rows, cols = np.where(np.abs(data) > 3.5)
    for r, c in zip(rows, cols):
        if c == 0: data[r, c] = data[r, c + 1]
        elif c == last: data[r, c] = data[r, c - 1]
        else: data[r, c] = (data[r, c - 1] + data[r, c + 1])/2
    return torch.tensor(data).float().numpy()


def squigglenet_inputs(pa):
    values = np.array(_windows(pa, np.float32), copy=True)
    with np.errstate(all='ignore'):
        mad = stats.median_abs_deviation(values, axis=1, scale='normal')
    valid = np.isfinite(mad) & (mad > 0)
    out = np.zeros(values.shape, np.float32)
    if valid.any():
        normalized = (squigglenet_normalization()(values[valid], 0).numpy() if values.shape[1] == 3000
                      else _squigglenet_normalization_any(values[valid]))
        finite = np.isfinite(normalized).all(axis=1)
        rows = np.flatnonzero(valid)
        out[rows[finite]] = normalized[finite]
        valid[rows[~finite]] = False
    return out, valid


def prepare_windows(windows, tool):
    """Model-ready float32 inputs and a validity mask for one batch of native windows."""
    if tool == 'nanodeep':
        x, valid = nanodeep_inputs(windows)
        return x[:, None, :], valid
    if tool == 'readcurrent':
        return readcurrent_inputs(windows)
    if tool == 'cafes':
        return cafes_inputs(windows)
    if tool == 'squigglenet':
        return squigglenet_inputs(windows)
    raise ValueError(f'unknown baseline tool: {tool}')


def host_margin(logits, tool):
    """Host logit minus microbe logit; > 0 is the tool's argmax host (reject) call."""
    value = np.asarray(logits, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError('two logits per window required')
    if not np.isfinite(value).all():
        raise FloatingPointError('nonfinite logits')
    contract = CONTRACTS[tool]
    return value[:, contract['host_index']] - value[:, contract['microbe_index']]
