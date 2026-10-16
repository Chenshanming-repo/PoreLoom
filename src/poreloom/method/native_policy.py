"""Native single-window baseline replay, metrics and DEV-B threshold selection.

Copied unchanged from experiments/20261013_nanodeep_baselines/policy.py; FS (Hz) is set by method.baselines to the
dataset rate before any replay.

Shared native operational policies and DEV-only threshold selection.

Host margin is oriented so larger values mean reject (host). Native argmax
semantics, including exact ties, come from score['native_reject']. Score IDs
must match the complete registered data order. Replay the full TEST group once,
then use metrics(..., rows=mask) for repeats so per-read phases stay identical.
Short and no-call reads remain in all operational metric denominators.
"""
import math

import numpy as np

FS = 4000.
PHASE_SEED = 20261012
ALL_KEEP = 'all_keep'


def phase_uniform(n, seed=PHASE_SEED):
    """V5b phases are drawn over the full group in its registered row order."""
    if int(n) != n or n < 0:
        raise ValueError('Phase count must be a nonnegative integer')
    return np.random.default_rng(seed).random(int(n))


def _vector(value, n, name):
    result = np.asarray(value)
    if result.shape != (n,):
        raise ValueError(f'{name} must have exactly one value per read')
    return result


def _booleans(value, n, name):
    result = _vector(value, n, name)
    if result.dtype.kind not in 'biuf' or not np.isin(result, [False, True]).all():
        raise ValueError(f'{name} must contain only booleans or 0/1')
    return result.astype(bool)


def _data(data):
    ids = np.asarray(data['read_id'])
    if ids.ndim != 1 or len(set(ids.tolist())) != len(ids):
        raise ValueError('Data read_id must be a unique ordered one-dimensional array')
    n = len(ids)
    raw = _vector(data['raw_length'], n, 'raw_length').astype(np.float64)
    if not np.isfinite(raw).all() or (raw < 0).any() or (raw % 1 != 0).any():
        raise ValueError('raw_length must contain nonnegative finite integer sample counts')
    if 'is_target' in data:
        target = _booleans(data['is_target'], n, 'is_target')
    else:
        labels = _vector(data['labels'], n, 'labels').astype(np.float64)
        if not np.isin(labels, [0, 1]).all():
            raise ValueError('labels must be 1=host, 0=target')
        target = labels == 0
    if 'labels' in data and 'is_target' in data:
        labels = _vector(data['labels'], n, 'labels').astype(np.float64)
        if not np.isin(labels, [0, 1]).all() or not np.array_equal(labels == 0, target):
            raise ValueError('labels and is_target disagree')
    return ids, raw, target


def _score(score, ids):
    n = len(ids)
    if 'read_id' not in score or not np.array_equal(np.asarray(score['read_id']), ids):
        raise ValueError('Score read_id must match registered data IDs and row order')
    margin = _vector(score['margin'], n, 'margin').astype(np.float64)
    native = _booleans(score['native_reject'], n, 'native_reject')
    eligible = np.isfinite(margin)
    if 'eligible' in score:
        eligible &= _booleans(score['eligible'], n, 'eligible')
    return margin, native, eligible


def _timing(raw, service, delta, phase_seed, eject, raw_end):
    parameters = [service, delta, eject, raw_end]
    if not all(math.isfinite(float(v)) for v in parameters) or service < 0 or delta <= 0 or eject < 0 or raw_end < 0:
        raise ValueError('Timing must be finite: service/eject/raw_end >= 0 and delta > 0')
    phase = phase_uniform(len(raw), phase_seed) * delta * FS
    chunk = delta * FS
    # Exactly available_at(raw_end, phase=phase, chunk=chunk) from chunk_replay.
    callback = (phase + np.maximum(0., np.ceil((raw_end - phase) / chunk)) * chunk) / FS
    completion = callback + service
    end = raw / FS
    alive = (raw >= raw_end) & (completion < end)
    return completion, end, alive


def native_policy(score, data, threshold=None, service=.0, delta=.4, phase_seed=PHASE_SEED, eject=.1, raw_end=4500):
    """One native prefix inference, followed by a keep/reject command if alive.

    threshold=None uses the model's original argmax class direction. A finite
    number rejects margin >= threshold. 'all_keep' keeps every read. Commands
    complete strictly before the natural end; ejection must also finish strictly
    before it to reduce pore time. Invalid/no-call scores cannot issue a command.
    decision_seconds is the completion time for keep or reject, NaN for no call.
    """
    ids, raw, _ = _data(data)
    margin, native, eligible = _score(score, ids)
    completion, end, alive = _timing(raw, service, delta, phase_seed, eject, raw_end)
    if threshold is None:
        requested = native
    elif isinstance(threshold, str) and threshold == ALL_KEEP:
        requested = np.zeros(len(ids), bool)
    else:
        try:
            threshold = float(threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError('Threshold must be None, finite numeric, or all_keep') from exc
        if not math.isfinite(threshold):
            raise ValueError('Numeric threshold must be finite')
        requested = margin >= threshold
    command = eligible & alive
    reject = requested & command
    effective = reject & (completion + eject < end)
    return dict(command=command, command_reject=reject, effective_reject=effective,
                pore_seconds=np.where(effective, completion + eject, end),
                decision_seconds=np.where(command, completion, np.nan))


def metrics(outcome, data, rows=None):
    """Operational metrics over all selected reads; no-call is a keep decision."""
    ids, raw, target = _data(data)
    n = len(ids)
    if 'read_id' in outcome and not np.array_equal(outcome['read_id'], ids):
        raise ValueError('Outcome read_id differs from data order')
    command = _booleans(outcome['command_reject'], n, 'command_reject')
    effective = _booleans(outcome['effective_reject'], n, 'effective_reject')
    pore = _vector(outcome['pore_seconds'], n, 'pore_seconds').astype(np.float64)
    if (effective & ~command).any() or not np.isfinite(pore).all() or (pore < 0).any() or (pore > raw / FS + 1e-12).any():
        raise ValueError('Invalid operational outcome')
    if rows is None:
        selected = np.arange(n)
    else:
        selected = np.asarray(rows)
        if selected.dtype.kind == 'b':
            if selected.shape != (n,):
                raise ValueError('Rows mask must have one entry per read')
        elif selected.ndim != 1 or selected.dtype.kind not in 'iu' or (selected < 0).any() or (selected >= n).any():
            raise ValueError('Rows must be a boolean mask or valid integer indices')
    target, command, effective, pore = target[selected], command[selected], effective[selected], pore[selected]
    host = ~target
    size, nh, nt = len(target), int(host.sum()), int(target.sum())
    def confusion(reject):
        return dict(host_reject=int((host & reject).sum()), host_keep=int((host & ~reject).sum()),
                    target_reject=int((target & reject).sum()), target_keep=int((target & ~reject).sum()))
    counts, effective_counts = confusion(command), confusion(effective)
    def rate(count, total):
        return count / total if total else None
    return dict(n=size, n_host=nh, n_target=nt, acc=rate(counts['host_reject'] + counts['target_keep'], size),
                effective_acc=rate(effective_counts['host_reject'] + effective_counts['target_keep'], size),
                host_command_reject_rate=rate(counts['host_reject'], nh),
                host_effective_reject_rate=rate(effective_counts['host_reject'], nh),
                target_command_frr=rate(counts['target_reject'], nt),
                target_effective_frr=rate(effective_counts['target_reject'], nt),
                host_mean_pore_seconds=float(pore[host].mean()) if nh else None,
                counts=counts, effective_counts=effective_counts)


def _require_dev(data, n):
    if 'split' not in data and 'role' not in data:
        raise ValueError('DEV-only selection requires explicit split/role provenance')
    if 'split' in data:
        split = np.asarray(data['split'])
        if not (split.shape == () or split.shape == (n,)) or not np.isin(split, ['dev_b']).all():
            raise ValueError('DEV-only selection requires split=dev_b')
    if 'role' in data:
        role = np.asarray(data['role'])
        if not (role.shape == () or role.shape == (n,)) or not np.isin(role, ['dev', 'dev_b']).all():
            raise ValueError('DEV-only selection rejects TRAIN/TEST or mixed roles')


def select_native(score, data, service, raw_end):
    """Select complete-tie thresholds on DEV, O(N log N), never on TEST.

    Max command ACC ties prefer lower target command FRR, then the more
    conservative threshold. OP1 maximizes host command rejection with target
    command FRR <= .005, then minimizes target FRR and prefers conservatism.
    Short/in-flight/no-call reads stay in the class denominators. All finite
    unique margins are candidates, including those belonging to no-command
    reads; this preserves the complete threshold grid and deterministic ties.
    """
    ids, raw, target = _data(data)
    _require_dev(data, len(ids))
    if not target.any() or target.all():
        raise ValueError('DEV selection requires both host and target classes')
    margin, _, eligible = _score(score, ids)
    _, _, alive = _timing(raw, service, .4, PHASE_SEED, .1, raw_end)
    finite = np.isfinite(margin)
    thresholds, bins = np.unique(margin[finite], return_inverse=True)
    valid = eligible & alive
    host_bin = np.bincount(bins, weights=(valid & ~target)[finite], minlength=len(thresholds)).astype(np.int64)
    target_bin = np.bincount(bins, weights=(valid & target)[finite], minlength=len(thresholds)).astype(np.int64)
    hosts = np.r_[host_bin[::-1].cumsum()[::-1], 0]
    targets = np.r_[target_bin[::-1].cumsum()[::-1], 0]
    conservative = np.r_[thresholds, np.inf]  # all_keep is the most conservative.
    nt = int(target.sum())
    correct = nt + hosts - targets
    max_index = int(np.lexsort((conservative, -targets, correct))[-1])
    feasible = targets <= nt // 200  # .005 = 1/200, with an exact integer cap.
    feasible_indices = np.flatnonzero(feasible)
    op_index = int(feasible_indices[np.lexsort((conservative[feasible], -targets[feasible], hosts[feasible]))[-1]])
    def choice(index):
        return ALL_KEEP if index == len(thresholds) else float(thresholds[index])
    result = {}
    for name, threshold in [('native', None), ('max_acc', choice(max_index)), ('op1', choice(op_index))]:
        out = native_policy(score, data, threshold=threshold, service=service, raw_end=raw_end)
        result[name] = dict(threshold=threshold, dev_metrics=metrics(out, data))
    return result
