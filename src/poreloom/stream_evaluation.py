"""Classification metrics used during streaming encoder development."""
import numpy as np
from scipy.stats import rankdata


HOST_REJECT = (.90, .95, .99)


def auroc(score, is_target):
    ranks = rankdata(score); n_t = int(is_target.sum()); n_h = len(score) - n_t
    return float((ranks[is_target].sum() - n_t*(n_t + 1)/2)/(n_t*n_h))


def threshold_at_host_reject(p_target, is_target, host_reject):
    return np.quantile(p_target[~is_target], host_reject)


def frr_at_host_reject(p_target, is_target, host_reject=.95):
    """Target false-reject rate when the threshold rejects `host_reject` of hosts (reject if p_target < t)."""
    return float(np.mean(p_target[is_target] < threshold_at_host_reject(p_target, is_target, host_reject)))


def ece(p_target, is_target, bins=10):
    which = np.minimum((p_target*bins).astype(int), bins - 1)
    return float(sum(abs(p_target[which == b].mean() - is_target[which == b].mean())*np.mean(which == b)
                     for b in range(bins) if np.any(which == b)))


def summarize(p_target, is_target):
    ok = np.isfinite(p_target)
    p, y = p_target[ok], is_target[ok]
    out = dict(reads=int(len(p)), targets=int(y.sum()), auroc=auroc(p, y), ece=ece(p, y))
    out.update({f'frr_host{int(100*r)}': frr_at_host_reject(p, y, r) for r in HOST_REJECT})
    return out
