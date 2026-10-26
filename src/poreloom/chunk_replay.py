"""Offline decision replay at chunk arrivals, vectorized over reads.

Every read starts at t = 0. Chunks arrive at sample counts m_j = phase + j*chunk (phase in [0, chunk), both in
samples); at time m_j/fs the client holds floor(m_j) samples and may act. With chunk=None,
data counts as available exactly when it is needed. The read ends
naturally if it ends before a decision completes; a reject is effective if the eject completes before the natural
end, and the pore time is then decision + eject, otherwise the natural end. A service counts as compute if it
starts while the read is still in the pore.
"""
import numpy as np

from .contracts import Action
from .streaming import FS

NO_DECISION = -1


def available_at(needed, phase=0., chunk=None, fs=FS):
    """Time [s] at which `needed` samples are first held: the first arrival m_j >= needed."""
    needed = np.asarray(needed, np.float64)
    if chunk is None: return needed/fs
    j = np.maximum(0., np.ceil((needed - phase)/chunk))
    return (phase + j*chunk)/fs


def arrivals(phase, chunk, last, fs=FS):
    """Arrival sample counts [N,J] up to and including the first arrival >= last; later entries are nan."""
    phase = np.asarray(phase, np.float64)
    j_last = np.maximum(0., np.ceil((last - phase)/chunk)).astype(int)
    grid = phase[:, None] + chunk*np.arange(int(j_last.max()) + 1)
    return np.where(np.arange(grid.shape[1]) <= j_last[:, None], grid, np.nan)


def replay(times, actions, starts, costs, t_end, eject=.1):
    """Decision sequences -> outcome arrays.

    times [N,K]: completion time of step k (nan: no step); actions [N,K]: Action, or NO_DECISION for an update
    that cannot act; starts [N,K]: when its service starts; costs: service seconds ([N,K], [K] or scalar), or a
    tuple of consecutive service phases, each counted only if the read is still in the pore when it starts;
    t_end [N]: natural end. Returns a dict of [N] arrays: command, command_reject, effective_reject, pore_seconds, decision_seconds
    (nan without a command), compute_seconds, steps (index of the terminal step, -1 for a natural end).
    """
    times, actions, starts = np.asarray(times, float), np.asarray(actions), np.asarray(starts, float)
    phases = [np.broadcast_to(np.asarray(c, float), times.shape) for c in (costs if isinstance(costs, tuple) else (costs,))]
    t_end = np.asarray(t_end, float)
    n, k_max = times.shape
    alive = np.ones(n, bool)
    out = dict(command=np.zeros(n, bool), command_reject=np.zeros(n, bool), effective_reject=np.zeros(n, bool),
               pore_seconds=t_end.copy(), decision_seconds=np.full(n, np.nan), compute_seconds=np.zeros(n),
               steps=np.full(n, -1))
    for k in range(k_max):
        step = alive & np.isfinite(times[:, k])
        begin = starts[:, k].copy()
        for cost in phases:
            out['compute_seconds'] += np.where(step & (begin < t_end), cost[:, k], 0.); begin += cost[:, k]
        alive &= ~(step & (t_end <= times[:, k]))  # natural end before the decision completes
        act = alive & step & np.isin(actions[:, k], (Action.ACCEPT, Action.REJECT))
        reject = act & (actions[:, k] == Action.REJECT)
        effective = reject & (times[:, k] + eject < t_end)
        out['command'] |= act; out['command_reject'] |= reject; out['effective_reject'] |= effective
        out['decision_seconds'][act] = times[act, k]
        out['pore_seconds'][effective] = times[effective, k] + eject
        out['steps'][act] = k
        alive &= ~act
    return out


def threshold_actions(p, forced, reject_below, final_reject_below, accept_at=np.inf):
    """Threshold rule on p(target) [N,J] (nan: cannot act yet): reject if p < reject_below, accept if
    p >= accept_at, otherwise defer; on the forced step (forced [N,J]) reject if p < final_reject_below, otherwise
    accept. Thresholds are scalars or arrays broadcastable to p."""
    reject_below, final_reject_below = (np.broadcast_to(t, p.shape) for t in (reject_below, final_reject_below))
    actions = np.full(p.shape, int(Action.DEFER))
    actions[p < reject_below] = Action.REJECT
    actions[p >= accept_at] = Action.ACCEPT
    actions[forced] = np.where(p[forced] < final_reject_below[forced], int(Action.REJECT), int(Action.ACCEPT))
    actions[np.isnan(p) & ~forced] = NO_DECISION
    return actions


def calibrated_thresholds(used, calibrated_used, quantiles):
    """Per-look thresholds that calibrate a score drifting with n: quantiles [G] (e.g. the alpha quantile of
    target p(target) on calibration data) at calibrated stream lengths [G] (ascending); each look takes the
    entry at the nearest calibrated length <= its own `used` (clipped to the first)."""
    g = np.searchsorted(calibrated_used, used, 'right') - 1
    return np.asarray(quantiles)[np.clip(g, 0, len(calibrated_used) - 1)]


def summarize(outcome, is_target):
    """The simloop read-level metrics of one outcome dict."""
    target = np.asarray(is_target, bool)
    return dict(n_reads=int(len(target)), n_host=int((~target).sum()), n_target=int(target.sum()),
                host_effective_reject_rate=float(outcome['effective_reject'][~target].mean()),
                target_command_frr=float(outcome['command_reject'][target].mean()),
                target_effective_frr=float(outcome['effective_reject'][target].mean()),
                host_mean_pore_seconds=float(outcome['pore_seconds'][~target].mean()),
                mean_compute_seconds=float(outcome['compute_seconds'].mean()))
