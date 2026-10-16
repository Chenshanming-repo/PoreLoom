"""Exact scalar final-threshold sweeps after fixed causal prefix decisions."""
import numpy as np
from .hostfloor import hard_components, component_metrics


def exact_final_sweep(probability, actions, reachable, branches, ends, is_target):
    probability=np.asarray(probability,dtype=np.float32)
    actions=np.asarray(actions);reachable=np.asarray(reachable,dtype=bool)
    target=np.asarray(is_target,dtype=bool)
    if probability.shape != (len(actions),) or actions.shape != reachable.shape:
        raise ValueError('matching scalar probabilities and prefix geometry required')
    if not np.isfinite(probability).all() or (probability<0).any() or (probability>1).any():
        raise ValueError('probabilities must be finite in [0,1]')
    if not np.all(actions[:,-1]==0):raise ValueError('base final action must be ACCEPT')
    base=hard_components(actions,reachable,branches,ends)
    alive=np.ones(len(actions),bool)
    for k in range(actions.shape[1]-1):
        alive &= reachable[:,k]
        alive &= actions[:,k]==2
    ids=np.flatnonzero(alive & reachable[:,-1])
    ordered=ids[np.argsort(probability[ids],kind='stable')]
    thresholds=np.unique(np.r_[np.float32(0),probability[ordered],np.float32(1)]).astype(np.float32)
    counts=np.searchsorted(probability[ordered],thresholds,side='left')
    last=actions.shape[1]-1
    conditions=dict(target_command_frr=('command_reject',target),target_effective_frr=('effective_reject',target),
        host_effective_reject_rate=('effective_reject',~target),host_mean_pore_seconds=('pore_seconds',~target),
        mean_compute_seconds=('compute_seconds',np.ones(len(target),bool)),
        host_mean_compute_seconds=('compute_seconds',~target),target_mean_compute_seconds=('compute_seconds',target))
    summary=component_metrics(base,target)
    metrics={k:np.full(len(thresholds),summary[k]) for k in ('n_reads','n_host','n_target')}
    for name,(field,condition) in conditions.items():
        delta=(branches[field][ordered,last,1].astype(np.float64)-branches[field][ordered,last,0].astype(np.float64))*condition[ordered]
        cumulative=np.r_[0.,np.cumsum(delta)]
        numerator=np.asarray(base[field],dtype=np.float64)[condition].sum()
        metrics[name]=(numerator+cumulative[counts])/condition.sum()
    return dict(thresholds=thresholds,metrics=metrics,remaining_reads=len(ids))
