"""Host-floor selection and paired read-level experimental analysis."""
import numpy as np


def hostfloor_key(metrics, host_floor):
    host = metrics['host_effective_reject_rate']
    tail = (metrics['target_command_frr'], metrics['host_mean_pore_seconds'], metrics['mean_compute_seconds'])
    return (0, *tail) if host >= host_floor else (1, host_floor - host, *tail)


def hard_components(actions, reachable, branches, ends):
    """Vectorized three-action replay using authenticated ReplayEnv branches."""
    actions = np.asarray(actions)
    if actions.shape != reachable.shape or not np.isin(actions, [0, 1, 2]).all():
        raise ValueError('three-action array must match reachable windows')
    result = {key: value.copy() for key, value in ends.items()}
    alive = np.ones(len(actions), dtype=bool)
    for k in range(actions.shape[1]):
        alive &= reachable[:, k]
        stopped = alive & (actions[:, k] < 2)
        ids = np.flatnonzero(stopped)
        for key in result:
            result[key][ids] = branches[key][ids, k, actions[ids, k]]
        alive &= ~stopped
    return result


def component_metrics(components, is_target):
    target = np.asarray(is_target, dtype=bool)
    if not target.any() or target.all():
        raise ValueError('both classes required')
    return dict(n_reads=len(target), n_host=int((~target).sum()), n_target=int(target.sum()),
                target_command_frr=float(components['command_reject'][target].mean()),
                target_effective_frr=float(components['effective_reject'][target].mean()),
                host_effective_reject_rate=float(components['effective_reject'][~target].mean()),
                host_mean_pore_seconds=float(components['pore_seconds'][~target].mean()),
                mean_compute_seconds=float(components['compute_seconds'].mean()),
                host_mean_compute_seconds=float(components['compute_seconds'][~target].mean()),
                target_mean_compute_seconds=float(components['compute_seconds'][target].mean()))


def _command(row):
    if 'command_reject' in row:
        return bool(row['command_reject'])
    return bool(row['command_issued'] and row.get('action') in (1, 'Action.REJECT', 'reject'))


def paired_summary(before, after, bootstrap=20000, seed=20261010, alpha=.05):
    """Paired multinomial bootstrap of {-1,0,+1} rate differences."""
    b = {r['read_id']: r for r in before}
    a = {r['read_id']: r for r in after}
    if len(b) != len(before) or len(a) != len(after) or b.keys() != a.keys():
        raise ValueError('identical unique read cohorts required')
    if any(bool(b[k]['is_target']) != bool(a[k]['is_target']) for k in b):
        raise ValueError('paired labels differ')
    rng = np.random.default_rng(seed)
    out = {'n_host': sum(not bool(r['is_target']) for r in before),
           'n_target': sum(bool(r['is_target']) for r in before), 'ci_method': 'paired multinomial percentile bootstrap',
           'bootstrap': bootstrap, 'alpha': alpha, 'seed': seed}
    for name, target, event in (('host', False, lambda r: bool(r['effective_reject'])),
                                ('target_command', True, _command),
                                ('target_effective', True, lambda r: bool(r['effective_reject']))):
        ids = [k for k in b if bool(b[k]['is_target']) == target]
        if not ids:
            raise ValueError('both classes required')
        gains = sum(not event(b[k]) and event(a[k]) for k in ids)
        losses = sum(event(b[k]) and not event(a[k]) for k in ids)
        n = len(ids)
        samples = rng.multinomial(n, [losses / n, (n - losses - gains) / n, gains / n], size=bootstrap)
        values = (samples[:, 2] - samples[:, 0]) / n
        out[name + '_delta'] = (gains - losses) / n
        out[name + '_delta_ci'] = np.quantile(values, [alpha / 2, 1 - alpha / 2]).tolist()
        if target:
            out[name + '_rescued'] = losses
            out[name + '_added'] = gains
        else:
            out['host_gained'], out['host_lost'] = gains, losses
    out['host_interval_degenerate'] = out['host_delta_ci'][0] == out['host_delta_ci'][1]
    out['host_noninferiority_delta0'] = not out['host_interval_degenerate'] and out['host_delta_ci'][0] >= 0
    out['target_command_interval_degenerate'] = out['target_command_delta_ci'][0] == out['target_command_delta_ci'][1]
    out['target_command_improvement'] = not out['target_command_interval_degenerate'] and out['target_command_delta_ci'][1] < 0
    return out
