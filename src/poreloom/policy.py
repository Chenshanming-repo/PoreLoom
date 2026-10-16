"""Fixed asymmetric score thresholds; class order host, target."""
import math


def decide(p_host,p_target,k,max_windows=8):
    if not 1<=k<=max_windows:raise ValueError('invalid window index')
    if not all(math.isfinite(p) and 0<=p<=1 for p in (p_host,p_target)) or abs(p_host+p_target-1)>1e-5:
        return 'accept','input_fallback'
    if p_host>=.99:return 'reject','confident_host'
    if p_target>=.90:return 'accept','confident_target'
    if k<max_windows:return 'defer','need_next_window'
    return 'accept','budget_accept'
