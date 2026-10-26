"""Class-balanced sampling shared by streaming encoder training."""
import numpy as np


def balanced_indices(is_target,batch_size,rng):
    target=np.flatnonzero(is_target);host=np.flatnonzero(~np.asarray(is_target,dtype=bool))
    if not len(target) or not len(host):raise ValueError('balanced sampler requires both classes')
    n=batch_size//2
    return np.concatenate([rng.choice(target,n,replace=True),rng.choice(host,batch_size-n,replace=True)])
