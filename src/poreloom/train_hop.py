"""Path-dependent adaptive-hop backward cost regression."""
from .train_cost import _train


def train_hop(model,train_data,dev_data,config,output,device='cpu',resume=False):
    """Enumerate step-1/2 paths and retain per-read weight at each head."""
    if config.get('objective')=='class_conditional_risk':
        from .train_factorized import train_factorized
        return train_factorized(model,train_data,dev_data,config,output,device,resume,hop=True)
    return _train(model,train_data,dev_data,config,output,device,resume,hop=True)
