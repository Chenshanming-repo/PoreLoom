"""Exact differentiable finite-horizon stopping, including environment termination."""
import torch
from .controllers import masked_softmax


def exact_expected_cost(logits, terminal_costs, available, end_cost, cost_scale=100.):
    """The reachable mask is environment supervision, never a policy input."""
    batch,windows,_=logits.shape
    survival=logits.new_ones(batch); expected=logits.new_zeros(batch);mass=logits.new_zeros(batch)
    for k in range(windows):
        alive=available[:,k].to(logits.dtype)
        ended=survival*(1-alive)
        expected=expected+ended*end_cost/cost_scale; mass=mass+ended
        survival=survival*alive
        legal=torch.ones_like(logits[:,k],dtype=torch.bool)
        if k==windows-1:legal[:,2]=False
        p=masked_softmax(logits[:,k],legal)
        expected=expected+survival*(p[:,:2]*terminal_costs[:,k]).sum(-1)/cost_scale
        mass=mass+survival*p[:,:2].sum(-1)
        survival=survival*p[:,2]
    expected=expected+survival*end_cost/cost_scale;mass=mass+survival
    return expected,mass


def train_softstop(model,train_data,dev_data,config,output,device='cpu',resume=False):
    from ._policy_training import train_policy
    return train_policy('softstop',model,train_data,dev_data,config,output,device,resume)
