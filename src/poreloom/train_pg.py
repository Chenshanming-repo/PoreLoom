"""Visited-trajectory REINFORCE with a state-only value baseline."""
import torch


def episode_policy_loss(log_probs,values,entropies,return_value,entropy_weight=.01):
    advantage=(return_value-values).detach()
    actor=-(log_probs*advantage).sum()
    critic=((values-return_value.detach())**2).sum()
    return actor+.5*critic-entropy_weight*entropies.sum()


def train_pg(model,train_data,dev_data,config,output,device='cpu',resume=False):
    from ._policy_training import train_policy
    return train_policy('pg',model,train_data,dev_data,config,output,device,resume)
