"""Action heads and legal action masking, independent of class label numbering."""
import torch
from torch import nn
from .contracts import Action

def _head(input_dim,n_actions):return nn.Sequential(nn.Linear(input_dim,64),nn.GELU(),nn.Linear(64,n_actions))
class CostController(nn.Module):
    score_kind='cost'
    def __init__(self,max_windows=4,input_dim=257,n_actions=3):
        super().__init__();self.max_windows=max_windows;self.n_actions=n_actions;self.heads=nn.ModuleList([_head(input_dim,n_actions) for _ in range(max_windows)])
    def score(self,state,k):return self.heads[k-1](state)
    def forward(self,state,k):return self.score(state,k)
class FactorizedCostController(CostController):
    """Predict resource components and an undiluted target rejection risk."""
    controller_type='factorized'
    def __init__(self,max_windows=4,input_dim=257,n_actions=3,beta=5.,mu=.01):
        super().__init__(max_windows,input_dim,n_actions)
        self.heads=nn.ModuleList([_head(input_dim,n_actions*3) for _ in range(max_windows)])
        self.beta=float(beta);self.mu=float(mu)
    def raw_components(self,state,k):
        return self.heads[k-1](state).reshape(*state.shape[:-1],self.n_actions,3)
    def components(self,state,k):
        raw=self.raw_components(state,k)
        u=torch.nn.functional.softplus(raw[...,0])
        r=raw[...,1].sigmoid()*raw.new_tensor([0.]+[1.]*(self.n_actions-1))
        c=torch.nn.functional.softplus(raw[...,2])
        return torch.stack((u,r,c),-1)
    def score(self,state,k):
        components=self.components(state,k)
        return components[...,0]+self.beta*components[...,1]+self.mu*components[...,2]

class PolicyController(nn.Module):
    score_kind='logits'
    def __init__(self,input_dim=257,n_actions=3,max_windows=4):
        super().__init__();self.max_windows=max_windows;self.n_actions=n_actions;self.head=_head(input_dim,n_actions)
    def score(self,state,k):return self.head(state)
    def forward(self,state,k=None):return self.head(state)
class ValueHead(nn.Module):
    def __init__(self,input_dim=257,n_actions=1):super().__init__();self.head=_head(input_dim,1)
    def forward(self,state):return self.head(state).squeeze(-1)

def legal_action_mask(k,max_windows,n_actions=3,device=None):
    if not 1<=k<=max_windows or n_actions not in (3,4):raise ValueError('invalid action budget')
    return torch.tensor([True,True,k<max_windows]+([k+2<=max_windows] if n_actions==4 else []),device=device)
def mask_scores(scores,score_kind,k,max_windows):
    return scores.masked_fill(~legal_action_mask(k,max_windows,scores.shape[-1],scores.device),float('inf') if score_kind=='cost' else -float('inf'))
def masked_softmax(logits,legal_actions=None,k=None,max_windows=None):
    if legal_actions is None:legal_actions=legal_action_mask(k,max_windows,logits.shape[-1],logits.device)
    return torch.softmax(logits.masked_fill(~torch.as_tensor(legal_actions,dtype=torch.bool,device=logits.device),-float('inf')),dim=-1)
def choose_action(scores,score_kind,k,max_windows,rng=None):
    scores=mask_scores(scores,score_kind,k,max_windows).reshape(-1,scores.shape[-1])
    if len(scores)!=1:raise ValueError('choose_action expects one read')
    if rng is not None and score_kind=='logits':
        p=torch.softmax(scores[0],-1)
        if isinstance(rng,torch.Generator): idx=torch.multinomial(p,1,generator=rng).item()
        else:idx=int(rng.choice(len(p),p=p.detach().cpu().numpy()))
    else:idx=(scores[0].argmin() if score_kind=='cost' else scores[0].argmax()).item()
    return Action(idx)
