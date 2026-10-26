"""Window-pretraining model container and shared feature/loss helpers."""
import torch
from torch import nn


class PoreLoom(nn.Module):
    def __init__(self,encoder,feature_dim=512,max_windows=8):
        super().__init__();self.encoder=encoder
        if not 1 <= max_windows <= 8: raise ValueError("max_windows must be in [1,8]")
        self.max_windows=max_windows
        self.projection=nn.Sequential(nn.Linear(feature_dim,128),nn.LayerNorm(128),nn.GELU())
        self.gru=nn.GRUCell(129,128)
        self.head=nn.Sequential(nn.Linear(257,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,2))
        self.encoder_frozen=False
    def encode(self,x):
        if x.ndim!=3 or x.shape[1:]!=(1,1500):raise ValueError('expected [B,1,1500]')
        return self.projection(self.encoder(x).mean(-1))
    def step_state(self,e,h,k):
        if not 1<=k<=self.max_windows:raise ValueError('window index outside budget')
        position=e.new_full((len(e),1),k/self.max_windows)
        next_h=self.gru(torch.cat([e,position],1),h)
        return torch.cat([next_h,e,position],1),next_h
    def step_features(self,e,h,k):
        state,h=self.step_state(e,h,k)
        return self.head(state),h
    def forward_states(self,e,valid):
        if e.ndim!=3 or e.shape[2]!=128 or valid.shape!=e.shape[:2] or not 1<=e.shape[1]<=self.max_windows:raise ValueError('invalid feature sequence')
        valid=valid.bool();h=e.new_zeros((len(e),128));out=[]
        for k in range(e.shape[1]):
            selected=valid[:,k];z=e.new_zeros((len(e),257));next_h=h.clone()
            if selected.any():
                state,updated=self.step_state(e[selected,k],h[selected],k+1)
                z[selected]=state;next_h[selected]=updated
            h=next_h;out.append(z)
        return torch.stack(out,1)
    def forward_features(self,e,valid):
        states=self.forward_states(e,valid)
        return self.head(states)*valid.to(e.dtype).unsqueeze(-1)
    def freeze_encoder(self):
        self.encoder_frozen=True
        for module in (self.encoder,self.projection):
            module.eval()
            for p in module.parameters():p.requires_grad_(False)
    def train(self,mode=True):
        super().train(mode)
        if self.encoder_frozen:self.encoder.eval();self.projection.eval()
        return self


def build_model(checkpoint=None,max_windows=8):
    # Vendored ReadCurrent architecture (poreloom.method.backbone, identical to third_party/ReadCurrent at 6db8921).
    from .method.backbone import window_classifier
    base=window_classifier()
    if checkpoint is not None:
        state=torch.load(checkpoint,map_location='cpu')
        state=state.get('model_state',state)
        base.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
    # Discard adaptive max pool and all original fully connected layers.
    return PoreLoom(nn.Sequential(*list(base.layers.children())[:-1]),512,max_windows=max_windows)


def prefix_loss(logits,labels,valid):
    if valid.shape!=logits.shape[:2] or labels.shape!=(len(logits),):raise ValueError('invalid prefix labels/mask')
    counts=valid.sum(1);active=counts>0
    if not active.any():raise ValueError('no valid read prefixes')
    loss=nn.functional.cross_entropy(logits[valid],labels[:,None].expand_as(valid)[valid],reduction='none')
    per_read=logits.new_zeros(len(logits));rows=valid.nonzero(as_tuple=True)[0]
    per_read.index_add_(0,rows,loss)
    return (per_read[active]/counts[active]).mean()
