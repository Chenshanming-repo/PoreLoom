"""Shared frozen-encoder adaptation, sampling weights, and checkpoint utilities."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.nn import functional as F
from .model import build_model,prefix_loss
from .data import load_dataset,file_sha256

def auxiliary_prefix_ce(logits,is_target,valid):return prefix_loss(logits,is_target.long(),valid.bool())
def read_weights(is_target,rho_target=.01,q_target=.5):
    if not 0<q_target<1 or not 0<=rho_target<=1:raise ValueError('invalid mixture')
    if isinstance(is_target,torch.Tensor):return torch.where(is_target.bool(),rho_target/q_target,(1-rho_target)/(1-q_target))
    return np.where(is_target,rho_target/q_target,(1-rho_target)/(1-q_target)).astype(np.float32)
sampler_weights=read_weights

def balanced_indices(is_target,batch_size,rng):
    target=np.flatnonzero(is_target);host=np.flatnonzero(~np.asarray(is_target,dtype=bool))
    if not len(target) or not len(host):raise ValueError('balanced sampler requires both classes')
    n=batch_size//2
    return np.concatenate([rng.choice(target,n,replace=True),rng.choice(host,batch_size-n,replace=True)])

def save_checkpoint(path,model,controller=None,**extra):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload={'model_state':model.state_dict(),'max_windows':model.max_windows,**extra}
    if controller is not None:payload['controller_state']=controller.state_dict()
    torch.save(payload,path)
    return path

def load_prefix_model(shared,device='cpu',max_windows=None):
    path=Path(shared);path=path/'prefix_best.pt' if path.is_dir() else path
    checkpoint=torch.load(path,map_location='cpu')
    model=build_model(max_windows=max_windows or checkpoint.get('max_windows',8))
    model.load_state_dict(checkpoint['model_state']);model.to(device);model.freeze_encoder()
    return model

def cache_states(model,dataset,batch_size=256,device=None):
    device=device or next(model.parameters()).device;model.eval();out=[]
    with torch.inference_mode():
        for start in range(0,len(dataset),batch_size):
            e=torch.as_tensor(np.array(dataset.features[start:start+batch_size]),device=device)
            valid=torch.as_tensor(np.array(dataset.valid[start:start+batch_size]),device=device)
            out.append(model.forward_states(e,valid).cpu().numpy())
    return np.concatenate(out)

def verify_feature_backbone(initial_checkpoint,feature_checkpoint):
    current=torch.load(initial_checkpoint,map_location='cpu')['model_state']
    cached=torch.load(feature_checkpoint,map_location='cpu')['model_state']
    keys=[k for k in current if k.startswith(('encoder.','projection.'))]
    if not keys or any(k not in cached or not torch.equal(current[k],cached[k]) for k in keys):
        raise ValueError('feature cache encoder/projection do not match initialization')
    return {'feature_checkpoint':str(feature_checkpoint),'feature_checkpoint_sha256':file_sha256(feature_checkpoint),'encoder_projection_identical':True}

def warmup_prefix_model(data_dir,initial_checkpoint,output,max_windows=4,device='cpu',epochs=5,batch_size=64,lr=3e-4,weight_decay=1e-4,patience=2,seed=0,resume=None,**kwargs):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=load_prefix_model(resume or initial_checkpoint,device,max_windows)
    train=load_dataset(data_dir,'train',max_windows);dev=load_dataset(data_dir,'dev',max_windows)
    if train.features is None or dev.features is None:raise ValueError('warmup requires common frozen encoder feature cache')
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=weight_decay)
    eligible=np.flatnonzero(train.complete);dev_ids=np.flatnonzero(dev.complete)
    if not len(eligible) or not len(dev_ids):raise ValueError('no complete prefixes')
    best=float('inf');stale=0;history=[];start_epoch=0
    if resume:
        restored=torch.load(resume,map_location=device)
        if 'optimizer_state' in restored:opt.load_state_dict(restored['optimizer_state'])
        start_epoch=restored.get('epoch',0);best=restored.get('dev_ce',best)
    provenance={'initial_checkpoint':str(initial_checkpoint),'initial_checkpoint_sha256':file_sha256(initial_checkpoint),'data_dir':str(Path(data_dir).resolve()),'max_windows':max_windows}
    feature_checkpoint=Path(data_dir).parent/'stage_A_best.pt'
    if feature_checkpoint.exists():provenance.update(verify_feature_backbone(initial_checkpoint,feature_checkpoint))
    else:provenance['encoder_projection_identity']='not independently verified; explicit feature provenance required'
    (output/'prefix_manifest.json').write_text(json.dumps(provenance,indent=2))
    for epoch in range(start_epoch,min(epochs,5)):
        model.train();rng.shuffle(eligible);total=0.;count=0
        for start in range(0,len(eligible),batch_size):
            ids=eligible[start:start+batch_size];e=torch.as_tensor(np.array(train.features[ids]),device=device);v=torch.as_tensor(np.array(train.valid[ids]),device=device);y=torch.as_tensor(train.labels[ids],device=device)
            opt.zero_grad();loss=auxiliary_prefix_ce(model.forward_features(e,v),y,v);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=loss.item()*len(ids);count+=len(ids)
        model.eval();dev_total=0.
        with torch.inference_mode():
            for start in range(0,len(dev_ids),batch_size):
                ids=dev_ids[start:start+batch_size];e=torch.as_tensor(np.array(dev.features[ids]),device=device);v=torch.as_tensor(np.array(dev.valid[ids]),device=device);y=torch.as_tensor(dev.labels[ids],device=device)
                dev_total+=auxiliary_prefix_ce(model.forward_features(e,v),y,v).item()*len(ids)
        score=dev_total/len(dev_ids);history.append({'epoch':epoch+1,'train_ce':total/count,'dev_ce':score})
        if score<best:
            best=score;stale=0;save_checkpoint(output/'prefix_best.pt',model,epoch=epoch+1,dev_ce=score,optimizer_state=opt.state_dict(),initial_checkpoint=str(initial_checkpoint))
        else:stale+=1
        (output/'warmup_history.json').write_text(json.dumps(history,indent=2))
        if stale>=patience:break
    return load_prefix_model(output if (output/'prefix_best.pt').exists() else resume,device,max_windows)
