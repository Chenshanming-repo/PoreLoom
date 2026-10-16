"""Shared balanced sampling, causal cost supervision and policy checkpoint training."""
from pathlib import Path
import copy,json,time
import numpy as np
import torch
from torch import nn
from .contracts import Action
from .costs import CostConfig,TimingConfig
from .replay import ReplayEnv
from .controllers import PolicyController,ValueHead,masked_softmax
from .model import prefix_loss
from .train_softstop import exact_expected_cost
from .train_pg import episode_policy_loss


def branch_cost_table(data,config,return_details=False):
    """A/R branches and natural/fallback mass after the last reachable decision."""
    n,K=data.valid.shape
    costs=np.zeros((n,K,2),np.float32);available=np.zeros((n,K),bool);end=np.zeros(n,np.float32)
    details={key:np.zeros((n,K,2),np.float32) for key in ('effective_reject','command_reject','pore_seconds','compute_seconds')}
    end_details={key:np.zeros(n,np.float32) for key in details}
    timing=TimingConfig(**config['timing']);cost=CostConfig(**config['cost'])
    for i,record in enumerate(data.records()):
        env=ReplayEnv(record,timing,cost,K);obs=env.reset()
        while obs is not None:
            k=obs.k-1;available[i,k]=True
            for a in (Action.ACCEPT,Action.REJECT):
                branch=copy.copy(env);_,out=branch.step(a);costs[i,k,int(a)]=out.total_cost
                for key in details:details[key][i,k,int(a)]=getattr(out,key)
            if obs.k==K:
                end[i]=costs[i,k,0]
                for key in details:end_details[key][i]=details[key][i,k,0]
                break
            obs,out=env.step(Action.DEFER)
        if env.outcome is not None:
            end[i]=env.outcome.total_cost
            for key in details:end_details[key][i]=getattr(env.outcome,key)
    return (costs,available,end,details,end_details) if return_details else (costs,available,end)


def sample_balanced_ids(is_target,eligible,batch_size,rng):
    host=eligible[~is_target[eligible]];target=eligible[is_target[eligible]]
    if not len(host) or not len(target):raise ValueError('balanced training requires both classes')
    half=batch_size//2
    for _ in range((len(eligible)+batch_size-1)//batch_size):
        ids=np.concatenate([rng.choice(host,half,replace=True),rng.choice(target,batch_size-half,replace=True)])
        rng.shuffle(ids);yield ids,(batch_size-half)/batch_size


def selection_key(metrics,config):
    frr=metrics['target_effective_frr'];cost=metrics['weighted_cost'];limit=config.get('target_frr_limit')
    return (0,cost) if limit is None or frr<=limit else (1,frr,cost)


def pg_batch(model,controller,value,e,available,costs,end,cost_scale,entropy_weight):
    """Only live episode members update GRU or visit actor/value states."""
    B,K,_=e.shape;h=e.new_zeros((B,128));alive=available[:,0].clone()
    logp=[[] for _ in range(B)];values=[[] for _ in range(B)];ents=[[] for _ in range(B)]
    terminal=end.clone()
    for k in range(K):
        ids=alive.nonzero(as_tuple=True)[0]
        if not len(ids):break
        state,updated=model.step_state(e[ids,k],h[ids],k+1)
        # Functional update preserves the history graph for visited steps.
        h=h.index_copy(0,ids,updated)
        p=masked_softmax(controller.score(state,k+1),k=k+1,max_windows=K)
        distribution=torch.distributions.Categorical(probs=p);a=distribution.sample()
        lp=distribution.log_prob(a);v=value(state);ent=distribution.entropy()
        for j,i in enumerate(ids.tolist()):logp[i].append(lp[j]);values[i].append(v[j]);ents[i].append(ent[j])
        stops=a<2
        terminal[ids[stops]]=costs[ids[stops],k,a[stops]]
        next_alive=torch.zeros_like(alive)
        if k+1<K:next_alive[ids[~stops]]=available[ids[~stops],k+1]
        alive=next_alive
    losses=[]
    for i in range(B):
        if logp[i]:losses.append(episode_policy_loss(torch.stack(logp[i]),torch.stack(values[i]),torch.stack(ents[i]),-terminal[i]/cost_scale,entropy_weight))
        else:losses.append(e[i].sum()*0)
    return torch.stack(losses)


def train_policy(variant,model,train_data,dev_data,config,output,device='cpu',resume=False):
    if config.get('objective') == 'class_conditional_risk':
        from ._risk_policy_training import train_risk_policy
        return train_risk_policy(variant,model,train_data,dev_data,config,output,device,resume)
    from .control_evaluation import evaluate_policy
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    seed=config.get('seed',20261009);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=model.to(device);model.freeze_encoder()
    controller=PolicyController(max_windows=config['max_windows']).to(device)
    value=ValueHead().to(device) if variant=='pg' else None
    setting=config['training'][variant];params=list(model.gru.parameters())+list(model.head.parameters())+list(controller.parameters())+(list(value.parameters()) if value is not None else [])
    optimizer=torch.optim.AdamW(params,lr=setting['lr'],weight_decay=1e-4)
    cfg=dict(config,variant=variant);(output/'config.json').write_text(json.dumps(cfg,indent=2))
    table,reachable,end=branch_cost_table(train_data,config)
    eligible=np.flatnonzero(train_data.valid.all(1)&reachable.all(1))
    if not len(eligible):raise ValueError('no full-budget training episodes')
    print(f'{variant}: {len(eligible)} full-budget training reads',flush=True)
    history=[];start_epoch=1;best_key=None;stale=0;started=time.time()
    if resume and (output/'last.pt').exists():
        cp=torch.load(output/'last.pt',map_location=device);model.load_state_dict(cp['model_state']);controller.load_state_dict(cp['controller_state']);optimizer.load_state_dict(cp['optimizer_state'])
        if value is not None:value.load_state_dict(cp['value_state'])
        history=cp['history'];start_epoch=cp['epoch']+1;best_key=tuple(cp['best_key']);stale=cp['stale'];rng.bit_generator.state=cp['numpy_rng'];torch.set_rng_state(cp['torch_rng'].cpu())
        if str(device).startswith('cuda') and 'cuda_rng' in cp:torch.cuda.set_rng_state(cp['cuda_rng'],device)
    for epoch in range(start_epoch,setting['epochs']+1):
        model.train();controller.train()
        if value is not None:value.train()
        total=0.;count=0
        entropy_weight=.01*max(0.,1-(epoch-1)/4)
        for ids,q_target in sample_balanced_ids(train_data.is_target,eligible,setting['batch_size'],rng):
            e=torch.as_tensor(np.asarray(train_data.features[ids]).copy(),device=device)
            valid=torch.as_tensor(train_data.valid[ids],device=device)
            y=torch.as_tensor(train_data.is_target[ids].astype(np.int64),device=device)
            weights=torch.where(y.bool(),config['rho_target']/q_target,(1-config['rho_target'])/(1-q_target))
            available=torch.as_tensor(reachable[ids],device=device);J=torch.as_tensor(table[ids],device=device);Jend=torch.as_tensor(end[ids],device=device)
            optimizer.zero_grad()
            # Auxiliary all-prefix CE is separate from the sampled actor record.
            states=model.forward_states(e,valid);ce=prefix_loss(model.head(states),y,valid)
            if variant=='softstop':
                expected,_=exact_expected_cost(controller(states),J,available,Jend,config['cost']['cost_scale']);objective=(weights*expected).mean()
            else:
                episode_losses=pg_batch(model,controller,value,e,available,J,Jend,config['cost']['cost_scale'],entropy_weight);objective=(weights*episode_losses).mean()
            loss=objective+config.get('auxiliary_weight',.1)*ce;loss.backward();nn.utils.clip_grad_norm_(params,1.);optimizer.step()
            total+=loss.item()*len(ids);count+=len(ids)
        model.eval();controller.eval()
        metrics=evaluate_policy(model,controller,dev_data,cfg,device=device,mode=cfg.get('deployment_mode','argmax'))
        key=selection_key(metrics,cfg)
        row=dict(epoch=epoch,train_loss=total/count,dev=metrics,entropy_weight=entropy_weight if variant=='pg' else None,elapsed_seconds=time.time()-started);history.append(row);print(row,flush=True)
        state=dict(model_state=model.state_dict(),controller_state=controller.state_dict(),config=cfg,variant=variant,epoch=epoch,dev_metrics=metrics)
        if value is not None:state['value_state']=value.state_dict()
        if best_key is None or key<best_key:best_key=key;stale=0;torch.save(state,output/'best.pt')
        else:stale+=1
        (output/'history.json').write_text(json.dumps(history,indent=2))
        state.update(optimizer_state=optimizer.state_dict(),history=history,best_key=best_key,stale=stale,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state())
        if str(device).startswith('cuda'):state['cuda_rng']=torch.cuda.get_rng_state(device)
        torch.save(state,output/'last.pt')
        if stale>=setting.get('patience',3):break
    return output/'best.pt'
