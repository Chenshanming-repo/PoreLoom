"""Backward component regression with fixed-dual rounds and fresh future labels."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .contracts import Action
from .controllers import FactorizedCostController, legal_action_mask
from .costs import CostConfig, TimingConfig
from .train_cost import _states, _nodes, _control_state


def outcome_components(outcome, rho, host_time_ref, compute_time_ref, event='command_reject'):
    target=outcome.is_target
    return [0. if target else outcome.pore_seconds/host_time_ref,
            float(target and getattr(outcome,event)),
            (rho if target else 1-rho)*outcome.compute_seconds/compute_time_ref]


def _batch_continue(branches, states, controller, hop, K):
    live=[(env,row) for env,row in branches if env.observation is not None]
    with torch.no_grad():
        while live:
            groups={}
            for env,row in live:groups.setdefault(env.observation.k,[]).append((env,row))
            live=[]
            for k,nodes in groups.items():
                for start in range(0,len(nodes),256):
                    batch=nodes[start:start+256]
                    x=torch.cat([_control_state(states,row,env.observation,hop) for env,row in batch])
                    scores=controller.score(x,k)
                    actions=scores.masked_fill(~legal_action_mask(k,K,controller.n_actions,scores.device),float('inf')).argmin(-1).cpu().tolist()
                    for (env,row),action in zip(batch,actions):
                        obs,_=env.step(Action(action))
                        if obs is not None:live.append((env,row))
    return [env.outcome for env,_ in branches]


def _examples(nodes,states,controller,hop,rho,records,K,host_time_ref,compute_time_ref,event='command_reject'):
    counts={}
    for row,_ in nodes:counts[row]=counts.get(row,0)+1
    if not counts:return None
    q=float(np.mean([records[row].is_target for row in counts]))
    xs=[];ys=[];weights=[];branches=[];destinations=[]
    for row,env in nodes:
        obs=env.observation;labels=[]
        for a in range(controller.n_actions):
            if not legal_action_mask(obs.k,K,controller.n_actions)[a]:continue
            branch=copy.copy(env);branch.step(Action(a))
            if branch.outcome is None:
                destinations.append((len(ys),len(labels)));branches.append((branch,row));labels.append([0.,0.,0.])
            else:labels.append(outcome_components(branch.outcome,rho,host_time_ref,compute_time_ref,event))
        xs.append(_control_state(states,row,obs,hop)[0]);ys.append(labels)
        w=.5/q if records[row].is_target else .5/(1-q)
        weights.append(w/counts[row])
    for (node,action),outcome in zip(destinations,_batch_continue(branches,states,controller,hop,K)):
        ys[node][action]=outcome_components(outcome,rho,host_time_ref,compute_time_ref,event)
    return torch.stack(xs),states.new_tensor(ys),states.new_tensor(weights)*(len(nodes)/len(counts))


def component_loss(controller,x,y,w,k):
    raw=controller.raw_components(x,k)[:,:y.shape[1]]
    squared=(F.softplus(raw[...,0])-y[...,0]).square()+(F.softplus(raw[...,2])-y[...,2]).square()
    risk=F.binary_cross_entropy_with_logits(raw[...,1],y[...,1],reduction='none')
    risk=risk*raw.new_tensor([0.]+[1.]*(y.shape[1]-1))
    return ((squared+risk).mean(-1)*w).mean()


def initialize_factorized(controller, checkpoint):
    """Expand a V1 head into Hop: zero runtime columns, copy short into long."""
    source=checkpoint['controller_state'];dest=controller.state_dict()
    if checkpoint.get('controller_type')!='factorized':raise ValueError('initial controller must be factorized')
    for name,value in source.items():
        if name not in dest:continue
        target=dest[name]
        if target.shape==value.shape:target.copy_(value)
        elif name.endswith('0.weight') and target.shape[0]==value.shape[0] and target.shape[1]>=value.shape[1]:
            target.zero_();target[:,:value.shape[1]].copy_(value)
        elif name.endswith(('2.weight','2.bias')) and target.shape[0]==12 and value.shape[0]==9:
            target[:9].copy_(value);target[9:12].copy_(value[6:9])
        else:raise ValueError(f'incompatible initial component head: {name}')
    controller.load_state_dict(dest)


def train_factorized(model,train_data,dev_data,config,output,device='cpu',resume=False,hop=False,initial_controller_checkpoint=None):
    from .control_evaluation import evaluate_policy
    from .risk_training import annotate_metrics, selection_key, update_beta
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    variant='hop' if hop else 'cost';K=int(config.get('max_windows',4))
    cfg=config.get('training',{}).get(variant,{})
    risk=config.get('risk',{});resource=config.get('resource',{})
    rho=float(config.get('rho_target',.01));event=risk.get('event','command_reject')
    if event not in ('command_reject','effective_reject'):raise ValueError('invalid risk event')
    records=[list(train_data.records()),list(dev_data.records())]
    ref=resource.get('host_time_ref','train_host_mean_natural_seconds')
    if isinstance(ref,str):
        if ref!='train_host_mean_natural_seconds':raise ValueError('unknown host_time_ref')
        ref=float(np.mean([r.raw_length/r.sample_rate_hz for r in records[0] if not r.is_target]))
    ref=float(ref)
    cref=float(resource.get('compute_time_ref',.03));mu=float(resource.get('compute_weight',.01))
    if not np.isfinite(ref) or ref<=0 or not np.isfinite(cref) or cref<=0:raise ValueError('finite positive resource references required')
    if not np.isfinite(rho) or not 0<=rho<=1:raise ValueError('rho_target must be a probability')
    config=copy.deepcopy(config);config.setdefault('resource',{})['host_time_ref']=ref
    torch.manual_seed(int(config.get('seed',0)));model.to(device).eval()
    for p in model.parameters():p.requires_grad_(False)
    controller=FactorizedCostController(K,259 if hop else 257,4 if hop else 3,risk.get('dual_initial',5.),mu).to(device)
    # controller_checkpoint belongs to the policy teacher, not these component heads.
    initial=initial_controller_checkpoint or config.get('initial_controller_checkpoint')
    if initial:
        saved=torch.load(initial,map_location=device) if isinstance(initial,(str,Path)) else initial
        initialize_factorized(controller,saved)
        controller.beta=float(saved.get('beta',controller.beta))
        if 'model_state' in saved:model.load_state_dict(saved['model_state'])
    history=[];start_round=0;best_key=None;best_feasible=None;progress=output/'resume.pt'
    if resume and progress.exists():
        saved=torch.load(progress,map_location=device)
        if saved['config']!=config:raise ValueError('resume configuration differs')
        model.load_state_dict(saved['model_state']);controller.load_state_dict(saved['controller_state'])
        controller.beta=saved['next_beta'];start_round=saved['round']+1;history=saved['history']
        best_key=saved.get('best_key');best_feasible=saved.get('best_feasible_key')
    states=[_states(model,d,device,K) for d in (train_data,dev_data)]
    groups=[_nodes(rs,TimingConfig(**config.get('timing',{})),CostConfig(**config.get('cost',{})),K,hop) for rs in records]
    rounds=min(3,int(cfg.get('outer_rounds',cfg.get('rounds',3))));epochs=int(cfg.get('epochs',20));batch=int(cfg.get('batch_size',config.get('batch_size',128)))
    if min(rounds,epochs,batch)<1:raise ValueError('positive rounds, epochs and batch size required')
    for round_id in range(start_round,rounds):
        # Labels are intentionally regenerated for every head in every dual round.
        fixed_beta=controller.beta
        for k in range(K,0,-1):
            examples=[_examples(g[k],s,controller,hop,rho,rs,K,ref,cref,event) for g,s,rs in zip(groups,states,records)]
            if examples[0] is None:continue
            head=controller.heads[k-1]
            optimizer=torch.optim.AdamW(head.parameters(),lr=cfg.get('lr',config.get('head_lr',1e-4)),weight_decay=cfg.get('weight_decay',1e-4))
            best=float('inf');best_state=copy.deepcopy(head.state_dict());stale=0
            for epoch in range(epochs):
                x,y,w=examples[0];total=0.
                for indices in torch.randperm(len(x),device=device).split(batch):
                    loss=component_loss(controller,x[indices],y[indices],w[indices],k)
                    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),cfg.get('gradient_clip',1.));optimizer.step();total+=float(loss.detach())*len(indices)
                with torch.no_grad():
                    val=float(component_loss(controller,*examples[1],k)) if examples[1] is not None else total/len(x)
                history.append(dict(round=round_id+1,head=k,epoch=epoch+1,beta=fixed_beta,train_component_loss=total/len(x),dev_component_loss=val))
                if val<best:best=val;best_state=copy.deepcopy(head.state_dict());stale=0
                else:stale+=1
                if stale>=int(cfg.get('patience',3)):break
            head.load_state_dict(best_state)
        train_metrics=evaluate_policy(model,controller,train_data,config,device,states=states[0])
        dev_metrics=evaluate_policy(model,controller,dev_data,config,device,states=states[1])
        train_metrics=annotate_metrics(train_metrics,config);dev_metrics=annotate_metrics(dev_metrics,config)
        key=selection_key(dev_metrics,config)
        checkpoint=dict(model_state=model.state_dict(),controller_state=controller.state_dict(),controller_type='factorized',beta=fixed_beta,mu=mu,config=config,variant=variant,dev_metrics=dev_metrics)
        if best_key is None or key<tuple(best_key):
            best_key=key;torch.save(checkpoint,output/'best_diagnostic.pt')
        if dev_metrics.get('constraint_met') is True and (best_feasible is None or key<tuple(best_feasible)):
            best_feasible=key;torch.save(checkpoint,output/'best_feasible.pt')
        history.append(dict(round=round_id+1,beta=fixed_beta,train_hard=train_metrics,dev_hard=dev_metrics))
        print(f"factorized {variant} round={round_id+1} beta={fixed_beta:.6g} dev_command_risk={dev_metrics['target_command_frr']:.6g} dev_effective_risk={dev_metrics['target_effective_frr']:.6g} resource={dev_metrics['resource_score']:.6g} feasible={dev_metrics['constraint_met']}",flush=True)
        controller.beta=update_beta(fixed_beta,train_metrics['target_command_frr' if event=='command_reject' else 'target_effective_frr'],config)
        torch.save(dict(checkpoint,round=round_id,next_beta=controller.beta,history=history,best_key=best_key,best_feasible_key=best_feasible),progress)
    (output/'config.json').write_text(json.dumps(config,indent=2));(output/'history.json').write_text(json.dumps(history,indent=2))
    if best_feasible is None:print('No feasible factorized checkpoint; returning best_diagnostic.pt (risk constraint not met).',flush=True)
    return output/('best_feasible.pt' if best_feasible is not None else 'best_diagnostic.pt')
