"""Backward cost regression with labels from the frozen causal future policy."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from .contracts import Action
from .controllers import CostController, choose_action, legal_action_mask
from .costs import CostConfig, TimingConfig
from .replay import ReplayEnv


def _control_state(states, row, obs, hop=False):
    state=states[row,obs.k-1:obs.k]
    if hop:
        state=torch.cat((state,state.new_tensor([[obs.pre_head_elapsed_seconds,obs.pre_head_compute_seconds]])),dim=-1)
    return state


def _continue(env, states, row, controller, hop=False):
    with torch.no_grad():
        while env.observation is not None:
            obs=env.observation
            scores=controller.score(_control_state(states,row,obs,hop),obs.k)
            env.step(choose_action(scores,'cost',obs.k,env.max_windows))
    return env.outcome.total_cost


def costs_if_accept_or_reject_at_k(reads,k,timing_config,cost_config,max_windows=None):
    """Direct terminal costs after the real sequential prefix (NaN if unreachable)."""
    result=[]
    for record in reads:
        env=ReplayEnv(record,timing_config,cost_config,max_windows);obs=env.reset()
        while obs is not None and obs.k<k:obs,_=env.step(Action.DEFER)
        if obs is None:result.append([float('nan')]*2);continue
        result.append([copy.copy(env).step(a)[1].total_cost for a in (Action.ACCEPT,Action.REJECT)])
    return torch.tensor(result,dtype=torch.float32)


def rollout_after_forced_defer(reads,k,states,controller,timing_config,cost_config,max_windows=None):
    """The learned future argmin, including its mistakes, determines D labels."""
    result=[]
    for row,record in enumerate(reads):
        env=ReplayEnv(record,timing_config,cost_config,max_windows);obs=env.reset()
        while obs is not None and obs.k<k:obs,_=env.step(Action.DEFER)
        if obs is None:result.append(float('nan'));continue
        env.step(Action.DEFER);result.append(_continue(env,states,row,controller))
    return torch.tensor(result,dtype=torch.float32)


def _states(model,data,device,K,batch_size=256):
    valid=torch.as_tensor(np.asarray(data.valid)[:,:K],device=device,dtype=torch.bool)
    chunks=[]
    with torch.no_grad():
        for start in range(0,len(valid),batch_size):
            stop=min(start+batch_size,len(valid))
            if data.features is not None:
                e=torch.as_tensor(np.asarray(data.features)[start:stop,:K],device=device,dtype=torch.float32)
            else:
                windows=torch.as_tensor(np.asarray(data.windows)[start:stop,:K],device=device,dtype=torch.float32)
                e=model.encode(windows.reshape(-1,1,windows.shape[-1])).reshape(stop-start,K,-1)
            chunks.append(model.forward_states(e,valid[start:stop]).detach())
    return torch.cat(chunks) if chunks else torch.empty((0,K,257),device=device)


def _nodes(records,timing,cost,K,hop=False):
    by_k={k:[] for k in range(1,K+1)}
    for row,record in enumerate(records):
        env=ReplayEnv(record,timing,cost,K);obs=env.reset()
        stack=[env] if obs is not None else []
        while stack:
            node=stack.pop();k=node.observation.k;by_k[k].append((row,node))
            for action in ([Action.DEFER,Action.DEFER_LONG] if hop else [Action.DEFER]):
                if legal_action_mask(k,K,4)[int(action)]:
                    child=copy.copy(node);obs,_=child.step(action)
                    if obs is not None:stack.append(child)
    return by_k


def _batch_continue(branches,states,controller,hop,K):
    """Batch neural scores for live future nodes; environment events stay per read."""
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
                    mask=legal_action_mask(k,K,scores.shape[-1],scores.device)
                    actions=scores.masked_fill(~mask,float('inf')).argmin(-1).cpu().tolist()
                    for (env,row),action in zip(batch,actions):
                        obs,_=env.step(Action(action))
                        if obs is not None:live.append((env,row))
    return [env.outcome.total_cost for env,_ in branches]


def _examples(nodes,states,controller,hop,cost_scale,rho,records,K):
    counts={}
    for row,_ in nodes:counts[row]=counts.get(row,0)+1
    # Uniform node sampling, with inverse multiplicity below, represents the
    # actually reachable read cohort at this head, not all original records.
    q=float(np.mean([records[row].is_target for row in counts])) if counts else .5
    xs=[];ys=[];weights=[];branches=[];destinations=[]
    for row,env in nodes:
        obs=env.observation;labels=[]
        for a in range(4 if hop else 3):
            if not legal_action_mask(obs.k,K,4 if hop else 3)[a]:continue
            branch=copy.copy(env);branch.step(Action(a))
            if branch.outcome is not None:labels.append(branch.outcome.total_cost/cost_scale)
            else:
                destinations.append((len(ys),len(labels)));branches.append((branch,row));labels.append(0.)
        xs.append(_control_state(states,row,obs,hop)[0]);ys.append(labels)
        target=records[row].is_target
        w=rho/q if target else (1-rho)/(1-q)
        weights.append(w/counts[row])
    if not xs:return None
    for (node,action),value in zip(destinations,_batch_continue(branches,states,controller,hop,K)):
        ys[node][action]=value/cost_scale
    # Node enumeration must preserve each read's total weight at this k.
    normalization=len(nodes)/len(counts)
    return torch.stack(xs),states.new_tensor(ys),states.new_tensor(weights)*normalization


def _metric_key(metrics,limit):
    frr=metrics['target_effective_frr'];cost=metrics['weighted_cost']
    return (0,cost) if limit is None or frr<=limit else (1,frr,cost)


def _train(model,train_data,dev_data,config,output,device,resume=False,hop=False):
    variant='hop' if hop else 'cost';output=Path(output);output.mkdir(parents=True,exist_ok=True)
    K=int(config.get('max_windows',4));seed=int(config.get('seed',0));torch.manual_seed(seed)
    model.to(device);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    controller=CostController(K,input_dim=259 if hop else 257,n_actions=4 if hop else 3).to(device)
    cfg=config.get('training',{}).get(variant,{})
    epochs=int(cfg.get('epochs',20));batch=int(cfg.get('batch_size',256));patience=int(cfg.get('patience',3))
    timing=TimingConfig(**config.get('timing',{}));cost=CostConfig(**config.get('cost',{}));rho=float(config.get('rho_target',.01))
    if not 0<=rho<=1:raise ValueError('rho_target must be a probability')
    if epochs<1 or batch<1:raise ValueError('positive epochs and batch_size required')
    records=[list(train_data.records()),list(dev_data.records())]
    history=[];completed=set()
    progress=output/'resume.pt'
    if resume and progress.exists():
        saved=torch.load(progress,map_location=device)
        if saved['config']!=config:raise ValueError('resume configuration differs')
        model.load_state_dict(saved['model_state']);controller.load_state_dict(saved['controller_state']);completed=set(saved['completed_heads']);history=saved['history']
    states=[_states(model,d,device,K) for d in (train_data,dev_data)]
    groups=[_nodes(rs,timing,cost,K,hop) for rs in records]
    # Initial fitting uses complete K-window reads. Hard DEV replay below
    # still evaluates every original read, including short/fallback outcomes.
    for groups_for_split,data in zip(groups,(train_data,dev_data)):
        complete=np.asarray(data.valid)[:,:K].all(axis=1)
        for k in groups_for_split:
            groups_for_split[k]=[(row,env) for row,env in groups_for_split[k] if complete[row]]
    for k in range(K,0,-1):
        if k in completed:continue
        examples=[_examples(g[k],s,controller,hop,cost.cost_scale,rho,rs,K) for g,s,rs in zip(groups,states,records)]
        if examples[0] is None:raise ValueError(f'no reachable training nodes at window {k}')
        if examples[1] is None:raise ValueError(f'no reachable DEV nodes at window {k}')
        head=controller.heads[k-1];optimizer=torch.optim.AdamW(head.parameters(),lr=cfg.get('lr',3e-4),weight_decay=cfg.get('weight_decay',1e-4))
        best=float('inf');best_state=copy.deepcopy(head.state_dict());stale=0
        for epoch in range(epochs):
            x,y,w=examples[0];order=torch.randperm(len(x),device=x.device);total=0.
            head.train()
            for indices in order.split(batch):
                prediction=head(x[indices])[:,:y.shape[1]]
                loss=(((prediction-y[indices])**2).mean(-1)*w[indices]).mean()
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),cfg.get('gradient_clip',1.));optimizer.step();total+=loss.item()*len(indices)
            head.eval()
            with torch.no_grad():
                dx,dy,dw=examples[1];dev=(((head(dx)[:,:dy.shape[1]]-dy)**2).mean(-1)*dw).mean().item()
            history.append({'head':k,'epoch':epoch+1,'train_mse':total/len(x),'dev_mse':dev})
            print(f'{variant} head={k} epoch={epoch+1} train_mse={total/len(x):.6g} dev_mse={dev:.6g}',flush=True)
            if dev<best:best=dev;best_state=copy.deepcopy(head.state_dict());stale=0
            else:stale+=1
            if stale>=patience:break
        head.load_state_dict(best_state);head.eval()
        for p in head.parameters():p.requires_grad_(False)
        completed.add(k)
        torch.save({'model_state':model.state_dict(),'controller_state':controller.state_dict(),'config':config,'completed_heads':sorted(completed),'history':history},progress)
    from .control_evaluation import evaluate_policy
    metrics=evaluate_policy(model,controller,dev_data,config,device=device,mode='argmax',states=states[1])
    metrics['constraint_met']=config.get('target_frr_limit') is None or metrics['target_effective_frr']<=config['target_frr_limit']
    history.append({'dev_hard':metrics})
    checkpoint={'model_state':model.state_dict(),'controller_state':controller.state_dict(),'config':config,'variant':variant,'dev_metrics':metrics}
    torch.save(checkpoint,output/'best.pt')
    (output/'config.json').write_text(json.dumps(config,indent=2))
    (output/'history.json').write_text(json.dumps(history,indent=2))
    return output/'best.pt'


def train_cost(model,train_data,dev_data,config,output,device='cpu',resume=False):
    if config.get('objective')=='class_conditional_risk':
        from .train_factorized import train_factorized
        return train_factorized(model,train_data,dev_data,config,output,device,resume)
    return _train(model,train_data,dev_data,config,output,device,resume,hop=False)
