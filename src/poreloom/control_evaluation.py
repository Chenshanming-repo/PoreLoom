"""Read-level causal replay, frozen rule controls, and complete-cohort outputs."""
import csv
import gzip
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .contracts import Action, Trace
from .controllers import choose_action, legal_action_mask, masked_softmax
from .costs import CostConfig, TimingConfig, terminal_cost
from .replay import ReplayEnv


def _configs(config):
    return TimingConfig(**config.get('timing',{})),CostConfig(**config.get('cost',config.get('costs',{})))


def prefix_scores(model,data,device='cpu',batch_size=256):
    """Encode states once; each state's recurrent input is its causal prefix."""
    model=model.to(device).eval();states=[];probs=[]
    with torch.inference_mode():
        for start in range(0,len(data.valid),batch_size):
            end=start+batch_size
            if data.features is not None:e=torch.as_tensor(data.features[start:end,:model.max_windows],device=device,dtype=torch.float32)
            else:
                x=torch.as_tensor(data.windows[start:end,:model.max_windows],device=device,dtype=torch.float32)
                e=model.encode(x.reshape(-1,1,x.shape[-1])).reshape(len(x),model.max_windows,-1)
            valid=torch.as_tensor(data.valid[start:end,:model.max_windows],device=device,dtype=torch.bool)
            s=model.forward_states(e,valid);states.append(s.cpu());probs.append(model.head(s).softmax(-1)[...,1].cpu())
    return torch.cat(states),torch.cat(probs)


def summarize_outcomes(outcomes, rho_target=.01, target_frr_limit=None):
    rows=[asdict(o) if not isinstance(o,dict) else o for o in outcomes]
    targets=[r for r in rows if r['is_target']];hosts=[r for r in rows if not r['is_target']]
    def avg(rs,key):return float(np.mean([r[key] for r in rs])) if rs else None
    def reject(r):return bool(r['command_issued'] and r.get('action') in (Action.REJECT,1,'reject'))
    fr=float(np.mean([r['effective_reject'] for r in targets])) if targets else None
    tc=avg(targets,'total_cost');hc=avg(hosts,'total_cost')
    weighted=(rho_target*(tc or 0)+(1-rho_target)*(hc or 0))
    metrics=dict(n_reads=len(rows),n_target=len(targets),n_host=len(hosts),mean_cost=avg(rows,'total_cost'),weighted_cost=weighted,
        rho_target=rho_target,target_command_frr=float(np.mean([reject(r) for r in targets])) if targets else None,
        target_effective_frr=fr,host_mean_pore_seconds=avg(hosts,'pore_seconds'),host_effective_reject_rate=avg(hosts,'effective_reject'),
        mean_compute_seconds=avg(rows,'compute_seconds'),host_mean_compute_seconds=avg(hosts,'compute_seconds'),
        target_mean_compute_seconds=avg(targets,'compute_seconds'),constraint_enabled=target_frr_limit is not None,
        constraint_met=None if target_frr_limit is None else (fr is not None and fr<=target_frr_limit),
        target_frr_limit=target_frr_limit,target_bp_per_allocated_pore_hour=None,
        yield_status='not_computed_no_bp_or_channel_time_inputs')
    for key in ('cnn_calls','gru_calls','head_calls'):metrics[key]=sum(r.get(key,0) for r in rows)
    metrics['reasons']={reason:sum(r['reason']==reason for r in rows) for reason in sorted({r['reason'] for r in rows})}
    for key in ('decision_seconds','effective_seconds'):
        vals=[r.get(key) for r in rows if r.get(key) is not None];metrics['mean_'+key]=float(np.mean(vals)) if vals else None
    return metrics


def _write_rows(path,rows):
    rows=list(rows)
    with gzip.open(path,'wt',newline='') as handle:
        keys=list(dict.fromkeys(key for row in rows for key in row))
        writer=csv.DictWriter(handle,keys);writer.writeheader()
        for row in rows:writer.writerow({key:json.dumps(v) if isinstance(v,(list,tuple,dict)) else v for key,v in row.items()})


def evaluate_policy(model,controller,data,config,device='cpu',output=None,mode='argmax',states=None,probabilities=None):
    if mode not in ('argmax','sample'):raise ValueError('invalid deployment mode')
    timing,cost=_configs(config);K=model.max_windows
    if states is None:states,probabilities=prefix_scores(model,data,device)
    if probabilities is None:
        with torch.inference_mode():probabilities=torch.cat([model.head(batch.to(device)).softmax(-1)[...,1].cpu() for batch in states.split(256)])
    controller=controller.to(device).eval();n_actions=controller.n_actions
    hop=n_actions==4
    score_cache=None
    if not hop:
        score_cache=torch.empty((len(states),K,n_actions))
        with torch.inference_mode():
            for start in range(0,len(states),256):
                for k in range(1,K+1):score_cache[start:start+256,k-1]=controller.score(states[start:start+256,k-1].to(device),k).cpu()
    rng=np.random.default_rng(config.get('seed',0));outcomes=[];decisions=[]
    records=list(data.records())
    if hop:
        outcomes,decisions=_batched_hop_replay(records,states,probabilities,controller,timing,cost,K,device,mode,rng,output is not None)
    for i,record in enumerate([] if hop else records):
        env=ReplayEnv(record,timing,cost,max_windows=K);obs=env.reset()
        while obs is not None:
            k=obs.k
            if hop:
                s=states[i,k-1:k].to(device)
                runtime=s.new_tensor([[obs.pre_head_elapsed_seconds,obs.pre_head_compute_seconds]])
                with torch.inference_mode():scores=controller.score(torch.cat([s,runtime],-1),k).cpu()
            else:scores=score_cache[i,k-1:k]
            action=choose_action(scores,controller.score_kind,k,K,rng if mode=='sample' else None)
            row={'action':action.name.lower()}
            if output is not None:
                row.update({key:value for key,value in vars(obs).items() if key!='window'})
                row.update(read_id=record.read_id,scores=scores.reshape(-1).tolist(),
                           legal_mask=legal_action_mask(k,K,n_actions).tolist(),auxiliary_target_probability=float(probabilities[i,k-1]))
                if controller.score_kind=='logits':row['probabilities']=masked_softmax(scores,k=k,max_windows=K).reshape(-1).tolist()
            decisions.append(row);obs,_=env.step(action)
        if hasattr(controller,'rule') and env.outcome.action is not None:env.outcome.reason=controller.rule+'_'+env.outcome.action.name.lower()
        outcomes.append(env.outcome)
    metrics=summarize_outcomes(outcomes,config.get('rho_target',.01),config.get('target_frr_limit'))
    metrics.update(mode=mode,observation_count=len(decisions),action_distribution={a:sum(r['action']==a for r in decisions) for a in ('accept','reject','defer','defer_long')},
                   timing_source=timing.source,comparison_scope='matched_1500_sample_windows_and_backbone',historical_results='separate_observation_budget_no_net_improvement_claim')
    complete=[o for o,v in zip(outcomes,data.valid) if bool(np.asarray(v[:K]).all())]
    metrics['complete_K_subset']=summarize_outcomes(complete,config.get('rho_target',.01),config.get('target_frr_limit'))
    if output is not None:
        path=Path(output);path.mkdir(parents=True,exist_ok=True)
        _write_rows(path/'decisions.csv.gz',decisions);_write_rows(path/'outcomes.csv.gz',[asdict(o) for o in outcomes])
        (path/'metrics.json').write_text(json.dumps(metrics,indent=2))
    return metrics


class RuleController(torch.nn.Module):
    score_kind='cost';n_actions=3
    def __init__(self,rule,thresholds,max_windows):
        super().__init__();self.rule=rule;self.thresholds=thresholds;self.max_windows=max_windows
    def score(self,state,k):
        p=state[:,0];a=torch.full_like(p,2,dtype=torch.long)
        reject=self.thresholds.get('reject_target_probability',.01);accept=self.thresholds.get('accept_target_probability',.9)
        final=self.thresholds.get('final_target_probability',.5)
        if self.rule=='first_window':a=torch.where(p<=reject,1,0)
        elif self.rule=='fixed_K_gru':a=torch.where(p>=final,0,1) if k==self.max_windows else a
        elif self.rule=='causal_any_target':
            a=torch.where(p>=accept,0,a)
            if k==self.max_windows:a=torch.where(p>=accept,0,1)
        else:
            a=torch.where(p>=accept,0,torch.where(p<=reject,1,a))
            if k==self.max_windows:a=torch.where(p>=final,0,1)
        scores=state.new_ones((len(state),3));scores.scatter_(1,a[:,None],0.);return scores


def evaluate_rules(model,data,config,device='cpu',output=None,thresholds=None,states=None,probabilities=None):
    if states is None:states,probabilities=prefix_scores(model,data,device)
    thresholds=thresholds or config.get('rule_thresholds')
    if thresholds is None:raise ValueError('rule_thresholds must be selected on DEV and frozen before evaluation')
    results={}
    for rule in ('first_window','fixed_K_gru','gru_threshold','causal_any_target'):
        head=RuleController(rule,thresholds.get(rule,thresholds),model.max_windows)
        rule_states=probabilities[...,None]
        results[rule]=evaluate_policy(model,head,data,config,device,None if output is None else Path(output)/rule,
                                     states=rule_states,probabilities=probabilities)
    timing,cost=_configs(config)
    outcomes=[]
    from .contracts import Outcome
    for record in data.records():
        seconds=record.raw_length/record.sample_rate_hz
        outcomes.append(Outcome(record.read_id,record.is_target,'all_accept',False,False,seconds,0.,terminal_cost(record.is_target,seconds,False,0.,cost),action=Action.ACCEPT))
    results['all_accept']=summarize_outcomes(outcomes,config.get('rho_target',.01),config.get('target_frr_limit'))
    if output is not None:
        destination=Path(output)/'all_accept';destination.mkdir(parents=True,exist_ok=True)
        _write_rows(destination/'decisions.csv.gz',[])
        _write_rows(destination/'outcomes.csv.gz',[asdict(o) for o in outcomes])
        (destination/'metrics.json').write_text(json.dumps(results['all_accept'],indent=2))
    if output is not None:
        Path(output).mkdir(parents=True,exist_ok=True);(Path(output)/'rule_metrics.json').write_text(json.dumps(results,indent=2))
    return results


def select_rule_thresholds(model,data,config,device='cpu',candidates=None):
    if getattr(data,'split','dev')=='test' or any(r.split=='test' for r in data.records()):raise ValueError('TEST cannot select thresholds')
    states,probs=prefix_scores(model,data,device)
    supplied_candidates=candidates is not None
    candidates=candidates or [dict(reject_target_probability=r,accept_target_probability=a,final_target_probability=.5)
                            for r,a in ((.01,.9),(.05,.9),(.1,.9),(.01,.8),(.01,.99))]
    selected={}
    for rule in ('first_window','fixed_K_gru','gru_threshold','causal_any_target'):
        best=None
        rule_candidates=candidates
        if not supplied_candidates and rule=='fixed_K_gru':
            rule_candidates=[dict(final_target_probability=f) for f in (.01,.05,.1,.2,.5)]
        elif not supplied_candidates and rule=='gru_threshold':
            rule_candidates=candidates+[dict(reject_target_probability=.01,accept_target_probability=.9,final_target_probability=f) for f in (.01,.05,.1,.2)]
        for t in rule_candidates:
            metrics=evaluate_policy(model,RuleController(rule,t,model.max_windows),data,config,device,
                                    states=probs[...,None],probabilities=probs)
            key=(not metrics['constraint_met'],metrics['weighted_cost'] if metrics['constraint_met'] else metrics['target_effective_frr'],metrics['weighted_cost'])
            if best is None or key<best[0]:best=(key,t)
        selected[rule]=best[1]
    return selected


def exact_policy_expectation(model,controller,record,timing,cost,max_windows,device='cpu'):
    """Enumerate the finite tree, including environment-only terminal probability."""
    import copy
    model=model.to(device).eval();controller=controller.to(device).eval()
    env=ReplayEnv(record,timing,cost,max_windows=max_windows);obs=env.reset()
    branches=[]
    def visit(env,obs,h,mass):
        if obs is None:branches.append((mass,env.outcome));return
        with torch.inference_mode():
            state=None
            for k in obs.processed_ks or (obs.k,):
                if record.features is not None:e=torch.as_tensor(record.features[k-1:k],dtype=torch.float32,device=device)
                else:e=model.encode(torch.as_tensor(record.windows[k-1:k,None],dtype=torch.float32,device=device))
                state,h=model.step_state(e,h,k)
            logits=controller.score(state,obs.k);p=masked_softmax(logits,k=obs.k,max_windows=max_windows)[0].cpu().numpy()
        for a,prob in enumerate(p):
            if prob<=0:continue
            next_env=copy.deepcopy(env);next_obs,_=next_env.step(Action(a));visit(next_env,next_obs,h,mass*float(prob))
    visit(env,obs,torch.zeros(1,128,device=device),1.)
    return dict(probability_mass=sum(m for m,o in branches),expected_cost=sum(m*o.total_cost for m,o in branches),
                target_effective_frr=sum(m*o.effective_reject for m,o in branches) if record.is_target else None,
                target_command_frr=sum(m*o.command_reject for m,o in branches) if record.is_target else None,
                expected_pore_seconds=sum(m*o.pore_seconds for m,o in branches),
                expected_compute_seconds=sum(m*o.compute_seconds for m,o in branches),terminal_branches=len(branches))


def fit_independent_window_head(train_data,dev_data,max_windows=4,device='cpu',epochs=5,seed=0):
    """Optional two-class linear control; fit TRAIN and select only by DEV CE."""
    torch.manual_seed(seed)
    head=torch.nn.Linear(128,2).to(device);optimizer=torch.optim.AdamW(head.parameters(),lr=3e-4)
    train_e=torch.as_tensor(train_data.features[:,:max_windows],dtype=torch.float32)
    train_v=torch.as_tensor(train_data.valid[:,:max_windows],dtype=torch.bool)
    labels=torch.as_tensor(train_data.is_target,dtype=torch.long)
    best=None;best_loss=float('inf')
    for _ in range(epochs):
        head.train()
        for ids in torch.randperm(len(labels)).split(256):
            x=train_e[ids].to(device);v=train_v[ids].to(device);y=labels[ids].to(device)
            losses=torch.nn.functional.cross_entropy(head(x).reshape(-1,2),y[:,None].expand(-1,max_windows).reshape(-1),reduction='none').reshape(len(ids),max_windows)
            loss=((losses*v).sum(1)/v.sum(1).clamp_min(1)).mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
        head.eval()
        with torch.inference_mode():
            x=torch.as_tensor(dev_data.features[:,:max_windows],dtype=torch.float32,device=device)
            v=torch.as_tensor(dev_data.valid[:,:max_windows],dtype=torch.bool,device=device)
            y=torch.as_tensor(dev_data.is_target,dtype=torch.long,device=device)
            losses=torch.nn.functional.cross_entropy(head(x).reshape(-1,2),y[:,None].expand(-1,max_windows).reshape(-1),reduction='none').reshape(len(y),max_windows)
            val=float(((losses*v).sum(1)/v.sum(1).clamp_min(1)).mean())
        if val<best_loss:best_loss=val;best={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    head.load_state_dict(best);head.eval();return head


def evaluate_independent_rules(model,head,data,config,thresholds,device='cpu',output=None):
    """Opt-in independent current-window and four fixed-budget aggregation controls."""
    with torch.inference_mode():
        p=head.to(device).eval()(torch.as_tensor(data.features[:,:model.max_windows],dtype=torch.float32,device=device)).softmax(-1)[...,1].cpu()
    results={}
    controls={'independent_gray':('gru_threshold',p),'independent_any_target':('causal_any_target',p)}
    aggregates={'mean':p.mean(1),'vote':(p>=.5).float().mean(1),
                'highest_confidence':p.gather(1,(p-.5).abs().argmax(1)[:,None]).squeeze(1),'any_target':p.max(1).values}
    for name,aggregate in aggregates.items():controls['fixed_K_'+name]=('fixed_K_gru',aggregate[:,None].expand_as(p))
    for name,(rule,probabilities) in controls.items():
        controller=RuleController(rule,thresholds.get(name,thresholds),model.max_windows)
        results[name]=evaluate_policy(model,controller,data,config,device,None if output is None else Path(output)/name,
                                     states=probabilities[...,None],probabilities=p)
    return results


def compare_effective_rejections(reference_outcomes,policy_outcomes):
    """Paired correction relative to the frozen first-window rule."""
    reference={o.read_id:o for o in reference_outcomes}
    avoided=added=0
    for o in policy_outcomes:
        if not o.is_target:continue
        before=reference[o.read_id].effective_reject
        avoided+=bool(before and not o.effective_reject);added+=bool(o.effective_reject and not before)
    return dict(avoided_effective_target_rejections=avoided,new_effective_target_rejections=added)


def _batched_hop_replay(records,states,probabilities,controller,timing,cost,K,device,mode,rng,save_decisions=True):
    """Batch currently reached path nodes; timing remains in each read environment."""
    envs=[ReplayEnv(r,timing,cost,max_windows=K) for r in records]
    active={i:e.reset() for i,e in enumerate(envs)}
    active={i:o for i,o in active.items() if o is not None};decisions=[]
    while active:
        groups={}
        for i,o in active.items():groups.setdefault(o.k,[]).append(i)
        next_active={}
        for k,ids in groups.items():
            for start in range(0,len(ids),256):
                batch=ids[start:start+256];s=states[batch,k-1].to(device)
                runtime=s.new_tensor([[active[i].pre_head_elapsed_seconds,active[i].pre_head_compute_seconds] for i in batch])
                with torch.inference_mode():scores=controller.score(torch.cat([s,runtime],-1),k).cpu()
                for idx,i in enumerate(batch):
                    o=active[i];score=scores[idx:idx+1]
                    action=choose_action(score,controller.score_kind,k,K,rng if mode=='sample' else None)
                    row={'action':action.name.lower()}
                    if save_decisions:
                        row.update({key:value for key,value in vars(o).items() if key!='window'})
                        row.update(read_id=records[i].read_id,scores=score.reshape(-1).tolist(),
                                   legal_mask=legal_action_mask(k,K,4).tolist(),auxiliary_target_probability=float(probabilities[i,k-1]))
                    decisions.append(row);next_obs,_=envs[i].step(action)
                    if next_obs is not None:next_active[i]=next_obs
        active=next_active
    return [e.outcome for e in envs],decisions


def select_independent_rule_thresholds(model,head,dev_data,config,device='cpu',candidates=None):
    """Choose independent-control thresholds once on DEV for subsequent freezing."""
    if dev_data.split=='test':raise ValueError('TEST cannot select thresholds')
    candidates=candidates or [dict(reject_target_probability=r,accept_target_probability=a,final_target_probability=f)
                             for r,a,f in ((.01,.9,.5),(.05,.9,.5),(.1,.9,.5),(.01,.8,.5),(.01,.99,.5),
                                           (.01,.9,.01),(.01,.9,.05),(.01,.9,.1),(.01,.9,.2))]
    best={}
    for thresholds in candidates:
        results=evaluate_independent_rules(model,head,dev_data,config,thresholds,device)
        for name,metrics in results.items():
            key=(not metrics['constraint_met'],metrics['weighted_cost'] if metrics['constraint_met'] else metrics['target_effective_frr'],metrics['weighted_cost'])
            if name not in best or key<best[name][0]:best[name]=(key,thresholds)
    return {name:entry[1] for name,entry in best.items()}
