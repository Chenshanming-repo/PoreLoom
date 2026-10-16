"""Constrained stopping training; legacy deployment-cost training stays separate."""
from pathlib import Path
import copy
import json
import time
import numpy as np
import torch
from torch import nn
from .controllers import PolicyController, ValueHead, masked_softmax
from .model import prefix_loss
from .risk_training import (exact_expected_components, conditional_resource_risk,
    resource_parameters, risk_event, update_beta, RiskCheckpointManager,
    generate_teacher_labels, teacher_loss)


def sampled_resource_loss(states, controller, value, available, branches, ends,
                          labels, config, entropy_weight, beta=0., risk_mode='exact', warmup=False):
    """Sample terminal resources, with class-conditional importance correction."""
    B,K,_=states.shape
    alive=available[:,0].clone()
    terminal={key:val.clone() for key,val in ends.items()}
    visits=[]
    for k in range(K):
        ids=alive.nonzero(as_tuple=True)[0]
        if not len(ids):break
        state=states[ids,k]
        p=masked_softmax(controller(state),k=k+1,max_windows=K)
        dist=torch.distributions.Categorical(probs=p)
        action=dist.sample()
        visits.append((ids,dist.log_prob(action),value(state.detach() if warmup else state),dist.entropy()))
        stopped=action<2
        for key in terminal:
            terminal[key][ids[stopped]]=branches[key][ids[stopped],k,action[stopped]]
        next_alive=torch.zeros_like(alive)
        if k+1<K:next_alive[ids[~stopped]]=available[ids[~stopped],k+1]
        alive=next_alive
    T,C,mu=resource_parameters(config)
    q=labels.float().mean()
    host_weight=(~labels.bool()).float()/(1-q)
    compute_weight=torch.where(labels.bool(),config.get('rho_target',.01)/q,(1-config.get('rho_target',.01))/(1-q))
    cost=host_weight*terminal['pore_seconds']/T+mu*compute_weight*terminal['compute_seconds']/C
    if risk_mode=='reinforce':cost=cost+beta*labels.float()/q*terminal[risk_event(config)]
    reward=-cost.detach()
    actor=states.sum()*0;critic=sum((p.sum()*0 for p in value.parameters()),states.sum()*0)
    for ids,lp,v,entropy in visits:
        actor=actor-(lp*(reward[ids]-v).detach()).sum()/B-entropy_weight*entropy.sum()/B
        critic=critic+((v-reward[ids])**2).sum()/B
    return critic if warmup else actor+.5*critic


def train_risk_policy(variant,model,train_data,dev_data,config,output,device='cpu',resume=False):
    from ._policy_training import branch_cost_table,sample_balanced_ids
    from .control_evaluation import prefix_scores,evaluate_policy
    cfg=copy.deepcopy(config);cfg['variant']=variant;output=Path(output);output.mkdir(parents=True,exist_ok=True)
    seed=cfg.get('seed',20261010);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=model.to(device);model.freeze_encoder()
    if cfg.get('backbone_train_mode','frozen_encoder')!='frozen_encoder':
        raise ValueError('risk policy training currently supports frozen_encoder feature training only')
    controller=PolicyController(max_windows=cfg['max_windows']).to(device)
    value=ValueHead().to(device) if variant=='pg' else None
    setting=cfg['training'][variant];batch_size=setting.get('batch_size',128)
    checkpoint=cfg.get('controller_checkpoint')
    if not checkpoint or not Path(checkpoint).exists():
        raise ValueError('class conditional policy training requires an existing controller_checkpoint')
    cp=torch.load(checkpoint,map_location=device,weights_only=False)
    if variant=='pg' and (cp.get('variant')!='softstop' or
                         cp.get('config',{}).get('variant')!='softstop' or
                         cp.get('config',{}).get('objective')!='class_conditional_risk'):
        raise ValueError('PG initialization requires a constrained V2 softstop checkpoint')
    model.load_state_dict(cp['model_state']);controller.load_state_dict(cp['controller_state'])
    beta=float(cp.get('beta',cfg.get('risk',{}).get('dual_initial',5.)))
    if value is not None and 'value_state' in cp:value.load_state_dict(cp['value_state'])
    if isinstance(cfg.get('resource',{}).get('host_time_ref'),str):
        hosts=~train_data.is_target
        cfg['resource']['host_time_ref']=float(np.mean(train_data.raw_length[hosts]/train_data.sample_rate_hz))
    resource_parameters(cfg)
    _,reachable,_,details,ends=branch_cost_table(train_data,cfg,return_details=True)
    _,dev_reachable,_,dev_details,dev_ends=branch_cost_table(dev_data,cfg,return_details=True)
    # Include natural-ended reads: their zero risk still belongs in the class denominator.
    eligible=np.arange(len(train_data.valid))
    independent=nn.Linear(128,2).to(device)
    independent_cp=torch.load(cfg['independent_checkpoint'],map_location=device,weights_only=False)
    independent.load_state_dict(independent_cp.get('model_state',independent_cp.get('head_state',independent_cp.get('state_dict',independent_cp))))
    independent.eval();probabilities=[]
    with torch.no_grad():
        for start in range(0,len(eligible),256):
            e=torch.as_tensor(np.asarray(train_data.features[start:start+256]).copy(),dtype=torch.float32,device=device)
            probabilities.append(independent(e).softmax(-1)[...,1].cpu().numpy())
    teacher=generate_teacher_labels(train_data,np.concatenate(probabilities),cfg)
    reference_model=copy.deepcopy(model).eval() if variant=='pg' else None
    reference_controller=copy.deepcopy(controller).eval() if variant=='pg' else None
    if reference_model is not None:
        for module in (reference_model,reference_controller):
            for p in module.parameters():p.requires_grad_(False)
    recurrent=list(model.gru.parameters())+list(model.head.parameters())
    params=list(controller.parameters())+recurrent+(list(value.parameters()) if value is not None else [])
    groups=[dict(params=list(controller.parameters()),lr=setting.get('head_lr',1e-4)),dict(params=recurrent,lr=setting.get('gru_lr',3e-5))]
    if value is not None:groups.append(dict(params=list(value.parameters()),lr=setting.get('value_lr',1e-4)))
    optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
    manager=RiskCheckpointManager(output,cfg);history=[];start_epoch=1;stale=0;started=time.time()
    def tensor_dict(values,ids):return {key:torch.as_tensor(val[ids],device=device) for key,val in values.items()}
    def snapshot(epoch,metrics):
        state=dict(model_state=model.state_dict(),controller_state=controller.state_dict(),config=cfg,variant=variant,epoch=epoch,dev_metrics=metrics,beta=beta)
        if value is not None:state['value_state']=value.state_dict()
        return state
    def validation():
        model.eval();controller.eval()
        states,probs=prefix_scores(model,dev_data,device)
        metrics=evaluate_policy(model,controller,dev_data,cfg,device=device,mode='argmax',states=states,probabilities=probs)
        sums={key:[] for key in dev_details}
        with torch.no_grad():
            for start in range(0,len(states),256):
                ids=np.arange(start,min(start+256,len(states)))
                comps=exact_expected_components(controller(states[ids].to(device)),tensor_dict(dev_details,ids),torch.as_tensor(dev_reachable[ids],device=device),tensor_dict(dev_ends,ids))
                for key in sums:sums[key].append(comps[key])
            resource,risk=conditional_resource_risk({key:torch.cat(val) for key,val in sums.items()},torch.as_tensor(dev_data.is_target,device=device),cfg)
        metrics.update(soft_risk=float(risk),soft_resource_score=float(resource))
        return metrics
    if resume and (output/'last.pt').exists():
        saved=torch.load(output/'last.pt',map_location=device,weights_only=False)
        model.load_state_dict(saved['model_state']);controller.load_state_dict(saved['controller_state']);optimizer.load_state_dict(saved['optimizer_state'])
        if value is not None:value.load_state_dict(saved['value_state'])
        beta=saved['beta'];history=saved['history'];start_epoch=saved['epoch']+1;stale=saved['stale']
        for key in ('best_key','feasible_key','diagnostic_key'):setattr(manager,key,saved[key])
        rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng'].cpu())
        if str(device).startswith('cuda') and 'cuda_rng' in saved:torch.cuda.set_rng_state(saved['cuda_rng'],device)
        if reference_model is not None:
            reference_model.load_state_dict(saved['reference_model_state']);reference_controller.load_state_dict(saved['reference_controller_state'])
    else:
        metrics=validation();manager.consider(snapshot(0,metrics),metrics,beta)
        history.append(dict(epoch=0,dev=metrics,beta=beta,stage='initialization'))
    (output/'config.json').write_text(json.dumps(dict(cfg,variant=variant),indent=2))
    (output/'history.json').write_text(json.dumps(history,indent=2))
    risk_mode=setting.get('risk_mode','exact')
    if risk_mode not in ('exact','reinforce'):raise ValueError('risk_mode must be exact or reinforce')
    warmup_epochs=setting.get('value_warmup_epochs',1) if variant=='pg' else 0
    # Value warmup updates only its state baseline; actor and recurrent parameters stay frozen.
    if start_epoch==1:
        for _ in range(warmup_epochs):
            model.eval();controller.eval();value.train()
            for ids,_ in sample_balanced_ids(train_data.is_target,eligible,batch_size,rng):
                e=torch.as_tensor(np.asarray(train_data.features[ids]).copy(),dtype=torch.float32,device=device)
                valid=torch.as_tensor(train_data.valid[ids],device=device)
                with torch.no_grad():states=model.forward_states(e,valid)
                loss=sampled_resource_loss(states,controller,value,torch.as_tensor(reachable[ids],device=device),tensor_dict(details,ids),tensor_dict(ends,ids),torch.as_tensor(train_data.is_target[ids],device=device),cfg,0.,beta,risk_mode,warmup=True)
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(value.parameters(),1.);optimizer.step()
    for epoch in range(start_epoch,setting.get('epochs',10)+1):
        head_only=epoch<=setting.get('head_only_epochs',2 if variant=='softstop' else 0)
        for p in recurrent:p.requires_grad_(not head_only)
        model.train(not head_only);controller.train()
        if value is not None:value.train()
        teacher_cfg=cfg.get('teacher',{});fraction=min(1.,max(0.,(epoch-1)/4))
        kappa=teacher_cfg.get('weight_initial',1.)+(teacher_cfg.get('weight_final',.1)-teacher_cfg.get('weight_initial',1.))*fraction
        totals=dict(loss=0.,resource=0.,risk=0.,prefix_ce=0.,teacher=0.,kl=0.);count=0
        for ids,_ in sample_balanced_ids(train_data.is_target,eligible,batch_size,rng):
            e=torch.as_tensor(np.asarray(train_data.features[ids]).copy(),dtype=torch.float32,device=device);valid=torch.as_tensor(train_data.valid[ids],device=device)
            y=torch.as_tensor(train_data.is_target[ids],device=device)
            states=model.forward_states(e,valid);logits=controller(states)
            comps=exact_expected_components(logits,tensor_dict(details,ids),torch.as_tensor(reachable[ids],device=device),tensor_dict(ends,ids))
            resource,risk=conditional_resource_risk(comps,y,cfg)
            ce=prefix_loss(model.head(states),y.long(),valid) if valid.any() else logits.sum()*0
            imitation=teacher_loss(logits,torch.as_tensor(teacher['actions'][ids],device=device),torch.as_tensor(teacher['visits'][ids],device=device))
            kl=logits.sum()*0
            objective=resource
            if variant=='pg':
                objective=sampled_resource_loss(states,controller,value,torch.as_tensor(reachable[ids],device=device),tensor_dict(details,ids),tensor_dict(ends,ids),y,cfg,setting.get('entropy_weight',.001),beta,risk_mode)
                with torch.no_grad():reference=reference_controller(reference_model.forward_states(e,valid))
                terms=[]
                for k in range(cfg['max_windows']):
                    p=masked_softmax(logits[:,k],k=k+1,max_windows=cfg['max_windows']);q=masked_softmax(reference[:,k],k=k+1,max_windows=cfg['max_windows'])
                    terms.append((q*(q.clamp_min(1e-8).log()-p.clamp_min(1e-8).log())).sum(-1))
                per_read=(torch.stack(terms,1)*valid).sum(1)/valid.sum(1).clamp_min(1)
                kl=per_read.mean()
            if variant=='softstop' or risk_mode=='exact':objective=objective+beta*(risk-cfg.get('risk',{}).get('epsilon_train',.0025))
            loss=objective+cfg.get('auxiliary_weight',.1)*ce+kappa*imitation+setting.get('kl_weight',.05)*kl
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(params,1.);optimizer.step()
            for key,val in zip(totals,(loss,resource,risk,ce,imitation,kl)):totals[key]+=float(val.detach())*len(ids)
            count+=len(ids)
        # Estimate the complete TRAIN cohort with the final epoch policy, without DEV feedback.
        model.eval();controller.eval();train_risk_sum=0.;train_targets=0
        with torch.no_grad():
            for start in range(0,len(eligible),256):
                ids=eligible[start:start+256];e=torch.as_tensor(np.asarray(train_data.features[ids]).copy(),dtype=torch.float32,device=device)
                valid=torch.as_tensor(train_data.valid[ids],device=device);target=torch.as_tensor(train_data.is_target[ids],device=device)
                comps=exact_expected_components(controller(model.forward_states(e,valid)),tensor_dict(details,ids),torch.as_tensor(reachable[ids],device=device),tensor_dict(ends,ids))
                train_risk_sum+=float(comps[risk_event(cfg)][target].sum());train_targets+=int(target.sum())
        train_risk=train_risk_sum/train_targets;used_beta=beta;beta=update_beta(beta,train_risk,cfg)
        metrics=validation();state=snapshot(epoch,metrics)
        improved=manager.consider(state,metrics,beta);stale=0 if improved else stale+1
        row=dict(epoch=epoch,train={key:val/count for key,val in totals.items()},train_risk=train_risk,beta_used=used_beta,beta=beta,teacher_weight=kappa,head_only=head_only,dev=metrics,elapsed_seconds=time.time()-started)
        history.append(row);print(row,flush=True)
        state.update(optimizer_state=optimizer.state_dict(),history=history,stale=stale,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),best_key=manager.best_key,feasible_key=manager.feasible_key,diagnostic_key=manager.diagnostic_key)
        if reference_model is not None:state.update(reference_model_state=reference_model.state_dict(),reference_controller_state=reference_controller.state_dict())
        if str(device).startswith('cuda'):state['cuda_rng']=torch.cuda.get_rng_state(device)
        torch.save(state,output/'last.pt');(output/'history.json').write_text(json.dumps(history,indent=2))
        if stale>=setting.get('patience',5):break
    return manager.best_path
