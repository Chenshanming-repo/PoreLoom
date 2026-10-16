"""Causal single-read event replay with cumulative service accounting."""
import numpy as np
import torch
from .contracts import Action,Observation,Outcome,Trace
from .costs import terminal_cost
from .controllers import choose_action,legal_action_mask
from .windowing import START,WINDOW

class ReplayEnv:
    def __init__(self,read_record,timing_config,cost_config,max_windows=None):
        self.record=read_record;self.timing=timing_config;self.cost=cost_config
        self.max_windows=max_windows or len(read_record.valid)
        self.outcome=None;self.observation=None;self.k=0;self.elapsed_seconds=0.;self.compute_seconds=0.
        self.cnn_calls=self.gru_calls=self.head_calls=0;self._started=False
        self.t_end=read_record.raw_length/timing_config.sample_rate_hz
    def _finish(self,reason,action=None):
        reject=action==Action.REJECT;command=action in (Action.ACCEPT,Action.REJECT)
        effective=reject and self.elapsed_seconds+self.timing.eject_latency_seconds<self.t_end
        pore=self.elapsed_seconds+self.timing.eject_latency_seconds if effective else self.t_end
        self.outcome=Outcome(self.record.read_id,bool(self.record.is_target),reason,command,effective,pore,self.compute_seconds,terminal_cost(self.record.is_target,pore,effective,self.compute_seconds,self.cost),self.cnn_calls,self.gru_calls,self.head_calls,action,self.elapsed_seconds if command else None,pore if effective else None,reason in ('input_fallback','timeout_accept'))
        self.observation=None
        return None,self.outcome
    def _observe(self,target):
        # A long wait buffers until its destination before processing any new window.
        arrival=(START+target*WINDOW)/self.timing.sample_rate_hz
        if self.t_end<=max(self.elapsed_seconds,arrival):return self._finish('natural_end')
        processed=[]
        self.elapsed_seconds=max(self.elapsed_seconds,arrival)+self.timing.queue_at(target)
        if self.t_end<=self.elapsed_seconds:return self._finish('natural_end')
        for k in range(self.k+1,target+1):
            end=START+k*WINDOW
            if not self.record.valid[k-1]:
                self.compute_seconds+=self.timing.preprocessing_seconds
                self.elapsed_seconds+=self.timing.preprocessing_seconds
                if self.t_end<=self.elapsed_seconds:return self._finish('natural_end')
                return self._finish('natural_end' if self.record.raw_length<end else 'input_fallback')
            # Started service runs to completion even if the read ends in flight.
            self.cnn_calls+=1;self.gru_calls+=1
            service=self.timing.window_service_seconds
            self.compute_seconds+=service;self.elapsed_seconds+=service;processed.append(k)
            if self.t_end<=self.elapsed_seconds:return self._finish('natural_end')
        pre_elapsed=self.elapsed_seconds;pre_compute=self.compute_seconds
        self.head_calls+=1;self.compute_seconds+=self.timing.head_seconds;self.elapsed_seconds+=self.timing.head_seconds
        if self.t_end<=self.elapsed_seconds:return self._finish('natural_end')
        self.k=target
        window=None if self.record.windows is None else self.record.windows[target-1]
        self.observation=Observation(target,window,START+(target-1)*WINDOW,START+target*WINDOW,self.elapsed_seconds,self.compute_seconds,self.cnn_calls,self.gru_calls,self.head_calls,tuple(processed),pre_elapsed,pre_compute)
        return self.observation,None
    def reset(self):
        if self._started:raise RuntimeError('environment reset twice')
        self._started=True
        return self._observe(1)[0]
    def step(self,action):
        if not self._started or self.outcome is not None:raise RuntimeError('no active decision')
        action=Action(action)
        if not legal_action_mask(self.k,self.max_windows,4)[int(action)]:raise ValueError('illegal action')
        if action in (Action.ACCEPT,Action.REJECT):return self._finish('learned_accept' if action==Action.ACCEPT else 'learned_reject',action)
        return self._observe(self.k+(2 if action==Action.DEFER_LONG else 1))

def rollout_policy(env,backbone,controller,mode='argmax',seed=0):
    if mode not in ('argmax','sample'):raise ValueError('invalid deployment mode')
    trace=Trace();obs=env.reset();device=next(backbone.parameters()).device
    backbone.eval();controller.eval();h=torch.zeros(1,128,device=device);rng=np.random.default_rng(seed) if mode=='sample' else None
    with torch.inference_mode():
        while obs is not None:
            state=None
            for k in obs.processed_ks:
                if env.record.features is not None:e=torch.as_tensor(env.record.features[k-1:k],device=device,dtype=torch.float32)
                else:
                    x=torch.as_tensor(env.record.windows[k-1:k,None],device=device,dtype=torch.float32)
                    e=backbone.encode(x)
                state,h=backbone.step_state(e,h,k)
            aux=torch.softmax(backbone.head(state),-1)[0,1].item()
            control_state=state
            if getattr(controller,'n_actions',3)==4:
                control_state=torch.cat([state,state.new_tensor([[obs.pre_head_elapsed_seconds,obs.pre_head_compute_seconds]])],-1)
            scores=controller.score(control_state,obs.k)
            action=choose_action(scores,controller.score_kind,obs.k,env.max_windows,rng)
            trace.observations.append(obs);trace.actions.append(action);trace.scores.append(scores[0].cpu().tolist());trace.auxiliary_probabilities.append(aux)
            obs,_=env.step(action)
    trace.outcome=env.outcome
    return trace
