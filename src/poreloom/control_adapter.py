"""Causal batch-one controller bridge with identity-keyed physical outcomes."""
from dataclasses import dataclass
import time
import numpy as np
import torch
from .windowing import SignalBuffer, normalize_windows, START, WINDOW
from .controllers import choose_action, legal_action_mask
from .contracts import Action

@dataclass
class ReadState:
    read_id: str
    channel_id: int
    generation: int
    h: object
    buffer: object
    started: float
    k: int = 0
    next_decision_k: int = 1
    elapsed_seconds: float = 0.
    compute_seconds: float = 0.
    cnn_calls: int = 0
    gru_calls: int = 0
    head_calls: int = 0

class ControlStreamingAdapter:
    def __init__(self, backbone, controller, current_read, send_action, timing_config,
                 device='cpu', timeout_seconds=5., mode='argmax', seed=0):
        self.backbone=backbone.to(device).eval(); self.controller=controller.to(device).eval()
        self.current_read=current_read; self.send_action=send_action; self.timing=timing_config
        self.device=device; self.timeout_seconds=timeout_seconds; self.mode=mode
        self.rng=np.random.default_rng(seed); self.states={}; self.last_seen={}
        self.pending_outcomes={}; self.logs=[]; self.outcomes=[]; self.generations={}
        self.max_windows=backbone.max_windows

    def start(self, read_id, channel_id, now=None):
        now=time.monotonic() if now is None else now
        for old,s in list(self.states.items()):
            if s.channel_id==channel_id:self.end(old,now)
        generation=self.generations.get(channel_id,0)+1; self.generations[channel_id]=generation
        s=ReadState(read_id,channel_id,generation,torch.zeros(1,128,device=self.device),
                    SignalBuffer(max_windows=self.max_windows),now)
        self.states[read_id]=s; self.last_seen[read_id]=now
        return generation

    def _current(self,s):
        return self.states.get(s.read_id) is s and self.current_read(s.channel_id)==s.read_id

    def _finish(self,s,action,reason,now,scores=None,aux=None):
        if not self._current(s):
            self.end(s.read_id,now); return
        name=Action(action).name.lower()
        record=dict(read_id=s.read_id,channel_id=s.channel_id,generation=s.generation,
                    window_index=s.k,signal_end=START+s.k*WINDOW,action=name,reason=reason,
                    decision_time=now,elapsed_seconds=s.elapsed_seconds,compute_seconds=s.compute_seconds,
                    cnn_calls=s.cnn_calls,gru_calls=s.gru_calls,head_calls=s.head_calls,
                    scores=scores,auxiliary_target_probability=aux,
                    legal_mask=legal_action_mask(max(s.k,1),self.max_windows,
                                getattr(self.controller,'n_actions',3)).tolist())
        self.logs.append(record)
        if action in (Action.DEFER,Action.DEFER_LONG):
            s.next_decision_k=s.k+(2 if action==Action.DEFER_LONG else 1); return
        key=(s.channel_id,s.read_id,s.generation)
        self.pending_outcomes[key]=dict(record,started=s.started,command_issued=action in (Action.ACCEPT,Action.REJECT),
                                        effective_reject=False,fallback=reason in ('input_fallback','timeout_accept'))
        self.states.pop(s.read_id,None); self.last_seen.pop(s.read_id,None)
        self.send_action(s.channel_id,s.read_id,name)

    def feed(self,read_id,offset,adc,now=None):
        now=time.monotonic() if now is None else now; s=self.states.get(read_id)
        if s is None:return
        if not self._current(s):self.end(read_id,now);return
        self.last_seen[read_id]=now
        try:s.buffer.push(offset,adc)
        except ValueError:self._finish(s,Action.ACCEPT,'input_fallback',now);return
        while self._current(s):
            if s.buffer.received < START+s.next_decision_k*WINDOW:return
            # Buffer to the chosen destination, then queue before starting work.
            arrival=(START+s.next_decision_k*WINDOW)/self.timing.sample_rate_hz
            s.elapsed_seconds=max(s.elapsed_seconds,now-s.started,arrival)+self.timing.queue_at(s.next_decision_k)
            while s.k<s.next_decision_k:
                window=s.buffer.pop()
                if window is None:return
                # Account each started task before invoking code that can deliver
                # an end callback. Completed snapshots then include in-flight work.
                preprocessing=self.timing.preprocessing_seconds
                s.compute_seconds+=preprocessing;s.elapsed_seconds+=preprocessing
                x,valid=normalize_windows(window[None]);s.k+=1
                if not self._current(s):self.end(read_id,now);return
                if not valid[0]:self._finish(s,Action.ACCEPT,'input_fallback',now);return
                s.cnn_calls+=1;s.compute_seconds+=self.timing.encoder_seconds;s.elapsed_seconds+=self.timing.encoder_seconds
                with torch.inference_mode():
                    e=self.backbone.encode(torch.from_numpy(x[:,None]).to(self.device))
                if not self._current(s):self.end(read_id,now);return
                s.gru_calls+=1;s.compute_seconds+=self.timing.gru_seconds;s.elapsed_seconds+=self.timing.gru_seconds
                with torch.inference_mode():
                    state,s.h=self.backbone.step_state(e,s.h,s.k)
                if not self._current(s):self.end(read_id,now);return
            control=state
            if getattr(self.controller,'n_actions',3)==4:
                control=torch.cat([state,state.new_tensor([[s.elapsed_seconds,s.compute_seconds]])],dim=-1)
            s.head_calls+=1;s.compute_seconds+=self.timing.head_seconds;s.elapsed_seconds+=self.timing.head_seconds
            with torch.inference_mode():
                scores=self.controller.score(control,s.k)
                if not self._current(s):self.end(read_id,now);return
                aux=float(self.backbone.head(state).softmax(-1)[0,1])
                action=choose_action(scores,self.controller.score_kind,s.k,self.max_windows,
                                     rng=self.rng if self.mode=='sample' else None)
            self._finish(s,action,'learned_'+Action(action).name.lower(),s.started+s.elapsed_seconds,
                         scores.detach().cpu().reshape(-1).tolist(),aux)

    def expire(self,now=None):
        now=time.monotonic() if now is None else now
        for rid,s in list(self.states.items()):
            if now-self.last_seen[rid]>=self.timeout_seconds:self._finish(s,Action.ACCEPT,'timeout_accept',now)

    def _outcome(self,key,now,effective):
        record=self.pending_outcomes.pop(key,None)
        if record is None:return None
        record.update(effective_reject=effective,pore_seconds=max(0.,now-record.pop('started')),
                      outcome_time=now,outcome_reason='effective_reject' if effective else 'natural_end')
        self.outcomes.append(record);return record

    def on_eject_effective(self,channel_id,read_id,generation,now=None):
        key=(channel_id,read_id,generation);r=self.pending_outcomes.get(key)
        if r is None or not r['command_issued']:return None
        return self._outcome(key,time.monotonic() if now is None else now,True)

    def on_natural_end(self,channel_id,read_id,generation,now=None):
        now=time.monotonic() if now is None else now
        s=self.states.get(read_id)
        if s is not None and (s.channel_id,s.generation)==(channel_id,generation):self.end(read_id,now)
        return self._outcome((channel_id,read_id,generation),now,False)

    def end(self,read_id,now=None):
        now=time.monotonic() if now is None else now;s=self.states.pop(read_id,None)
        self.last_seen.pop(read_id,None)
        if s is None:return
        self.outcomes.append(dict(read_id=read_id,channel_id=s.channel_id,generation=s.generation,
            action='none',outcome_reason='natural_end',command_issued=False,effective_reject=False,
            pore_seconds=max(0.,now-s.started),compute_seconds=s.compute_seconds,
            cnn_calls=s.cnn_calls,gru_calls=s.gru_calls,head_calls=s.head_calls,fallback=False))
