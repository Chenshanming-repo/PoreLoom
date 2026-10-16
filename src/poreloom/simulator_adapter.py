"""Signal/action callback interface; no sequencer or closed-loop run is started."""
from dataclasses import dataclass,field
import time
import torch
from .windowing import SignalBuffer,normalize_windows
from .policy import decide


@dataclass
class ReadState:
    read_id:str
    channel_id:int
    h:object
    k:int=0
    buffer:SignalBuffer=field(default_factory=SignalBuffer)


class StreamingAdapter:
    def __init__(self,model,current_read,send_action,device='cpu',timeout_seconds=5.):
        self.model=model.to(device).eval();self.current_read=current_read;self.send_action=send_action
        self.device=device;self.timeout_seconds=timeout_seconds;self.states={};self.logs=[];self.last_seen={}
    def start(self,read_id,channel_id):
        for old,state in list(self.states.items()):
            if state.channel_id==channel_id:self.end(old,time.monotonic())
        self.states[read_id]=ReadState(read_id,channel_id,torch.zeros(1,128,device=self.device))
        self.last_seen[read_id]=time.monotonic()
    def _finish(self,state,action,reason,now,p_host=float('nan'),p_target=float('nan')):
        current=self.current_read(state.channel_id)==state.read_id
        if not current:action,reason='none','natural_end'
        self.logs.append(dict(read_id=state.read_id,channel_id=state.channel_id,window_index=state.k,signal_end=state.buffer.next_start,p_host=p_host,p_target=p_target,action=action,reason=reason,decision_time=now))
        if action!='defer':
            if current and action in ('accept','reject'):self.send_action(state.channel_id,state.read_id,action)
            self.states.pop(state.read_id,None);self.last_seen.pop(state.read_id,None)
    def feed(self,read_id,offset,adc,now=None):
        now=time.monotonic() if now is None else now
        state=self.states.get(read_id)
        if state is None:return
        if self.current_read(state.channel_id)!=read_id:self._finish(state,'none','natural_end',now);return
        self.last_seen[read_id]=now
        processing_started=time.perf_counter()
        try:state.buffer.push(offset,adc)
        except ValueError:self._finish(state,'accept','input_fallback',now+time.perf_counter()-processing_started);return
        while read_id in self.states:
            window=state.buffer.pop()
            if window is None:return
            x,valid=normalize_windows(window[None]);state.k+=1
            if not valid[0]:self._finish(state,'accept','input_fallback',now+time.perf_counter()-processing_started);return
            with torch.inference_mode():
                e=self.model.encode(torch.from_numpy(x[:,None]).to(self.device))
                z,state.h=self.model.step_features(e,state.h,state.k)
                ph,pt=z.softmax(-1)[0].cpu().tolist()
            action,reason=decide(ph,pt,state.k)
            # The callback is queried again after inference, before sending any command.
            self._finish(state,action,reason,now+time.perf_counter()-processing_started,ph,pt)
    def expire(self,now=None):
        now=time.monotonic() if now is None else now
        for rid,state in list(self.states.items()):
            if now-self.last_seen[rid]>=self.timeout_seconds:self._finish(state,'accept','timeout_accept',now)
    def end(self,read_id,now=None):
        state=self.states.get(read_id)
        if state is None:return
        self.logs.append(dict(read_id=read_id,channel_id=state.channel_id,window_index=state.k,signal_end=state.buffer.next_start,p_host=float('nan'),p_target=float('nan'),action='none',reason='natural_end',decision_time=time.monotonic() if now is None else now))
        self.states.pop(read_id);self.last_seen.pop(read_id,None)
