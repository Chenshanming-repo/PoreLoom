"""Terminal cumulative read cost and explicit simulated service configuration."""
from dataclasses import dataclass
import math

@dataclass
class CostConfig:
    host_weight: float=1.0
    target_reject_penalty: float=100.0
    compute_weight: float=1.0
    cost_scale: float=100.0
    def __post_init__(self):
        if any(not math.isfinite(x) or x<0 for x in (self.host_weight,self.target_reject_penalty,self.compute_weight)) or self.cost_scale<=0: raise ValueError('invalid cost configuration')

@dataclass
class TimingConfig:
    sample_rate_hz: float=4000.0
    preprocessing_seconds: float=0.0
    encoder_seconds: float=0.0
    gru_seconds: float=0.0
    head_seconds: float=0.0
    eject_latency_seconds: float=0.0
    queue_seconds: object=0.0
    queue_mode: str='fixed'
    source: str='zero_service_smoke'
    def __post_init__(self):
        if self.sample_rate_hz<=0: raise ValueError('sample rate must be positive')
        vals=[self.preprocessing_seconds,self.encoder_seconds,self.gru_seconds,self.head_seconds,self.eject_latency_seconds]
        vals.extend(self.queue_seconds if isinstance(self.queue_seconds,(list,tuple)) else [self.queue_seconds])
        if any(not math.isfinite(float(x)) or x<0 for x in vals):raise ValueError('invalid service duration')
    def queue_at(self,k):
        return float(self.queue_seconds[min(k-1,len(self.queue_seconds)-1)]) if isinstance(self.queue_seconds,(list,tuple)) else float(self.queue_seconds)
    @property
    def window_service_seconds(self):return self.preprocessing_seconds+self.encoder_seconds+self.gru_seconds

def terminal_cost(is_target,pore_seconds,effective_reject,compute_seconds,config):
    return float(config.host_weight*(not is_target)*pore_seconds+config.target_reject_penalty*is_target*effective_reject+config.compute_weight*compute_seconds)
