"""Shared causal control records; labels and read length stay in the environment."""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
import numpy as np

class Action(IntEnum):
    ACCEPT=0
    REJECT=1
    DEFER=2
    DEFER_LONG=3

@dataclass
class ReadRecord:
    read_id: str
    is_target: bool
    raw_length: int
    valid: np.ndarray
    windows: Optional[np.ndarray]=None
    features: Optional[np.ndarray]=None
    sample_rate_hz: float=4000.0
    split: str=''
    run_or_batch: str='unknown'
    length_source: str='simulated_natural_end'

@dataclass
class Observation:
    k: int
    window: Optional[np.ndarray]
    signal_offset: int
    signal_end: int
    elapsed_seconds: float
    compute_seconds: float
    cnn_calls: int
    gru_calls: int
    head_calls: int
    processed_ks: tuple=()
    pre_head_elapsed_seconds: float=0.0
    pre_head_compute_seconds: float=0.0

@dataclass
class Outcome:
    read_id: str
    is_target: bool
    reason: str
    command_issued: bool
    effective_reject: bool
    pore_seconds: float
    compute_seconds: float
    total_cost: float
    cnn_calls: int=0
    gru_calls: int=0
    head_calls: int=0
    action: Optional[Action]=None
    decision_seconds: Optional[float]=None
    effective_seconds: Optional[float]=None
    fallback: bool=False
    @property
    def elapsed_seconds(self): return self.pore_seconds
    @property
    def command_reject(self): return self.command_issued and self.action==Action.REJECT

@dataclass
class Trace:
    observations: list=field(default_factory=list)
    actions: list=field(default_factory=list)
    scores: list=field(default_factory=list)
    auxiliary_probabilities: list=field(default_factory=list)
    outcome: Optional[Outcome]=None
