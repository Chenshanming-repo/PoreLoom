"""Focused regression coverage for learned stopping objectives and causal replay."""
import numpy as np
import pytest
import torch
from poreloom.train_softstop import exact_expected_cost
from poreloom.train_pg import episode_policy_loss


def test_softstop_exact_short_branch_mass_and_gradient():
    logits=torch.zeros(1,2,3,requires_grad=True)
    costs=torch.tensor([[[2.,8.],[3.,9.]]])
    available=torch.tensor([[True,False]])
    end=torch.tensor([5.])
    expected,mass=exact_expected_cost(logits,costs,available,end,100.)
    assert mass.item()==pytest.approx(1.)
    assert expected.item()==pytest.approx((2+8+5)/3/100)
    expected.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0,0].abs().sum()>0
    assert logits.grad[0,1].abs().sum()==0


def test_softstop_final_mask_exact_cost():
    logits=torch.zeros(1,2,3,requires_grad=True)
    costs=torch.tensor([[[2.,8.],[3.,9.]]])
    expected,mass=exact_expected_cost(logits,costs,torch.ones(1,2,dtype=torch.bool),torch.tensor([5.]),1.)
    assert expected.item()==pytest.approx((2+8)/3+(3+9)/6)
    assert mass.item()==pytest.approx(1.)


def test_pg_advantage_detached_and_visited_sum():
    lp=torch.tensor([-.4,-.6],requires_grad=True)
    value=torch.tensor([.1,.2],requires_grad=True)
    entropy=torch.tensor([.3,.4],requires_grad=True)
    loss=episode_policy_loss(lp,value,entropy,torch.tensor(-.5),.01)
    actor=-(lp.detach()*torch.tensor([-.6,-.7])).sum()
    critic=.5*((value.detach()+.5)**2).sum()
    assert loss.item()==pytest.approx((actor+critic-.007).item())
    loss.backward()
    assert torch.allclose(value.grad,torch.tensor([.6,.7]))
    assert torch.allclose(lp.grad,torch.tensor([.6,.7]))


def test_causal_terminal_costs_and_no_post_terminal_steps():
    from poreloom.contracts import Action, ReadRecord
    from poreloom.costs import CostConfig, TimingConfig
    from poreloom.replay import ReplayEnv
    record=ReadRecord('host',False,40000,np.ones(4,bool))
    env=ReplayEnv(record,TimingConfig(head_seconds=.1),CostConfig())
    assert env.reset().k==1
    _,outcome=env.step(Action.ACCEPT)
    assert outcome.pore_seconds==10
    assert outcome.total_cost==pytest.approx(10.1)
    with pytest.raises(RuntimeError):env.step(Action.DEFER)
    late=ReplayEnv(ReadRecord('target',True,4000,np.ones(4,bool)),TimingConfig(eject_latency_seconds=.3),CostConfig())
    late.reset();_,outcome=late.step(Action.REJECT)
    assert outcome.command_reject and not outcome.effective_reject
    assert outcome.total_cost==0


def test_short_read_and_inflight_natural_end():
    from poreloom.contracts import Action,ReadRecord
    from poreloom.costs import CostConfig,TimingConfig
    from poreloom.replay import ReplayEnv
    env=ReplayEnv(ReadRecord('short',True,4000,np.array([True,False,False,False])),TimingConfig(),CostConfig())
    env.reset();observation,outcome=env.step(Action.DEFER)
    assert observation is None and outcome.reason=='natural_end'
    env=ReplayEnv(ReadRecord('inflight',False,4000,np.ones(4,bool)),TimingConfig(encoder_seconds=.3),CostConfig())
    assert env.reset() is None
    assert env.outcome.compute_seconds==pytest.approx(.3)
    assert not env.outcome.command_issued


def test_hop_wait_processes_new_windows_and_one_head():
    from poreloom.contracts import Action,ReadRecord
    from poreloom.costs import CostConfig,TimingConfig
    from poreloom.replay import ReplayEnv
    env=ReplayEnv(ReadRecord('hop',False,40000,np.ones(4,bool)),TimingConfig(encoder_seconds=.01,head_seconds=.02),CostConfig())
    env.reset();observation,_=env.step(Action.DEFER_LONG)
    assert observation.processed_ks==(2,3)
    assert observation.cnn_calls==3 and observation.gru_calls==3 and observation.head_calls==2
    assert observation.compute_seconds==pytest.approx(.07)


def test_cost_future_policy_preserves_target_reject_penalty():
    from poreloom.train_cost import rollout_after_forced_defer
    from poreloom.contracts import ReadRecord
    from poreloom.controllers import CostController
    from poreloom.costs import TimingConfig, CostConfig
    import numpy as np
    import torch
    controller=CostController(3)
    with torch.no_grad():
        for head in controller.heads:
            for parameter in head.parameters():parameter.zero_()
            head[-1].bias.copy_(torch.tensor([1.,0.,2.]))
    record=ReadRecord('target',True,20000,np.ones(3,dtype=bool))
    labels=rollout_after_forced_defer([record],1,torch.zeros(1,3,257),controller,TimingConfig(),CostConfig(),3)
    assert labels.item()==100.


def test_hop_paths_preserve_runtime_and_read_weights():
    from poreloom.train_cost import _nodes,_examples
    from poreloom.contracts import ReadRecord
    from poreloom.controllers import CostController
    from poreloom.costs import TimingConfig,CostConfig
    import numpy as np
    import torch
    records=[ReadRecord('host',False,20000,np.ones(3,dtype=bool))]
    timing=TimingConfig(encoder_seconds=.01,gru_seconds=.02,head_seconds=.03)
    nodes=_nodes(records,timing,CostConfig(),3,hop=True)
    assert len(nodes[3])==2
    observations=[env.observation for _,env in nodes[3]]
    assert {o.head_calls for o in observations}=={2,3}
    long=next(o for o in observations if o.head_calls==2)
    assert long.processed_ks==(2,3)
    assert long.cnn_calls==long.gru_calls==3
    assert abs(long.pre_head_compute_seconds-.12)<1e-8
    x,y,w=_examples(nodes[3],torch.zeros(1,3,257),CostController(3,259,4),True,100,.01,records,3)
    assert torch.allclose(w,torch.tensor([.99,.99]))
    assert sorted(x[:,-1].tolist())==sorted([o.pre_head_compute_seconds for o in observations]) or torch.allclose(x[:,-1],torch.tensor([o.pre_head_compute_seconds for o in observations]))


def test_cost_weights_use_reachable_cohort_at_current_head():
    from poreloom.train_cost import _nodes,_examples
    from poreloom.contracts import ReadRecord
    from poreloom.controllers import CostController
    from poreloom.costs import TimingConfig,CostConfig
    import numpy as np
    import torch
    # The third target never reaches any observation and cannot alter q.
    records=[ReadRecord('host',False,20000,np.ones(2,dtype=bool)),ReadRecord('target',True,20000,np.ones(2,dtype=bool)),ReadRecord('short_target',True,10,np.zeros(2,dtype=bool))]
    nodes=_nodes(records,TimingConfig(),CostConfig(),2)
    _,_,weights=_examples(nodes[2],torch.zeros(3,2,257),CostController(2),False,100,.01,records,2)
    assert torch.allclose(weights,torch.tensor([1.98,.02]))


def test_streaming_natural_end_during_encoder_counts_started_work_only():
    from poreloom.control_adapter import ControlStreamingAdapter
    from poreloom.costs import TimingConfig
    from poreloom.controllers import CostController
    import numpy as np
    import torch
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__();self.max_windows=3;self.head=torch.nn.Linear(257,2);self.callback=None
        def encode(self,x):
            self.callback();return x.new_zeros((len(x),128))
        def step_state(self,e,h,k):raise AssertionError('GRU must not start after natural end')
    model=Backbone();sent=[]
    timing=TimingConfig(preprocessing_seconds=.01,encoder_seconds=.02,gru_seconds=.03,head_seconds=.04,queue_seconds=.05)
    adapter=ControlStreamingAdapter(model,CostController(3),lambda _: 'r',lambda *args:sent.append(args),timing)
    generation=adapter.start('r',1,now=0.)
    def callback():
        assert abs(adapter.states['r'].elapsed_seconds-.83)<1e-7
        adapter.on_natural_end(1,'r',generation,now=.82)
    model.callback=callback
    adapter.feed('r',0,np.sin(np.arange(3000,dtype=np.float32)),now=.75)
    assert sent==[] and 'r' not in adapter.states
    outcome=adapter.outcomes[0]
    assert abs(outcome['compute_seconds']-.03)<1e-7
    assert outcome['cnn_calls']==1 and outcome['gru_calls']==0 and outcome['head_calls']==0


def test_streaming_accept_command_and_preprocessing_fallback_accounting():
    from poreloom.control_adapter import ControlStreamingAdapter
    from poreloom.controllers import CostController
    from poreloom.costs import TimingConfig
    import numpy as np
    import torch
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__();self.max_windows=1;self.head=torch.nn.Linear(257,2)
        def encode(self,x):return x.new_zeros((len(x),128))
        def step_state(self,e,h,k):return e.new_zeros((len(e),257)),h
    controller=CostController(1)
    with torch.no_grad():
        for parameter in controller.parameters():parameter.zero_()
    sent=[];adapter=ControlStreamingAdapter(Backbone(),controller,lambda _: 'r',lambda *args:sent.append(args),TimingConfig(preprocessing_seconds=.01))
    generation=adapter.start('r',1,now=0.)
    adapter.feed('r',0,np.zeros(3000,dtype=np.float32),now=.75)
    pending=adapter.pending_outcomes[(1,'r',generation)]
    assert pending['command_issued'] and pending['fallback']
    assert pending['compute_seconds']==.01
    assert pending['cnn_calls']==pending['gru_calls']==pending['head_calls']==0
    assert sent==[(1,'r','accept')]


def test_replay_input_fault_charges_only_preprocessing_and_natural_end_priority():
    from poreloom.contracts import ReadRecord
    from poreloom.costs import CostConfig,TimingConfig
    from poreloom.replay import ReplayEnv
    fault=ReadRecord('fault',False,40000,np.zeros(4,dtype=bool))
    env=ReplayEnv(fault,TimingConfig(preprocessing_seconds=.03,encoder_seconds=.2,gru_seconds=.1,head_seconds=.1),CostConfig(compute_weight=2))
    assert env.reset() is None
    outcome=env.outcome
    assert outcome.reason=='input_fallback'
    assert outcome.compute_seconds==pytest.approx(.03)
    assert env.elapsed_seconds==pytest.approx(.78)
    assert outcome.total_cost==pytest.approx(10.06)
    assert outcome.cnn_calls==outcome.gru_calls==outcome.head_calls==0
    ending=ReadRecord('ending_fault',False,3200,np.zeros(4,dtype=bool))
    env=ReplayEnv(ending,TimingConfig(preprocessing_seconds=.05),CostConfig())
    assert env.reset() is None
    assert env.outcome.reason=='natural_end'
    assert env.outcome.compute_seconds==pytest.approx(.05)
    assert env.outcome.total_cost==pytest.approx(.85)
    assert not env.outcome.command_issued
    assert env.outcome.cnn_calls==env.outcome.gru_calls==env.outcome.head_calls==0
