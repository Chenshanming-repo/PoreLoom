"""Small component trainer regressions without fitting a production model."""
import json
import numpy as np
import pytest
import torch
from poreloom.contracts import ReadRecord
from poreloom.controllers import FactorizedCostController
from poreloom.costs import CostConfig, TimingConfig
from poreloom.replay import ReplayEnv
from poreloom.train_factorized import _examples, initialize_factorized


def test_reference_weights_correct_hop_node_multiplicity():
    records=[ReadRecord('host',False,30000,np.ones(2,dtype=bool)),
             ReadRecord('target',True,30000,np.ones(2,dtype=bool))]
    envs=[ReplayEnv(r,TimingConfig(),CostConfig(),2) for r in records]
    for env in envs:env.reset()
    nodes=[(0,envs[0]),(0,envs[0]),(1,envs[1])]
    _,_,weights=_examples(nodes,torch.zeros(2,2,257),FactorizedCostController(2),False,.01,records,2,7.5,.03)
    assert torch.allclose(weights,torch.tensor([.75,.75,1.5]))
    assert weights[:2].sum()==weights[2]


def test_factorized_initialization_rejects_policy_teacher():
    with pytest.raises(ValueError,match='must be factorized'):
        initialize_factorized(FactorizedCostController(1),dict(controller_type='policy',controller_state={}))


def test_trainer_ignores_teacher_checkpoint_and_honors_outer_rounds(monkeypatch,tmp_path):
    import poreloom.train_factorized as training
    import poreloom.control_evaluation as evaluation
    records=[ReadRecord('host',False,8000,np.ones(1,dtype=bool)),
             ReadRecord('target',True,20000,np.ones(1,dtype=bool))]
    class Data:
        def records(self):return iter(records)
    monkeypatch.setattr(training,'_states',lambda *args:torch.zeros(2,1,257))
    seen=[]
    def evaluate(model,controller,data,config,*args,**kwargs):
        seen.append(config['resource']['host_time_ref'])
        return dict(target_command_frr=0.,target_effective_frr=0.,host_mean_pore_seconds=1.,
                    host_mean_compute_seconds=.01,target_mean_compute_seconds=.02)
    monkeypatch.setattr(evaluation,'evaluate_policy',evaluate)
    config=dict(max_windows=1,controller_checkpoint=str(tmp_path/'missing_teacher.pt'),
                target_frr_limit=.005,resource=dict(host_time_ref='train_host_mean_natural_seconds'),
                training=dict(cost=dict(outer_rounds=2,epochs=1,batch_size=2)))
    result=training.train_factorized(torch.nn.Linear(1,1),Data(),Data(),config,tmp_path/'out')
    assert result.name=='best_feasible.pt'
    history=json.loads((result.parent/'history.json').read_text())
    assert [row['round'] for row in history if 'dev_hard' in row]==[1,2]
    assert seen==[2.,2.,2.,2.]
    assert config['resource']['host_time_ref']=='train_host_mean_natural_seconds'
