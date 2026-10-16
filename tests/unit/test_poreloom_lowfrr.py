"""Risk objectives retain target denominators and environment terminal mass."""
import torch
from poreloom.risk_training import (
    exact_expected_components, conditional_resource_risk, RiskCheckpointManager,
    annotate_metrics, teacher_loss,
)


def test_conditional_risk_not_diluted_and_survival_has_gradient():
    logits = torch.zeros(2, 2, 3, requires_grad=True)
    branches = {key: torch.zeros(2, 2, 2) for key in ('command_reject','pore_seconds','compute_seconds')}
    branches['command_reject'][:, :, 1] = 1
    branches['command_reject'][:, 1, 1] = .2
    ends = {key: torch.zeros(2) for key in branches}
    available = torch.tensor([[True, False], [True, True]])
    components = exact_expected_components(logits, branches, available, ends)
    assert torch.allclose(components['probability_mass'], torch.ones(2))
    y = torch.tensor([False, True])
    _, small = conditional_resource_risk(components, y, {'rho_target': .01})
    _, large = conditional_resource_risk(components, y, {'rho_target': .5})
    assert torch.allclose(small, large)
    small.backward()
    assert logits.grad[1, 0, 2].abs() > 0


def test_feasible_checkpoint_survives_lower_resource_infeasible(tmp_path):
    config = {'target_frr_limit': .005}
    def metrics(risk, time):
        return dict(target_command_frr=risk,target_effective_frr=0.,host_mean_pore_seconds=time,
                    host_mean_compute_seconds=0.,target_mean_compute_seconds=0.)
    manager = RiskCheckpointManager(tmp_path, config)
    manager.consider({'marker': 1}, metrics(.002, 4.), 5.)
    manager.consider({'marker': 2}, metrics(.02, .1), 7.)
    checkpoint = torch.load(manager.best_path, weights_only=False)
    assert checkpoint['marker'] == 1
    assert checkpoint['beta'] == 5.
    unconstrained = annotate_metrics(metrics(.5, 1.), {})
    assert unconstrained['constraint_enabled'] is False
    assert unconstrained['constraint_met'] is None


def test_teacher_cross_entropy_weights_reads_equally():
    logits = torch.zeros(2, 2, 3)
    actions = torch.tensor([[0, -1], [2, 1]])
    visits = actions >= 0
    loss = teacher_loss(logits, actions, visits)
    expected = (torch.log(torch.tensor(3.)) + (torch.log(torch.tensor(3.))+torch.log(torch.tensor(2.)))/2)/2
    assert torch.allclose(loss, expected)


def test_factorized_beta_refreshes_future_reject_labels():
    from poreloom.controllers import FactorizedCostController
    from poreloom.train_factorized import _examples
    from poreloom.contracts import ReadRecord
    from poreloom.replay import ReplayEnv
    from poreloom.costs import TimingConfig, CostConfig
    import numpy as np
    record=ReadRecord('target',True,30000,np.ones(2,dtype=bool))
    env=ReplayEnv(record,TimingConfig(),CostConfig(),2);env.reset()
    controller=FactorizedCostController(2,input_dim=257,beta=0.)
    with torch.no_grad():
        for p in controller.parameters():p.zero_()
        # At the future head reject has lower resource but higher risk.
        controller.heads[1][-1].bias.copy_(torch.tensor([2.,0.,0., -2.,5.,0., 0.,0.,0.]))
    states=torch.zeros(1,2,257)
    low=_examples([(0,env)],states,controller,False,.01,[record],2,7.5,.03)
    controller.beta=10.
    high=_examples([(0,env)],states,controller,False,.01,[record],2,7.5,.03)
    assert low[1][0,2,1].item()==1.
    assert high[1][0,2,1].item()==0.
    assert controller.components(states[:,0],1)[0,0,1].item()==0.


def test_factorized_hop_initialization_copies_short_and_zeros_runtime():
    from poreloom.controllers import FactorizedCostController
    from poreloom.train_factorized import initialize_factorized
    source=FactorizedCostController(2)
    hop=FactorizedCostController(2,input_dim=259,n_actions=4)
    initialize_factorized(hop,dict(controller_type='factorized',controller_state=source.state_dict()))
    assert torch.equal(hop.heads[0][0].weight[:,:257],source.heads[0][0].weight)
    assert torch.count_nonzero(hop.heads[0][0].weight[:,257:])==0
    assert torch.equal(hop.heads[0][-1].weight[9:12],source.heads[0][-1].weight[6:9])
