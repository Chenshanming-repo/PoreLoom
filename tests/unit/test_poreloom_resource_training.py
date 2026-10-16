"""Resource loss and hard gated training, including natural endings."""
import importlib
import json
import numpy as np
import pytest
import torch
from torch import nn
from poreloom.model import PoreLoom
from poreloom.data import ControlDataset
from poreloom.controllers import PolicyController
from poreloom.hostfloor_training import conditional_hostfloor_loss, train_hostfloor
from poreloom._policy_training import branch_cost_table
from poreloom.risk_training import exact_expected_components


def api():
    assert importlib.util.find_spec('poreloom.resource_training') is not None
    return importlib.import_module('poreloom.resource_training')


def test_resource_loss_denominators_gradients_and_zero_equivalence():
    comp = {key: torch.tensor(values, requires_grad=True) for key, values in dict(
        command_reject=[0., 0., .2, .8, 0.], effective_reject=[.9, 0., .1, .1, 0.],
        compute_seconds=[1., 2., 3., 4., 0.], pore_seconds=[2., 6., 8., 10., 1.]).items()}
    target = torch.tensor([0, 0, 1, 1, 1], dtype=torch.bool)
    old = conditional_hostfloor_loss(comp, target, .6, 2.)
    zero = api().resource_loss(comp, target, .6, 2., 0., 0., 2., 4.)
    assert torch.equal(zero[0], old[0])
    loss, risk, host, compute, pore = api().resource_loss(comp, target, .6, 2., .1, .2, 2., 4.)
    assert compute.item() == pytest.approx(1.)
    assert pore.item() == pytest.approx(1.)
    assert loss.item() == pytest.approx(old[0].item() + .3)
    loss.backward()
    assert comp['compute_seconds'].grad.tolist() == pytest.approx([.01]*5)
    assert comp['pore_seconds'].grad.tolist() == pytest.approx([.025, .025, 0., 0., 0.])
    assert risk.item() == pytest.approx(1/3)
    assert host.item() == pytest.approx(.45)


@pytest.mark.parametrize('host,target,flag', [(.94,.006,1), (.93,.001,1), (.9354,.0053,0)])
def test_resource_key_hard_protection(host, target, flag):
    key = api().resource_key(dict(host_effective_reject_rate=host, target_command_frr=target,
        mean_compute_seconds=.01, host_mean_pore_seconds=2.), .9354, .0053, .02, 2.5)
    assert key[0] == flag
    if flag == 0:
        assert key[1:] == pytest.approx((1.3, target, 2., .01))


@pytest.mark.parametrize('scale', [0., -1., float('nan')])
def test_invalid_scale_rejected(scale):
    comp = dict(command_reject=torch.zeros(2), effective_reject=torch.zeros(2),
        compute_seconds=torch.ones(2), pore_seconds=torch.ones(2))
    with pytest.raises(ValueError, match='positive'):
        api().resource_loss(comp, torch.tensor([True, False]), 0., 1., 0., 0., scale, 1.)


@pytest.fixture
def tiny(tmp_path):
    old_threads = torch.get_num_threads(); torch.set_num_threads(1)
    torch.manual_seed(19)
    model = PoreLoom(nn.Linear(128, 128), 128, 2)
    features = np.random.default_rng(19).normal(size=(6, 2, 128)).astype('float32')
    valid = np.array([[1,1],[1,1],[1,0],[1,0],[0,0],[0,0]], bool)
    data = ControlDataset(features, valid, np.array([0,1,0,1,0,1], bool),
        np.array([9000,9000,3100,3100,1000,1000]), np.array([str(i) for i in range(6)]),
        None, 'train', np.full(6, 'synthetic'))
    torch.save(dict(model_state=model.state_dict(), controller_state=PolicyController(max_windows=2).state_dict(),
        variant='softstop'), tmp_path/'initial.pt')
    cfg = dict(max_windows=2, seed=19, controller_checkpoint=str(tmp_path/'initial.pt'),
        timing=dict(eject_latency_seconds=.1), cost={}, host_constraint=dict(train_floor=.95,dev_floor=.95),
        training=dict(hostfloor=dict(epochs=3,patience=5,batch_size=4,head_only_epochs=2)),
        resource_ablation=dict(alpha=.01,beta=.01,compute_scale=.02,pore_scale=2.,
            dev_compute_scale=.02,dev_pore_scale=2.,target_limit=.0053,frozen_representation=False))
    yield model, data, cfg, tmp_path
    torch.set_num_threads(old_threads)


def test_natural_ends_conserve_mass_and_resource_cost(tiny):
    _, data, cfg, _ = tiny
    _, available, _, branches, ends = branch_cost_table(data, cfg, return_details=True)
    logits = torch.zeros(6,2,3,requires_grad=True)
    comp = exact_expected_components(logits, {k:torch.tensor(v) for k,v in branches.items()},
        torch.tensor(available), {k:torch.tensor(v) for k,v in ends.items()})
    assert torch.allclose(comp['probability_mass'], torch.ones(6))
    for key in ('compute_seconds','pore_seconds','command_reject'):
        assert comp[key][4:].tolist() == pytest.approx(ends[key][4:].tolist())
    api().resource_loss(comp, torch.tensor(data.is_target), .95, 1., .01, .01,.02,2.)[0].backward()
    assert torch.isfinite(logits.grad).all()


def test_no_feasible_checkpoint_is_not_promoted_and_all_epochs_saved(tiny):
    model,data,cfg,root = tiny
    assert api().train_resource(model,data,data,cfg,root/'run') is None
    assert not (root/'run'/'selected.pt').exists()
    history = json.loads((root/'run'/'history.json').read_text())
    assert [r['epoch'] for r in history] == [0,1,2,3]
    for row in history:
        cp = torch.load(root/'run'/'checkpoints'/f"epoch_{row['epoch']:03d}.pt",weights_only=False)
        assert cp['selection_status'] == 'diagnostic_infeasible'
        assert cp['selection_key'][0] == 1
        assert cp['stop_key']
        assert cp['hashes']['model'] and cp['hashes']['timing']
    assert (root/'run'/'diagnostic.pt').is_file()


def test_frozen_representation_keeps_classifier_and_gru_identical(tiny):
    model,data,cfg,root = tiny
    cp = torch.load(root/'initial.pt',weights_only=False); cp['variant']='hostfloor'
    torch.save(cp,root/'frozen.pt'); cfg['controller_checkpoint']=str(root/'frozen.pt')
    cfg['resource_ablation']['frozen_representation']=True
    original = {k:v.clone() for k,v in model.state_dict().items()}
    api().train_resource(model,data,data,cfg,root/'frozen')
    final = torch.load(root/'frozen'/'last.pt',weights_only=False)
    assert all(torch.equal(final['model_state'][k],v) for k,v in original.items())
    assert all(not p.requires_grad for p in model.parameters())
    assert any(not torch.equal(v,final['controller_state'][k]) for k,v in cp['controller_state'].items())


def test_A_exact_endpoint_matches_old_training(tiny):
    model,data,cfg,root = tiny
    cfg['resource_ablation'].update(alpha=0.,beta=0.)
    train_hostfloor(model,data,data,cfg,root/'old')
    api().train_resource(model,data,data,cfg,root/'A')
    old = torch.load(root/'old'/'last.pt',weights_only=False)
    new = torch.load(root/'A'/'last.pt',weights_only=False)
    assert old['epoch'] == new['epoch']
    assert old['dual'] == new['dual']
    for field in ('model_state','controller_state'):
        assert all(torch.equal(v,new[field][k]) for k,v in old[field].items())


def test_target_limit_is_relative_gate_and_equality_is_feasible(tiny):
    model,data,cfg,root = tiny
    cp = torch.load(root/'initial.pt',weights_only=False)
    for v in cp['controller_state'].values(): v.zero_()
    cp['controller_state']['head.2.bias'][1]=10.
    torch.save(cp,root/'initial.pt')
    cfg['host_constraint']['dev_floor']=1/3
    cfg['resource_ablation']['target_limit']=2/3
    cfg['training']['hostfloor']['epochs']=0
    cfg['target_frr_limit']=.0001
    selected = api().train_resource(model,data,data,cfg,root/'feasible')
    assert selected == root/'feasible'/'selected.pt'
    saved = torch.load(selected,weights_only=False)
    assert saved['selection_status']=='feasible'
    assert saved['target_constraint_met'] is True
    assert saved['best_key'][0]==0


def test_diagnostic_has_explicit_role_and_loss_history_has_resources(tiny):
    model,data,cfg,root = tiny
    api().train_resource(model,data,data,cfg,root/'roles')
    cp = torch.load(root/'roles'/'diagnostic.pt',weights_only=False)
    assert cp['checkpoint_role'] == 'diagnostic'
    assert cp['selection_status'] == 'diagnostic_only'
    row = json.loads((root/'roles'/'history.json').read_text())[1]
    assert row['train']['compute_normalized'] >= 0
    assert row['train']['host_pore_normalized'] > 0


def test_static_early_stop_and_input_guards(tiny):
    model,data,cfg,root = tiny
    cfg['training']['hostfloor'].update(epochs=8,patience=2,head_lr=0.,gru_lr=0.)
    api().train_resource(model,data,data,cfg,root/'static')
    assert torch.load(root/'static'/'last.pt',weights_only=False)['epoch']==2
    protected=root/'protected';protected.mkdir();(protected/'keep').write_text('old')
    with pytest.raises(ValueError,match='empty'):
        api().train_resource(model,data,data,cfg,protected)
    assert (protected/'keep').read_text()=='old'
    data.split='test'
    with pytest.raises(ValueError,match='TEST'):
        api().train_resource(model,data,data,cfg,root/'test')
