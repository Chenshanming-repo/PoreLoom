"""Host-floor objective and CPU training under natural-end replay semantics."""
import importlib
import json
import numpy as np
import pytest
import torch
from torch import nn
from poreloom.model import PoreLoom
from poreloom.data import ControlDataset
from poreloom.controllers import PolicyController
from poreloom._policy_training import branch_cost_table
from poreloom.risk_training import exact_expected_components


def api():
    assert importlib.util.find_spec('poreloom.hostfloor_training') is not None
    return importlib.import_module('poreloom.hostfloor_training')


def test_conditional_objective_uses_all_reads_and_correct_gradient():
    # Class sizes differ; naturally ended members remain in both denominators.
    command = torch.tensor([0., 0., .2, .8, 0.], requires_grad=True)
    effective = torch.tensor([.9, 0., .1, .1, 0.], requires_grad=True)
    y = torch.tensor([False, False, True, True, True])
    loss, risk, host = api().conditional_hostfloor_loss(
        dict(command_reject=command, effective_reject=effective), y, .6, 2.)
    assert risk.item() == pytest.approx(1 / 3)
    assert host.item() == pytest.approx(.45)
    assert loss.item() == pytest.approx(1 / 3 + .3)
    loss.backward()
    assert command.grad.tolist() == pytest.approx([0., 0., 1 / 3, 1 / 3, 1 / 3])
    assert effective.grad.tolist() == pytest.approx([-1., -1., 0., 0., 0.])


@pytest.mark.parametrize('labels', [[True, True], [False, False]])
def test_conditional_objective_requires_both_classes(labels):
    with pytest.raises(ValueError, match='both classes'):
        api().conditional_hostfloor_loss(dict(command_reject=torch.zeros(2),
            effective_reject=torch.zeros(2)), torch.tensor(labels), .5, 1.)


@pytest.fixture
def tiny(tmp_path):
    old_threads = torch.get_num_threads(); torch.set_num_threads(1)
    torch.manual_seed(19)
    model = PoreLoom(nn.Linear(128, 128), 128, 2)
    features = np.random.default_rng(19).normal(size=(6, 2, 128)).astype('float32')
    valid = np.array([[1, 1], [1, 1], [1, 0], [1, 0], [0, 0], [0, 0]], bool)
    data = ControlDataset(features, valid, np.array([0, 1, 0, 1, 0, 1], bool),
        np.array([9000, 9000, 3100, 3100, 1000, 1000]),
        np.array([str(i) for i in range(6)]), None, 'train', np.full(6, 'synthetic'))
    torch.save(dict(model_state=model.state_dict(),
        controller_state=PolicyController(max_windows=2).state_dict(), variant='softstop'), tmp_path/'initial.pt')
    cfg = dict(max_windows=2, seed=19, controller_checkpoint=str(tmp_path/'initial.pt'),
        timing=dict(eject_latency_seconds=.1), cost={},
        host_constraint=dict(train_floor=.95, dev_floor=.95, dual_initial=1., dual_lr=10.),
        training=dict(hostfloor=dict(epochs=3, patience=5, batch_size=4, head_only_epochs=2)))
    yield model, data, cfg, tmp_path
    torch.set_num_threads(old_threads)


def test_command_and_effective_events_and_mass_for_short_reads(tiny):
    model, data, cfg, _ = tiny
    _, available, _, branches, ends = branch_cost_table(data, cfg, return_details=True)
    # 3100 samples: command at .75s is possible; .1s eject completes after natural end.
    assert branches['command_reject'][2, 0, 1] == 1
    assert branches['effective_reject'][2, 0, 1] == 0
    logits = torch.zeros(6, 2, 3, requires_grad=True)
    c = exact_expected_components(logits,
        {k: torch.tensor(v) for k, v in branches.items()}, torch.tensor(available),
        {k: torch.tensor(v) for k, v in ends.items()})
    assert torch.allclose(c['probability_mass'], torch.ones(6))
    assert c['command_reject'][4:].tolist() == [0., 0.]
    assert c['effective_reject'][2].item() == 0
    _, risk, host = api().conditional_hostfloor_loss(c, torch.tensor(data.is_target), .5, 1.)
    assert risk.item() > host.item()


def test_training_saves_each_epoch_freezes_backbone_updates_dual(tiny):
    model, data, cfg, root = tiny
    original = {k: v.clone() for k, v in model.state_dict().items()}
    selected = api().train_hostfloor(model, data, data, cfg, root/'run')
    assert selected == root/'run'/'selected.pt'
    history = json.loads((root/'run'/'history.json').read_text())
    assert [r['epoch'] for r in history] == [0, 1, 2, 3]
    assert [r['head_only'] for r in history[1:]] == [True, True, False]
    for row in history:
        cp = torch.load(root/'run'/'checkpoints'/f"epoch_{row['epoch']:03d}.pt", weights_only=False)
        assert cp['dev_metrics']['n_host'] == cp['dev_metrics']['n_target'] == 3
        assert cp['host_constraint_met'] is False  # at most two thirds are rejectable
        assert row['dev']['soft_probability_mass_max_error'] < 1e-6
        for key in original:
            if key.startswith(('encoder.', 'projection.')):
                assert torch.equal(cp['model_state'][key], original[key])
        if row['epoch']:
            assert row['dual'] == pytest.approx(max(0., row['dual_used'] +
                10. * (.95 - row['train_host_rate'])))
            assert row['train_probability_mass_max_error'] < 1e-6
    first = torch.load(root/'run'/'checkpoints'/'epoch_001.pt', weights_only=False)
    assert all(torch.equal(first['model_state'][k], original[k]) for k in original if k.startswith(('gru.', 'head.')))
    final = torch.load(root/'run'/'last.pt', weights_only=False)
    assert final['epoch'] == 3
    assert any(not torch.equal(final['model_state'][k], original[k]) for k in original if k.startswith('gru.'))
    assert all(row['dev']['constraint_enabled'] is False for row in history)


def test_epoch_zero_and_no_improvement_early_stop(tiny):
    model, data, cfg, root = tiny
    cfg['training']['hostfloor'].update(epochs=8, patience=2, head_lr=0., gru_lr=0.)
    selected = api().train_hostfloor(model, data, data, cfg, root/'static')
    assert torch.load(selected, weights_only=False)['epoch'] == 0
    assert torch.load(root/'static'/'last.pt', weights_only=False)['epoch'] == 2
    cfg['training']['hostfloor']['epochs'] = 0
    api().train_hostfloor(model, data, data, cfg, root/'zero')
    assert torch.load(root/'zero'/'last.pt', weights_only=False)['epoch'] == 0


def test_host_floor_equality_is_feasible_without_target_hard_cap(tiny):
    model, data, cfg, root = tiny
    cp = torch.load(root/'initial.pt', weights_only=False)
    # Always issue REJECT when a command is possible. The one short host ends naturally.
    for key, value in cp['controller_state'].items():
        value.zero_()
    cp['controller_state']['head.2.bias'][1] = 10.
    torch.save(cp, root/'initial.pt')
    cfg['host_constraint']['dev_floor'] = 1/3  # only the long host is effectively ejectable
    cfg['target_frr_limit'] = .001  # a legacy setting must not impose a new hard cap
    cfg['training']['hostfloor']['epochs'] = 0
    selected = api().train_hostfloor(model, data, data, cfg, root/'equal')
    checkpoint = torch.load(selected, weights_only=False)
    assert checkpoint['host_constraint_met'] is True
    assert checkpoint['selection_status'] == 'feasible'
    assert checkpoint['dev_metrics']['target_command_frr'] == pytest.approx(2/3)
    assert checkpoint['dev_metrics']['constraint_enabled'] is False


def test_dev_floor_cannot_change_train_dual_and_dual_clips_at_zero(tiny):
    model, data, cfg, root = tiny
    cfg['training']['hostfloor']['epochs'] = 1
    cfg['host_constraint'].update(train_floor=0., dual_initial=0., dev_floor=0.)
    api().train_hostfloor(model, data, data, cfg, root/'low')
    cfg['host_constraint']['dev_floor'] = 1.
    api().train_hostfloor(model, data, data, cfg, root/'high')
    low = torch.load(root/'low'/'last.pt', weights_only=False)
    high = torch.load(root/'high'/'last.pt', weights_only=False)
    assert low['dual'] == high['dual'] == 0.
    assert low['history'][-1]['train_host_rate'] == high['history'][-1]['train_host_rate']
    for key in low['model_state']:
        assert torch.equal(low['model_state'][key], high['model_state'][key])
    for key in low['controller_state']:
        assert torch.equal(low['controller_state'][key], high['controller_state'][key])


def test_requires_softstop_and_protects_history_and_test_selection(tiny):
    model, data, cfg, root = tiny
    cp = torch.load(root/'initial.pt', weights_only=False)
    cp['variant'] = 'cost'
    torch.save(cp, root/'wrong.pt')
    cfg['controller_checkpoint'] = str(root/'wrong.pt')
    with pytest.raises(ValueError, match='softstop'):
        api().train_hostfloor(model, data, data, cfg, root/'invalid')
    cfg['controller_checkpoint'] = str(root/'initial.pt')
    protected = root/'historical'; protected.mkdir(); (protected/'keep').write_text('original')
    with pytest.raises(ValueError, match='empty'):
        api().train_hostfloor(model, data, data, cfg, protected)
    assert (protected/'keep').read_text() == 'original'
    data.split = 'test'
    with pytest.raises(ValueError, match='TEST'):
        api().train_hostfloor(model, data, data, cfg, root/'test')
