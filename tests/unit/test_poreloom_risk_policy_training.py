"""Tiny CPU integrations for constrained V2/V3, independent of signal caches."""
import numpy as np
import pytest
import torch
from torch import nn
from poreloom.model import PoreLoom
from poreloom.data import ControlDataset
from poreloom.controllers import PolicyController
from poreloom._policy_training import train_policy


@pytest.fixture
def synthetic_policy(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    K = 2
    model = PoreLoom(nn.Identity(), 128, K)
    features = np.random.default_rng(7).normal(size=(6,K,128)).astype('float32')
    valid = np.array([[1,1],[1,1],[1,0],[1,0],[0,0],[0,0]], bool)
    y = np.array([0,1,0,1,0,1], bool)
    data = ControlDataset(features, valid, y,
        np.array([9000,9000,3500,3500,1000,1000]), np.array([str(i) for i in range(6)]),
        None, 'train', np.full(6,'synthetic'))
    torch.save(dict(model_state=model.state_dict(),
        controller_state=PolicyController(max_windows=K).state_dict(),
        variant='softstop', beta=5., config=dict(objective='class_conditional_risk')),
        tmp_path/'teacher.pt')
    torch.save(nn.Linear(128,2).state_dict(), tmp_path/'independent.pt')
    cfg = dict(objective='class_conditional_risk', max_windows=K, rho_target=.01,
        target_frr_limit=.005, controller_checkpoint=str(tmp_path/'teacher.pt'),
        independent_checkpoint=str(tmp_path/'independent.pt'),
        resource=dict(host_time_ref='train_host_mean_natural_seconds',compute_time_ref=.03,compute_weight=.01),
        risk=dict(event='command_reject',epsilon_train=.0025),
        teacher=dict(weight_initial=1.,weight_final=.1), timing={}, cost={},
        training=dict(softstop=dict(epochs=3,batch_size=4,head_only_epochs=2),
                      pg=dict(epochs=1,batch_size=4,value_warmup_epochs=1)))
    yield model, data, cfg, tmp_path
    torch.set_num_threads(previous_threads)


def test_constrained_policy_training_short_reads_and_resume(synthetic_policy):
    model, data, cfg, output = synthetic_policy
    v2 = train_policy('softstop',model,data,data,cfg,output/'v2')
    assert v2.exists()
    checkpoint = torch.load(output/'v2'/'last.pt',weights_only=False)
    assert [row['head_only'] for row in checkpoint['history'][1:]] == [True, True, False]
    assert checkpoint['history'][0]['epoch'] == 0
    assert checkpoint['dev_metrics']['n_target'] == 3  # includes the pre-window natural end
    assert checkpoint['config']['variant'] == 'softstop'
    cfg['controller_checkpoint'] = str(v2)
    v3 = train_policy('pg',model,data,data,cfg,output/'v3')
    assert v3.exists()
    assert 'value_state' in torch.load(output/'v3'/'last.pt',weights_only=False)
    cfg['training']['softstop']['epochs'] = 4
    cfg['controller_checkpoint'] = str(output/'teacher.pt')
    assert train_policy('softstop',model,data,data,cfg,output/'v2',resume=True).exists()
    resumed = torch.load(output/'v2'/'last.pt',weights_only=False)
    assert resumed['epoch'] == 4
    assert len(resumed['history']) == 5


@pytest.mark.parametrize('source', ['teacher', 'legacy'])
def test_pg_requires_constrained_v2_checkpoint(synthetic_policy, source):
    model, data, cfg, output = synthetic_policy
    checkpoint = torch.load(output/'teacher.pt',weights_only=False)
    if source == 'legacy':
        checkpoint['config'] = dict(variant='softstop',objective='weighted_cost')
        torch.save(checkpoint, output/'teacher.pt')
    with pytest.raises(ValueError, match='constrained V2'):
        train_policy('pg',model,data,data,cfg,output/'pg')


def test_pg_accepts_constrained_v2_epoch_zero(synthetic_policy):
    model, data, cfg, output = synthetic_policy
    cfg['training']['softstop']['epochs'] = 0
    v2 = train_policy('softstop',model,data,data,cfg,output/'v2')
    assert torch.load(v2,weights_only=False)['epoch'] == 0
    cfg['controller_checkpoint'] = str(v2)
    cfg['training']['pg'].update(epochs=0,value_warmup_epochs=0)
    assert train_policy('pg',model,data,data,cfg,output/'pg').exists()
