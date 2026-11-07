"""FACP host pore-time normalization: one fixed T_norm for every dataset (13.93 s = mean natural duration of
80000 training host reads, 20000 drawn once from each of the four datasets); t_norm=None restores the per-dataset
training mean."""
import argparse
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from poreloom.method import data as D, head as H, pipeline
from test_method_runtime_config import trajectories

NATURAL = 24000/5000          # trajectories(): every read is 24000 samples at 5 kHz


def test_default_t_norm_is_fixed():
    assert H.HEAD['t_norm'] == 13.93


def test_train_heads_default_uses_fixed_t_norm(tmp_path):
    z = trajectories()
    choice = H.train_heads([z, z], z, 5000, tmp_path, 'cpu', log=lambda *a: None, epochs=1, hidden=4)
    assert torch.load(tmp_path/'controllers.pt')['t_norm'] == 13.93
    assert choice['t_norm'] == 13.93


def test_train_heads_none_derives_t_norm_from_training_hosts(tmp_path):
    z = trajectories()
    choice = H.train_heads([z, z], z, 5000, tmp_path, 'cpu', log=lambda *a: None, epochs=1, hidden=4, t_norm=None)
    assert torch.load(tmp_path/'controllers.pt')['t_norm'] == pytest.approx(NATURAL)
    assert choice['t_norm'] == pytest.approx(NATURAL)


def test_fixed_t_norm_reaches_backward_fitting(tmp_path, monkeypatch):
    seen, original = [], H.train_backward
    def spy(fit, sel, beta, t_norm, *args, **kwargs):
        seen.append(t_norm); return original(fit, sel, beta, t_norm, *args, **kwargs)
    monkeypatch.setattr(H, 'train_backward', spy)
    z = trajectories()
    H.train_heads([z, z], z, 5000, tmp_path, 'cpu', log=lambda *a: None, epochs=1, hidden=4, t_norm=7.5)
    assert seen == [7.5]


@pytest.mark.parametrize('bad', [0., -1., float('nan'), float('inf')])
def test_invalid_t_norm_is_rejected(tmp_path, bad):
    z = trajectories()
    with pytest.raises(ValueError, match='t_norm'):
        H.train_heads([z, z], z, 5000, tmp_path, 'cpu', log=lambda *a: None, epochs=1, hidden=4, t_norm=bad)


@pytest.mark.parametrize('text, value', [('13.93', 13.93), ('train', None), ('7', 7.)])
def test_cli_t_norm_argument(text, value):
    assert pipeline.t_norm_arg(text) == value


@pytest.mark.parametrize('text', ['0', '-2', 'nan', 'mean'])
def test_cli_t_norm_argument_rejects(text):
    with pytest.raises((argparse.ArgumentTypeError, ValueError)):
        pipeline.t_norm_arg(text)


@pytest.mark.parametrize('given, expected', [('absent', 13.93), (None, None), (5., 5.)])
def test_pipeline_head_passes_t_norm(tmp_path, monkeypatch, given, expected):
    monkeypatch.setattr(D, 'load_split', lambda root, split: SimpleNamespace(read_id=np.array(['a'])))
    monkeypatch.setattr(D, 'validation', lambda root: SimpleNamespace(read_id=np.array(['b'])))
    monkeypatch.setattr(pipeline, '_encoder', lambda *a, **kw: object())
    monkeypatch.setattr(H, 'extract', lambda *a, **kw: None)
    captured = {}
    class Stop(Exception):
        pass
    def fit(*args, **kwargs):
        captured.update(kwargs); raise Stop
    monkeypatch.setattr(H, 'train_heads', fit)
    args = SimpleNamespace(out=tmp_path, data=tmp_path, device='cpu', fs=4000, feature_fs=4000, max_samples=7500,
                           interval=.4, replicate=0, **({} if given == 'absent' else {'t_norm': given}))
    with pytest.raises(Stop):
        pipeline.head(args)
    assert captured['t_norm'] == expected


@pytest.mark.parametrize('t_norm, protocol', [(None, 'manuscript_20261104'), (13.93, 'fixed_tnorm_20261106'), (5., 'fixed_tnorm_20261106')])
def test_protocol_label_follows_t_norm(t_norm, protocol):
    assert pipeline.protocol_for(t_norm) == protocol


def test_run_json_records_t_norm_and_refuses_a_different_one(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'test', lambda a: None)
    out = tmp_path/'run'
    pipeline.main(['test', '--data', str(tmp_path), '--out', str(out), '--device', 'cpu'])
    import json
    run = json.loads((out/'run.json').read_text())
    assert run['t_norm'] == 13.93 and run['protocol'] == 'fixed_tnorm_20261106'
    with pytest.raises(ValueError, match='t_norm'):
        pipeline.main(['test', '--data', str(tmp_path), '--out', str(out), '--device', 'cpu', '--t-norm', 'train'])


def test_run_directory_from_the_manuscript_refuses_the_fixed_t_norm(tmp_path, monkeypatch):
    """A run started before t_norm existed used the per-dataset mean; resuming it must not switch to 13.93 s."""
    monkeypatch.setattr(pipeline, 'test', lambda a: None)
    out = tmp_path/'run'
    pipeline.main(['test', '--data', str(tmp_path), '--out', str(out), '--device', 'cpu', '--t-norm', 'train'])
    import json
    run = json.loads((out/'run.json').read_text()); run.pop('t_norm'); (out/'run.json').write_text(json.dumps(run))
    with pytest.raises(ValueError, match='t_norm'):
        pipeline.main(['test', '--data', str(tmp_path), '--out', str(out), '--device', 'cpu'])


@pytest.mark.parametrize('path', ['experiments/20261015_ablation/ablation.py', 'experiments/20261016_budget_up/budget_up.py'])
def test_archived_drivers_keep_the_per_dataset_t_norm(path):
    """Dated October drivers document the DEV-B protocol; their train_heads calls must not inherit the fixed T_norm."""
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2]/path).read_text()
    calls = re.findall(r'H\.train_heads\((.*?)\)\n', src, flags=re.S)
    assert calls and all('t_norm=None' in c for c in calls)
