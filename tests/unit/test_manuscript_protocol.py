"""Prevent accidental return to the pre-revision fitting and decision protocol."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.special import logit

from poreloom.method import data as D, head as H, pipeline, baselines, thresholds
from poreloom.stream_encoder import build_stream_r0


def test_heads_fit_training_reads_with_validation_early_stopping(tmp_path, monkeypatch):
    train = SimpleNamespace(read_id=np.array(['train-host', 'train-target']))
    validation = SimpleNamespace(read_id=np.array(['validation-host', 'validation-target']))
    monkeypatch.setattr(D, 'load_split', lambda root, split: train if split == 'train' else pytest.fail(split))
    monkeypatch.setattr(D, 'validation', lambda root: validation)
    monkeypatch.setattr(pipeline, '_encoder', lambda *a, **kw: object())
    monkeypatch.setattr(H, 'extract', lambda net, data, fs, interval, seed, device: (data, seed))
    captured = {}
    class StopAfterFit(Exception):
        pass
    def fit(fit, sel, *args, **kwargs):
        captured.update(fit=fit, sel=sel, kwargs=kwargs)
        raise StopAfterFit
    monkeypatch.setattr(H, 'train_heads', fit)
    args = SimpleNamespace(out=tmp_path, data=tmp_path/'data', device='cpu', fs=5000,
                           feature_fs=4000, max_samples=7500, interval=.4, replicate=2)
    with pytest.raises(StopAfterFit):
        pipeline.head(args)
    assert [pair[0] for pair in captured['fit']] == [train, train]
    assert captured['sel'][0] is validation
    assert captured['kwargs']['betas'] == [1.0]
    assert captured['kwargs']['train_seed'] == H.HEAD['train_seed'] + 2


def test_baseline_setup_never_scores_or_selects_a_threshold(tmp_path, monkeypatch):
    from test_method import _prefix
    validation = _prefix(4)
    monkeypatch.setattr(D, 'validation', lambda _: validation)
    monkeypatch.setattr(baselines.native, 'measure_service', lambda *a, **kw: {'p50_seconds': .012})
    monkeypatch.setattr(baselines.native, 'score', lambda *a: pytest.fail('fixed threshold must not be fitted'))
    monkeypatch.setattr(baselines.policy, 'select_native', lambda *a: pytest.fail('no operating point selection'))
    baselines.select(SimpleNamespace(out=tmp_path, data=tmp_path/'data', tool='readcurrent', device='cpu'))
    import json
    saved = json.loads((tmp_path/'selection.json').read_text())
    assert saved['rules'] == {'native': {'threshold': None, 'probability_threshold': .5}}


def test_feature_scale_is_independent_of_sampling_rate_and_budget():
    model = build_stream_r0(median=0, mad=1, max_samples=19500, fs=4000,
                            feature_max_samples=7500, time_features=True)
    with torch.no_grad():
        model.time_head.weight.copy_(torch.eye(2))
        model.time_head.bias.zero_()
    pooled = torch.zeros(1, 512)
    used = torch.tensor([18000])
    expected = model.head(model.projection(pooled)) + torch.tensor([[19500/7500, np.log1p(19500/4000)]])
    torch.testing.assert_close(model.logits(pooled, used), expected.float())


def test_symmetric_thresholds_force_half_probability_and_retain_short_reads():
    probabilities = np.array([[.001, .5], [.999, .5], [.5, .49], [.5, .5], [.5, .01]])
    z = dict(m=np.tile([2000., 7500.], (5, 1)), ell=logit(probabilities),
             raw_length=np.array([16000, 16000, 16000, 16000, 7000]),
             is_target=np.array([False, True, False, True, False]), budget=7500)
    out, endpoint = thresholds.evaluate(z, 4000, .01)
    np.testing.assert_array_equal(out['command_reject'], [True, False, True, False, False])
    np.testing.assert_array_equal(out['command'], [True, True, True, True, False])
    np.testing.assert_array_equal(endpoint, [False, False, True, True, False])
    assert endpoint.mean() == .4  # denominator includes the short, undecided read
    assert out['pore_seconds'][-1] == 7000/4000


def test_symmetric_thresholds_assess_each_length_bucket_only_once():
    # 2000 and 2100 are in the same head interval. The second, confident
    # rejection must be ignored; the final native decision accepts the read.
    z = dict(m=np.array([[2000., 2100., 7500.]]), ell=logit([[.5, .001, .8]]),
             raw_length=np.array([16000]), is_target=np.array([True]), budget=7500)
    out, endpoint = thresholds.evaluate(z, 4000, .01)
    assert out['command'][0] and not out['command_reject'][0]
    assert endpoint[0] and out['steps'][0] == 2


@pytest.mark.parametrize('delta', [0, .5, -1, float('nan')])
def test_symmetric_threshold_rejects_invalid_delta(delta):
    with pytest.raises(ValueError, match='delta'):
        thresholds.evaluate({}, 4000, delta)
