"""Runtime sampling rate / decision budget survive CLI, checkpoints and replay."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from poreloom.method import head as H
from poreloom.method import pipeline
from poreloom.stream_controller import BucketController
from poreloom.stream_encoder import build_stream_r0
from poreloom.stream_training import R0_DEFAULTS, load_stream_r0


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def cli(tmp_path, monkeypatch, extra=()):
    seen = {}
    monkeypatch.setattr(pipeline, 'stageA', lambda args: seen.update(vars(args)))
    pipeline.main(['stageA', '--data', str(tmp_path/'data'), '--out', str(tmp_path/'run'),
                   '--device', 'cpu', *extra])
    return seen, json.loads((tmp_path/'run'/'run.json').read_text())


@pytest.mark.parametrize('extra,fs,budget', [([], 4000, 7500),
    (['--fs', '5000', '--max-samples', '10500'], 5000, 10500),
    (['--fs', '5000', '--max-decision-seconds', '2.1'], 5000, 10500)])
def test_cli_resolves_rate_and_endpoint(tmp_path, monkeypatch, extra, fs, budget):
    args, run = cli(tmp_path, monkeypatch, extra)
    assert args['fs'] == run['fs'] == fs
    assert args['max_samples'] == run['max_samples'] == budget
    assert run['max_decision_seconds'] == pytest.approx(budget/fs)
    assert run['feature_fs'] == fs


@pytest.mark.parametrize('extra', [('--fs', '0'), ('--fs', '-1'), ('--fs', 'nan'), ('--fs', 'inf'),
    ('--max-samples', '1500'), ('--max-decision-seconds', '0'), ('--max-decision-seconds', 'nan'),
    ('--max-samples', '7500', '--max-decision-seconds', '2')])
def test_cli_rejects_invalid_configuration_before_creating_run(tmp_path, monkeypatch, extra):
    with pytest.raises(SystemExit):
        cli(tmp_path, monkeypatch, extra)
    assert not (tmp_path/'run').exists()


def test_cli_binds_existing_run_to_budget(tmp_path, monkeypatch):
    cli(tmp_path, monkeypatch, ['--max-samples', '10500'])
    with pytest.raises(ValueError, match='budget|endpoint'):
        cli(tmp_path, monkeypatch, ['--max-samples', '7500'])


def test_old_run_keeps_legacy_feature_rate(tmp_path, monkeypatch):
    out = tmp_path/'run'; out.mkdir()
    old = dict(data=str((tmp_path/'data').resolve()), fs=5000., replicate=0, select='max_acc', interval=.4)
    (out/'run.json').write_text(json.dumps(old))
    args, _ = cli(tmp_path, monkeypatch, ['--fs', '5000'])
    assert args['max_samples'] == 7500 and args['feature_fs'] == 4000


def checkpoint(tmp_path, fs=5000., budget=10500, legacy=False):
    torch.manual_seed(13)
    model = build_stream_r0(median=0., mad=1., fs=fs, max_samples=budget, time_features=True)
    with torch.no_grad():
        model.time_head.weight.copy_(torch.tensor([[1., 2.], [-2., 3.]]))
        model.time_head.bias.copy_(torch.tensor([.2, -.3]))
    config = {**R0_DEFAULTS, 'fs': fs, 'max_samples': budget}
    if legacy: config.pop('fs')
    path = tmp_path/'encoder.pt'
    torch.save(dict(model_state=model.state_dict(), config=config,
                    normalizer_prior=dict(median_pa=0., mad_pa=1., weight=config['prior_weight'])), path)
    return model, path


@pytest.mark.parametrize('legacy', [False, True])
def test_encoder_checkpoint_restores_feature_rate(tmp_path, legacy):
    original, path = checkpoint(tmp_path, legacy=legacy)
    model, _ = load_stream_r0(path)
    assert model.fs == (4000 if legacy else 5000) and model.max_samples == 10500
    pooled = torch.zeros(2, 512); used = torch.tensor([1488, 9000])
    n = 1500 + used
    phi = torch.stack([n/10500, torch.log1p(n/model.fs)], -1)
    expected = original.head(original.projection(pooled)) + original.time_head(phi)
    torch.testing.assert_close(model.logits(pooled, used), expected)


def trajectories(budget=10500, fs=5000, size=16):
    buckets = int(np.ceil((budget - 1500 - 432)/768))
    positions = np.r_[1932 + 768*np.arange(buckets), budget].astype(float)
    target = np.arange(size) % 2 == 1
    return dict(m=np.tile(positions, (size, 1)), ell=np.tile(np.where(target, 1., -1.)[:, None], (1, len(positions))),
                read_id=np.array([f'r{i}' for i in range(size)]), is_target=target,
                raw_length=np.full(size, 24000), budget=budget, skip=1500, stride=48,
                min_samples=432, bucket_samples=768, n_buckets=buckets,
                feature_fs=float(fs), feature_max_samples=budget)


def test_head_features_follow_configured_sampling_rate_and_budget():
    z = trajectories()
    looks = H.looks(z, 5000)
    np.testing.assert_allclose(looks.x[..., 1], z['m']/10500, rtol=1e-6)
    np.testing.assert_allclose(looks.x[..., 2], np.log1p(z['m']/5000), rtol=1e-6)
    assert looks.bucket[0, -1] == 12 and looks.forced[:, -1].all()
    np.testing.assert_allclose(looks.t, z['m']/5000)


def test_legacy_trajectories_keep_old_feature_definition():
    z = trajectories(budget=7500)
    z.pop('feature_fs'); z.pop('feature_max_samples')
    looks = H.looks(z, 5000)
    np.testing.assert_allclose(looks.x[..., 1], z['m']/7500, rtol=1e-6)
    np.testing.assert_allclose(looks.x[..., 2], np.log1p(z['m']/4000), rtol=1e-6)


def test_head_checkpoint_preserves_feature_contract(tmp_path):
    z = trajectories()
    choice = H.train_heads([z, z], z, 5000, tmp_path, 'cpu', log=lambda *a: None,
                           betas=[1.], epochs=1, hidden=4)
    saved = torch.load(tmp_path/'controllers.pt')
    assert saved['look_config']['feature_fs'] == 5000
    assert saved['look_config']['feature_max_samples'] == 10500
    controller = H.load_controller(tmp_path/'controllers.pt', choice['beta'])
    assert controller.n_buckets == 12 and controller.look_config == saved['look_config']
    expected = H.evaluate(controller, z, 5000)
    bare = {k: v for k, v in z.items() if k not in saved['look_config']}
    actual = H.evaluate(controller, bare, 5000)
    np.testing.assert_array_equal(actual['command_reject'], expected['command_reject'])
    np.testing.assert_allclose(actual['pore_seconds'], expected['pore_seconds'])
    with pytest.raises(ValueError, match='feature|configuration'):
        H.evaluate(controller, {**z, 'feature_fs': 4000}, 5000)


def test_cache_must_cover_configured_endpoint(tmp_path):
    from test_method import _prefix
    from poreloom.stream_training import train_stream_r0
    data = _prefix(4, length=9000)
    with pytest.raises(ValueError, match='cache|prefix|10500'):
        train_stream_r0(data, data, None, tmp_path/'run', device='cpu', log=lambda *a: None,
                        fs=5000, max_samples=10500, epochs=0)


@pytest.mark.parametrize('legacy', [False, True])
def test_online_policy_matches_offline_inputs_at_custom_endpoint(tmp_path, monkeypatch, legacy):
    repo = Path(__file__).resolve().parents[2]
    path = repo/'experiments/20261014_final/icarust_live.py'
    if not path.exists(): pytest.skip('experiments/ is not part of this checkout')
    spec = importlib.util.spec_from_file_location('runtime_final_live', path)
    live = importlib.util.module_from_spec(spec); spec.loader.exec_module(live)
    budget = 7500 if legacy else 10003
    _, encoder_path = checkpoint(tmp_path, budget=budget, legacy=legacy)
    z = trajectories(budget=budget, fs=4000 if legacy else 5000, size=1)
    config = {k: z[k] for k in ('budget','skip','stride','min_samples','bucket_samples','n_buckets',
                               'feature_fs','feature_max_samples')}
    ctrl = BucketController(6, config['n_buckets'], 4, 1.)
    bundle = tmp_path/'bundle.pt'
    head = dict(input_dim=6, n_buckets=ctrl.n_buckets, hidden=4)
    if not legacy: head['look_config'] = config
    torch.save(dict(encoder=str(encoder_path), head_state=ctrl.state_dict(), beta=1., fs=5000.,
                    max_samples=budget, head=head), bundle)
    monkeypatch.setitem(live.MODELS, 'PoreLoom', str(tmp_path))
    policy = live.PoreLoomPolicy('cpu', dict(offset=0., range=1., digitisation=1.), capacity=2, bundle_path=bundle)
    assert policy.max_samples == budget and policy.fs == 5000
    captured = []
    def actions(x, b):
        captured.append((x.cpu().numpy().copy(), b))
        return torch.full((len(x),), 0 if b == ctrl.n_buckets else 2, dtype=torch.long)
    policy.controller.actions = actions
    signal = np.random.default_rng(5).normal(0, 1, budget).astype(np.float32)
    positions = [1932, 2100, 2700, 4500, budget - 700, budget - 1, budget]
    state, margins, decisions = {}, [], []
    assert policy.needed(state) == 1932
    for n in positions:
        state.update(buf=signal[:n], received=n)
        _, decision, info = policy.decide([state])[0]
        margins.append(info['margin']); decisions.append(decision)
        if n == 1932: assert policy.needed(state) == 2700
        if n == budget - 1: assert policy.needed(state) == budget
    z.update(m=np.array([positions], float), ell=np.array([margins]))
    expected = H.looks(z, 5000)
    eligible = expected.bucket[0] >= 0
    assert len(captured) == int(eligible.sum())
    np.testing.assert_allclose(np.concatenate([x for x, _ in captured]), expected.x[0, eligible], rtol=1e-5, atol=1e-5)
    assert [b for _, b in captured] == expected.bucket[0, eligible].tolist()
    assert decisions[-1] in ('accept','reject') and captured[-1][1] == ctrl.n_buckets


@pytest.mark.parametrize('fs,seconds,budget,buckets,used', [(5000, 2.0006, 10003, 11, 8496),
                                                         (4000, 1., 4000, 3, 2496)])
def test_custom_configuration_through_complete_cli_pipeline(tmp_path, monkeypatch, fs, seconds, budget, buckets, used):
    from test_method import _prefix
    from poreloom.data import file_sha256
    from poreloom.method.pretrain import STAGEA
    for split, seed, size in (('train', 1, 16), ('dev', 2, 96), ('test', 4, 16)):
        data, out = _prefix(size, length=11000, seed=seed), tmp_path/'data'/split
        out.mkdir(parents=True)
        np.save(out/'adc.npy', data.adc)
        np.savez(out/'metadata.npz', read_id=data.read_id, is_target=data.is_target, raw_length=data.raw_length,
                 role=np.full(size, f'x_{split}'), offset=data.calibration[:, 0],
                 range=data.calibration[:, 1], digitisation=data.calibration[:, 2])
        (out/'manifest.json').write_text(json.dumps(dict(files={f: file_sha256(out/f) for f in ('adc.npy', 'metadata.npz')})))
    for config, changes in ((STAGEA, dict(epochs=1, batch_size=8)),
                            (R0_DEFAULTS, dict(epochs=1, batch_size=8, eval_batch=8)),
                            (H.HEAD, dict(epochs=1, betas=[1.], hidden=4))):
        for key, value in changes.items(): monkeypatch.setitem(config, key, value)
    run = tmp_path/'run'
    pipeline.main(['all', '--data', str(tmp_path/'data'), '--out', str(run), '--device', 'cpu',
                   '--fs', str(fs), '--max-decision-seconds', str(seconds)])
    model, saved = load_stream_r0(run/'stream'/'best.pt')
    assert model.fs == saved['config']['fs'] == fs
    assert model.max_samples == saved['config']['max_samples'] == budget
    assert max(saved['config']['dev_lengths']) == model.used(budget - model.skip) == used
    bundle = torch.load(run/'poreloom_sh_cp.pt')
    assert bundle['fs'] == fs and bundle['max_samples'] == budget
    assert bundle['head']['n_buckets'] == buckets
    assert bundle['head']['look_config']['feature_fs'] == fs
    assert bundle['head']['look_config']['feature_max_samples'] == budget
    metrics = json.loads((run/'test'/'metrics.json').read_text())
    assert metrics['fs'] == fs and metrics['max_samples'] == budget
    assert metrics['max_decision_seconds'] == pytest.approx(seconds)
