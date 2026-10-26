"""poreloom.method: vendored backbone, normalization equivalence, and a tiny end-to-end CPU run."""
import importlib.util
import json

import numpy as np
import pytest
import torch

from poreloom.method import data as D
from poreloom.method import head as H
from poreloom.method.backbone import window_classifier
from poreloom.method.pretrain import STAGEA, encoder_windows, train_window_encoder
from poreloom.paths import THIRD_PARTY
from poreloom.prefix_data import PrefixSet


def _upstream():
    path = THIRD_PARTY/'ReadCurrent/models/ReadCurrent.py'
    spec = importlib.util.spec_from_file_location('_upstream_rc', path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.ReadCurrent([32, 64, 128, 256, 512], n_fc_neurons=2048, depth=29, shortcut=True)


def test_vendored_backbone_matches_upstream():
    if not (THIRD_PARTY/'ReadCurrent/models/ReadCurrent.py').is_file():
        pytest.skip('upstream ReadCurrent source not installed')
    torch.manual_seed(7); ours = window_classifier()
    torch.manual_seed(7); theirs = _upstream()
    a, b = ours.state_dict(), theirs.state_dict()
    assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
    ours.eval(); theirs.eval(); x = torch.randn(3, 3000)
    assert torch.equal(ours(x), theirs(x))


def test_mad_normalize_matches_existing_implementations():
    from poreloom.baselines.preprocessing import readcurrent_inputs
    from poreloom.windowing import normalize_windows
    rng = np.random.default_rng(0)
    w3 = rng.integers(300, 700, (5, 3000)).astype(np.int16); w3[0, 10] = 3000; w3[1] = 5
    z, ok = D.mad_normalize(w3); zr, okr = readcurrent_inputs(w3)
    assert np.array_equal(ok, okr) and np.array_equal(z, zr)
    w1 = w3[:, :1500]
    z, ok = D.mad_normalize(w1); zw, okw = normalize_windows(w1)
    assert np.array_equal(ok, okw) and np.array_equal(z[ok], zw[okw])


def _prefix(n=48, length=9000, seed=0):
    """Two classes with different current levels; some short reads."""
    rng = np.random.default_rng(seed)
    target = np.arange(n) % 2 == 1
    raw_length = np.where(np.arange(n) % 7 == 0, 4000, 20000)
    adc = np.zeros((n, length), np.int16)
    for i in range(n):
        level = 520 if target[i] else 470
        sig = level + rng.normal(0, 25, length) + 30*np.sin(np.arange(length)/(9 + 5*target[i]))
        adc[i, :min(length, raw_length[i])] = sig[:min(length, raw_length[i])].astype(np.int16)
    cal = np.tile([-240., 1400., 8192.], (n, 1))
    return PrefixSet(adc, target, raw_length, cal, np.array([f'r{seed}_{i}' for i in range(n)]), np.full(n, 'x'))


def test_inputs_and_windows():
    d = _prefix()
    w, valid = encoder_windows(d)
    assert w.shape == (len(d), 8, 1500)
    # 9000-sample prefix: windows 1500-3000 ... 7500-9000 (five) for long reads, two for 4000-sample reads
    assert np.array_equal(valid.sum(1), np.where(d.raw_length >= 9000, 5, 1))


def test_end_to_end_cpu(tmp_path):
    from poreloom.stream_training import train_stream_r0, load_stream_r0
    train, dev_a, dev_b, test = _prefix(64, seed=1), _prefix(32, seed=2), _prefix(32, seed=3), _prefix(32, seed=4)
    sa = train_window_encoder(train, dev_a, tmp_path/'sa', 'cpu', log=lambda *a: None, epochs=1, batch_size=16)
    assert set(torch.load(sa)) >= {'model_state', 'temporary_head'}
    train_stream_r0(train, dev_a, sa, tmp_path/'st', device='cpu', log=lambda *a: None, epochs=1, batch_size=8, eval_batch=8)
    net = load_stream_r0(tmp_path/'st'/'best.pt', 'cpu')[0].eval()
    fs = 5000.
    fit = [H.extract(net, dev_a, fs, .4, s, 'cpu') for s in (1, 2)]
    sel = H.extract(net, dev_b, fs, .4, 1, 'cpu')
    choice = H.train_heads(fit, sel, fs, tmp_path/'head', 'cpu', log=lambda *a: None, betas=[1., 10.], epochs=2)
    assert choice['beta'] in (1., 10.)
    ctrl = H.load_controller(tmp_path/'head'/'controllers.pt', choice['beta'])
    z = H.extract(net, test, fs, .4, 1, 'cpu')
    o = H.evaluate(ctrl, z, fs)
    m = H.metrics(o, z['is_target'])
    assert 0 <= m['acc'] <= 1 and np.all(o['pore_seconds'] <= test.raw_length/fs + 1e-9)
    json.dumps(m)


def test_stage_a_recipe_is_a1():
    assert (STAGEA['encoder_lr'], STAGEA['new_lr'], STAGEA['epochs']) == (1e-3, 1e-3, 30)


def test_head_selection_rules():
    table = [dict(beta=1., acc=.80, target_command_frr=.20, host_command_reject=.90, host_mean_pore_seconds=1.),
             dict(beta=2., acc=.85, target_command_frr=.15, host_command_reject=.88, host_mean_pore_seconds=1.2),
             dict(beta=3., acc=.70, target_command_frr=.008, host_command_reject=.45, host_mean_pore_seconds=3.)]
    assert H.select(table)['beta'] == 2.
    assert H.select(table, 'frr:0.01')['beta'] == 3.
    with pytest.raises(ValueError): H.select(table, 'frr:0.001')


@pytest.mark.parametrize('tool,recipe', [('readcurrent', 'project'), ('nanodeep', 'project'), ('squigglenet', 'project'),
                                         ('nanodeep', 'upstream')])
def test_baseline_runner_cpu(tmp_path, tool, recipe):
    from poreloom.method import baselines, native_policy
    from poreloom.data import file_sha256
    source = {'readcurrent': 'ReadCurrent/models/ReadCurrent.py', 'nanodeep': 'NanoDeep/read_deep/model/nanodeep.py',
              'squigglenet': 'SquiggleNet/model.py'}[tool]
    if not (THIRD_PARTY/source).is_file():
        pytest.skip(f'upstream {tool} source not installed')
    for split, seed, n in (('train', 1, 48), ('dev', 2, 64), ('test', 4, 32)):
        d, out = _prefix(n, seed=seed), tmp_path/'data'/split
        out.mkdir(parents=True); np.save(out/'adc.npy', d.adc)
        np.savez(out/'metadata.npz', read_id=d.read_id, is_target=d.is_target, raw_length=d.raw_length, role=np.full(n, f'x_{split}'),
                 offset=d.calibration[:, 0], range=d.calibration[:, 1], digitisation=d.calibration[:, 2])
        (out/'manifest.json').write_text(json.dumps(dict(files={f: file_sha256(out/f) for f in ('adc.npy', 'metadata.npz')})))
    baselines.TRAINING.update(epochs=1, batch_size=16, eval_batch=16); baselines.UPSTREAM['nanodeep'].update(epochs=2)
    try:
        baselines.main(['--tool', tool, '--data', str(tmp_path/'data'), '--out', str(tmp_path/'run'), '--fs', '5000', '--device', 'cpu',
                        '--recipe', recipe])
    finally:
        baselines.TRAINING.update(epochs=100, batch_size=128, eval_batch=128); baselines.UPSTREAM['nanodeep'].update(epochs=30)
    m = json.loads((tmp_path/'run'/'test'/'metrics.json').read_text())
    assert native_policy.FS == 5000. and {r['rule'] for r in m['rows']} == {'native', 'max_acc'}
    assert {(r['interval'], r['service']) for r in m['rows']} == {(.4, 'measured')}
    o = np.load(tmp_path/'run'/'test'/'outcomes_max_acc_0.4_measured.npz')
    assert np.all(o['pore_seconds'] <= np.load(tmp_path/'data'/'test'/'metadata.npz')['raw_length']/5000 + 1e-9)
