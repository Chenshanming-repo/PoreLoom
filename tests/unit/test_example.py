"""Subset replay with stored per-read phases (example/), and the example's own data and results."""
import numpy as np
import pytest

from poreloom.method import head as H
from poreloom.method import native_policy as P
from poreloom.method.data import subset
from poreloom.prefix_data import PrefixSet


def _reads(n=24, length=9000, seed=0):
    rng = np.random.default_rng(seed)
    target = np.arange(n) % 2 == 1
    raw_length = rng.choice([3000, 5200, 8000, 30000], n)
    adc = np.zeros((n, length), np.int16)
    for i in range(n):
        level = 520 if target[i] else 470
        k = min(length, raw_length[i])
        adc[i, :k] = (level + rng.normal(0, 25, k)).astype(np.int16)
    cal = np.tile([-240., 1400., 8192.], (n, 1))
    return PrefixSet(adc, target, raw_length, cal, np.array([f'r{i}' for i in range(n)]), np.full(n, 'test'))


ROWS = np.array([1, 4, 5, 11, 17, 22])


def test_extract_with_stored_phases_reproduces_full_group_rows(tmp_path):
    from test_method_runtime_config import checkpoint
    model, _ = checkpoint(tmp_path, fs=4000., budget=7500)
    model.eval()
    data = _reads()
    full = H.extract(model, data, 4000., .4, 20261012, 'cpu')
    phase = np.random.default_rng(20261012).random(len(data))[ROWS]
    part = H.extract(model, subset(data, ROWS), 4000., .4, None, 'cpu', phase=phase)
    assert np.array_equal(part['m'], full['m'][ROWS], equal_nan=True)
    assert np.array_equal(part['ell'], full['ell'][ROWS], equal_nan=True)


def test_extract_rejects_phase_of_wrong_length_or_range(tmp_path):
    from test_method_runtime_config import checkpoint
    model, _ = checkpoint(tmp_path, fs=4000., budget=7500)
    data = _reads(4)
    with pytest.raises(ValueError):
        H.extract(model, data, 4000., .4, None, 'cpu', phase=np.full(3, .5))
    with pytest.raises(ValueError):
        H.extract(model, data, 4000., .4, None, 'cpu', phase=np.array([.1, .2, 1., .3]))


def test_native_policy_with_stored_phases_reproduces_full_group_rows():
    rng = np.random.default_rng(1)
    n = 40
    data = dict(read_id=np.array([f'r{i}' for i in range(n)]), raw_length=rng.choice([3000, 4700, 6000, 20000], n),
                is_target=np.arange(n) % 2 == 1)
    margin = rng.normal(0, 1, n)
    score = dict(read_id=data['read_id'], margin=margin, native_reject=margin > 0)
    full = P.native_policy(score, data, service=.004, raw_end=4500)
    rows = np.array([0, 3, 8, 9, 21, 39])
    pick = lambda d: {k: np.asarray(v)[rows] for k, v in d.items()}
    phase = P.phase_uniform(n)[rows]
    part = P.native_policy(pick(score), pick(data), service=.004, raw_end=4500, phase=phase)
    for k in full:
        assert np.array_equal(part[k], full[k][rows], equal_nan=True), k
    with pytest.raises(ValueError):
        P.native_policy(pick(score), pick(data), service=.004, raw_end=4500, phase=phase[:-1])


# --- example/ ----------------------------------------------------------------------------------------------------
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[2]/'example'
needs_example = pytest.mark.skipif(not (EXAMPLE/'data'/'test'/'adc.npy').exists(), reason='example data not present')


def _example():
    import importlib.util
    spec = importlib.util.spec_from_file_location('poreloom_example', EXAMPLE/'run_example.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _reference(name, rows):
    with np.load(EXAMPLE/'reference'/'paper_outcomes.npz', allow_pickle=False) as z:
        return {k: z[f'{name}__{k}'][rows] for k in ('command', 'command_reject', 'effective_reject', 'pore_seconds',
                                                       'decision_seconds')}


def _same(o, ref):
    for k, b in ref.items():
        a = np.asarray(o[k])
        assert (np.array_equal(a, b) if a.dtype == bool else np.allclose(a, b, rtol=0, atol=1e-6, equal_nan=True)), k


@needs_example
def test_example_data_is_balanced_and_carries_paper_phases():
    ex = _example()
    data, phase = ex.load_data()                      # verifies the manifest hashes
    with np.load(EXAMPLE/'data'/'test'/'metadata.npz') as z:
        source_row = z['source_row']
    assert len(data) == 1000 and int(np.sum(data.is_target)) == 500
    assert np.all(np.diff(source_row) > 0)
    assert np.array_equal(phase, P.phase_uniform(30000)[source_row])


@needs_example
def test_example_poreloom_reproduces_paper_outcomes():
    ex = _example()
    data, phase = ex.load_data()
    rows = np.arange(48)
    o = ex.run_poreloom(subset(data, rows), phase[rows], 'cpu')
    _same(o, _reference('poreloom', rows))


@needs_example
def test_example_baseline_reproduces_paper_outcomes():
    from poreloom.paths import THIRD_PARTY
    if not (THIRD_PARTY/'NanoDeep/read_deep/model/nanodeep.py').is_file():
        pytest.skip('upstream NanoDeep source not installed (run example/run_example.py once)')
    ex = _example()
    data, phase = ex.load_data()
    rows = np.arange(48)
    o = ex.run_baseline('nanodeep', subset(data, rows), phase[rows], 'cpu')
    _same(o, _reference('nanodeep', rows))


def test_example_reads_are_ones_where_poreloom_does_better():
    ex = _example()
    y = np.array([False, False, True, True, False, False])     # host, host, target, target, host, host

    def outcome(reject, when):
        reject = np.array(reject, bool)
        return dict(command=np.ones(6, bool), command_reject=reject, effective_reject=reject,
                    decision_seconds=np.array(when, float), pore_seconds=np.ones(6))
    outcomes = dict(poreloom=outcome([1, 1, 0, 0, 1, 1], [.9, .5, 1., .8, .6, 1.2]),
                    readcurrent=outcome([0, 0, 1, 1, 1, 1], [1.3]*6),
                    nanodeep=outcome([0, 0, 1, 0, 1, 1], [1.3]*6),
                    squigglenet=outcome([0, 1, 0, 1, 1, 1], [1.3]*6))
    data = PrefixSet(np.zeros((6, 10), np.int16), y, np.full(6, 20000), np.tile([0., 1., 1.], (6, 1)),
                     np.array([f'r{i}' for i in range(6)]), np.full(6, 'test'))
    picks = [i for i, _ in ex.pick_examples(data, outcomes)]
    # host only PoreLoom rejects (r0; r1 is also rejected by SquiggleNet); target kept by PoreLoom with the most
    # baseline rejections, earliest PoreLoom keep on ties (r2 and r3 both have 2; r3 at 0.8 s); host every method
    # rejects with the widest PoreLoom lead (r4 over r5)
    assert picks == [0, 3, 4]
