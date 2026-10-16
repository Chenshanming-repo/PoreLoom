"""M3b learned controller over streaming R0 looks: look grid, inputs, backward labels and the encoder hook."""
import numpy as np
import pytest
import torch

from poreloom.chunk_replay import NO_DECISION, arrivals
from poreloom.stream_controller import (ACCEPT, DEFER, REJECT, Looks, encode_looks, look_grid, look_inputs, next_look_used,
                                        one_look_per_bucket,
                                        node_labels, outcome, policy_actions, train_backward, trajectory)
from poreloom.stream_encoder import build_stream_r0
from test_stream_training import synthetic_prefix

SERVICE, EJECT = .0158, .1


def synthetic_looks(n=600, chunk=1600, seed=0, dim=5):
    rng = np.random.default_rng(seed)
    m = arrivals(rng.random(n)*chunk, chunk, 7500)
    n_, t, valid, forced, bucket = look_grid(m)
    is_target = rng.random(n) < .5
    ell = np.where(valid, np.where(is_target, 1., -1.)[:, None]*np.linspace(.5, 3, m.shape[1]) + rng.normal(0, 1.5, m.shape), np.nan)
    x = look_inputs(ell, n_, valid, h=rng.normal(size=m.shape + (dim,)))
    t_end = np.where(rng.random(n) < .1, rng.uniform(.3, 2., n), rng.uniform(2, 30, n))
    return Looks(t, valid, forced, bucket, x, t_end, is_target)


@pytest.mark.parametrize('chunk', [800, 1600, 4000])
def test_look_grid_buckets_increase_and_force_once(chunk):
    m = arrivals(np.random.default_rng(1).random(500)*chunk, chunk, 7500)
    n, t, valid, forced, bucket = look_grid(m)
    assert (forced.sum(1) == 1).all() and (bucket[forced] == 8).all()
    for row in range(len(m)):
        b = bucket[row][bucket[row] >= 0]
        assert (np.diff(b) > 0).all()
    used = (np.clip(n, 1500, 7500) - 1500)//48*48
    assert np.array_equal(valid, (n > 1500) & (used >= 432))
    assert np.array_equal(np.isfinite(t), n > 1500)


@pytest.mark.parametrize('chunk', [768, 800, 1600, 4000])
def test_guard_leaves_chunks_of_a_bucket_or_more_unchanged(chunk):
    m = arrivals(np.random.default_rng(2).random(500)*chunk, chunk, 7500)
    _, _, valid, _, bucket = look_grid(m)
    assert np.array_equal(one_look_per_bucket(valid, bucket), valid)


def test_guard_keeps_one_look_per_bucket_as_the_live_rule():
    rng = np.random.default_rng(3)
    steps = rng.integers(50, 2000, (300, 200))
    m = np.cumsum(np.c_[rng.integers(0, 1600, 300), steps], 1).astype(float)
    over = np.cumsum(m >= 7500, 1)
    m = np.where(over <= 1, m, np.nan)            # up to and including the first arrival >= 7500
    n, _, valid, forced, bucket = look_grid(m)
    keep = one_look_per_bucket(valid, bucket)
    used = (np.clip(n, 1500, 7500) - 1500)//48*48
    for row in range(len(m)):
        b = bucket[row][keep[row]]
        assert len(b) == len(np.unique(b)) and keep[row][forced[row]].all()
        need, live = 432, []                      # the live rule: act once used >= need, then need = next_look_used
        for j in np.flatnonzero(valid[row]):
            if used[row, j] >= need: live.append(j); need = int(next_look_used(used[row, j]))
        assert np.array_equal(np.flatnonzero(keep[row]), live)


def test_guard_on_a_1000_1000_500_sequence():
    m = np.array([[1000., 2000., 2500., 4100., 5700., 7300., 8900.]])
    n, _, valid, _, bucket = look_grid(m)
    assert bucket[0].tolist() == [-1, 0, 0, 2, 4, 6, 8]
    assert one_look_per_bucket(valid, bucket)[0].tolist() == [False, True, False, True, True, True, True]
    assert int(next_look_used(480)) == 1200 and int(next_look_used(5808)) == 6000 and int(next_look_used(1104)) == 1200


def test_trajectory_tracks_previous_minimum_and_count():
    ell = np.array([[np.nan, 2., -1., 3.]])
    valid = np.isfinite(ell)
    prev, low, count = trajectory(np.nan_to_num(ell), valid)
    np.testing.assert_array_equal(prev[0], [0, 2, 2, -1])
    np.testing.assert_array_equal(low[0], [0, 2, -1, -1])
    np.testing.assert_array_equal(count[0], [0, 1, 2, 3])


def test_backward_labels_match_the_realized_policy_outcome():
    fit, sel = synthetic_looks(seed=0), synthetic_looks(seed=1)
    ref = float(fit.t_end[~fit.is_target].mean())
    controller, p_fit, _, history = train_backward(fit, sel, 5., ref, SERVICE, EJECT, 'cpu', epochs=2, batch=128)
    assert {h['bucket'] for h in history} == set(range(9))
    assert np.array_equal(p_fit, policy_actions(controller, fit, 'cpu'))
    assert ((p_fit == NO_DECISION) == (fit.bucket < 0)).all()
    assert set(np.unique(p_fit[fit.forced])) <= {ACCEPT, REJECT}
    realized = outcome(fit, p_fit, SERVICE, EJECT)
    alive = fit.t + SERVICE < fit.t_end[:, None]
    first = np.where((fit.bucket >= 0) & alive, np.arange(fit.t.shape[1]), 99).min(1)
    checked = 0
    for b in range(9):
        rows, cols, u, r = node_labels(fit, p_fit, b, SERVICE, EJECT, ref, forced=b == 8)
        for k, (i, j) in enumerate(zip(rows, cols)):
            if j != first[i]: continue
            a = p_fit[i, j]
            expect_u = 0. if fit.is_target[i] else realized['pore_seconds'][i]/ref
            assert u[k, a] == pytest.approx(expect_u, rel=1e-6)
            assert r[k, a] == float(fit.is_target[i] and realized['command_reject'][i])
            checked += 1
    assert checked > 400


def test_defer_label_is_the_outcome_of_the_later_policy():
    fit = synthetic_looks(seed=2)
    ref = float(fit.t_end[~fit.is_target].mean())
    policy = np.where(fit.forced, ACCEPT, np.where(fit.bucket >= 0, DEFER, NO_DECISION))
    policy[(fit.bucket == 7)] = REJECT
    rows, cols, u, r = node_labels(fit, policy, 3, SERVICE, EJECT, ref)
    act = policy[rows].copy(); act[:, :] = np.where(np.arange(act.shape[1]) > cols[:, None], act, NO_DECISION)
    o = outcome(fit.subset(rows), act, SERVICE, EJECT)
    np.testing.assert_allclose(u[:, DEFER], np.where(fit.is_target[rows], 0, o['pore_seconds']/ref), rtol=1e-6)
    np.testing.assert_array_equal(r[:, DEFER], fit.is_target[rows] & o['command_reject'])
    assert (r[:, ACCEPT] == 0).all() and (r[:, REJECT] == fit.is_target[rows]).all()


def test_encode_looks_returns_the_projection_behind_the_logits():
    torch.manual_seed(0)
    model = build_stream_r0(median=95., mad=15.).eval()
    data = synthetic_prefix(n=6)
    n = np.array([[-1, 1900, 2500, 7500]]*6)
    h, ell = encode_looks(model, data, n, 'cpu', batch=4)
    logits = model.head(torch.as_tensor(np.nan_to_num(h))).detach().numpy()
    ok = np.isfinite(ell)
    expected = (n > 1500) & ((np.minimum(n, data.raw_length[:, None]) - 1500)//48*48 >= 432)
    assert np.array_equal(ok, expected) and ok.any() and np.isnan(h[~ok]).all()
    np.testing.assert_allclose(ell[ok], (logits[..., 1] - logits[..., 0])[ok], rtol=1e-4, atol=1e-5)
