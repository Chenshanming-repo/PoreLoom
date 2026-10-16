"""Chunk-arrival replay: arrival arithmetic, equivalence with ReplayEnv in compatibility mode, threshold rules."""
import numpy as np
import pytest

from poreloom.chunk_replay import (NO_DECISION, arrivals, available_at, calibrated_thresholds, paired_mean_difference,
                                   replay, threshold_actions)
from poreloom.contracts import Action, ReadRecord
from poreloom.costs import CostConfig, TimingConfig
from poreloom.replay import ReplayEnv

A, R, D = int(Action.ACCEPT), int(Action.REJECT), int(Action.DEFER)


def test_available_at_and_arrivals():
    assert available_at(3000) == .75
    np.testing.assert_allclose(available_at([3000, 3001, 100], phase=200., chunk=1600.), [3400/4000, 3400/4000, 200/4000])
    np.testing.assert_allclose(available_at(3400, phase=200., chunk=1600.), 3400/4000)  # an arrival exactly at need
    grid = arrivals(np.array([0., 900.]), 1600., 4000)
    np.testing.assert_array_equal(grid[0], [0, 1600, 3200, 4800])
    np.testing.assert_array_equal(grid[1, :3], [900, 2500, 4100]) and np.isnan(grid[1, 3])


SERVICE = dict(preprocessing_seconds=.001, encoder_seconds=.006, gru_seconds=.0002, head_seconds=.0001)


def replay_env(raw_length, script):
    timing = TimingConfig(eject_latency_seconds=.1, **SERVICE)
    env = ReplayEnv(ReadRecord('r', False, raw_length, np.ones(4, bool)), timing, CostConfig(), max_windows=4)
    obs = env.reset()
    for action in script:
        if obs is None: break
        obs, _ = env.step(action)
    return env.outcome


@pytest.mark.parametrize('raw_length,script', [
    (20000, [R]), (20000, [A]), (20000, [D, D, R]), (20000, [D, D, D, A]), (20000, [D, D, D, R]),
    (3030, [R]),            # decision completes, eject does not
    (3010, [R]),            # read ends while the decision is in flight
    (5000, [D, D, R]),      # read ends before window 3 exists
    (2000, []),             # shorter than the first window
])
def test_compatibility_mode_matches_replay_env(raw_length, script):
    want = replay_env(raw_length, script)
    window, head = sum(SERVICE.values()) - SERVICE['head_seconds'], SERVICE['head_seconds']
    times, starts, actions, previous = [], [], [], 0.
    for k in range(1, 5):
        start = max(previous, available_at(1500 + 1500*k)); previous = start + window + head
        starts.append(start); times.append(previous); actions.append(script[k - 1] if k <= len(script) else NO_DECISION)
    got = replay(np.array([times]), np.array([actions]), np.array([starts]), (window, head), np.array([raw_length/4000]))
    assert (got['command'][0], got['effective_reject'][0]) == (want.command_issued, want.effective_reject)
    assert got['command_reject'][0] == want.command_reject
    assert got['pore_seconds'][0] == pytest.approx(want.pore_seconds, abs=1e-12)
    if want.decision_seconds is None: assert np.isnan(got['decision_seconds'][0])
    else: assert got['decision_seconds'][0] == pytest.approx(want.decision_seconds, abs=1e-12)
    assert got['compute_seconds'][0] == pytest.approx(want.compute_seconds, abs=1e-12)


def test_chunk_mode_waits_for_the_next_arrival():
    # native-style single decision needing 4500 samples; chunks at 300, 1900, 3500, 5100, ... samples
    t = available_at(4500, phase=300., chunk=1600.)
    assert t == 5100/4000
    got = replay(np.array([[t + .005]]), np.array([[R]]), np.array([[t]]), .005, np.array([10.]))
    assert got['pore_seconds'][0] == pytest.approx(5100/4000 + .005 + .1)


def test_threshold_actions():
    p = np.array([[np.nan, .2, .001, .9], [np.nan, .5, .5, .5], [np.nan, .999, .1, .1]])
    forced = np.zeros(p.shape, bool); forced[:, 3] = True
    got = threshold_actions(p, forced, reject_below=.01, final_reject_below=.6, accept_at=.99)
    np.testing.assert_array_equal(got, [[NO_DECISION, D, R, A], [NO_DECISION, D, D, R], [NO_DECISION, A, D, R]])
    out = replay(np.array([[1., 2., 3., 4.]]*3), got, np.array([[.9, 1.9, 2.9, 3.9]]*3), .01, np.full(3, 100.))
    np.testing.assert_array_equal(out['steps'], [2, 3, 1])
    np.testing.assert_array_equal(out['command_reject'], [True, True, False])
    np.testing.assert_allclose(out['compute_seconds'], [.03, .04, .02])


def test_paired_mean_difference():
    old = np.arange(10.); new = old - 1
    got = paired_mean_difference(new, old, np.ones(10, bool), reps=200)
    assert got['delta'] == -1 and got['ci'] == [-1, -1]


def test_calibrated_thresholds_take_the_nearest_lower_length():
    used = np.array([[432, 500, 527, 528, 6000], [100, 432, 9999, 600, 528]])
    got = calibrated_thresholds(used, np.array([432, 528, 6000]), np.array([.1, .2, .3]))
    np.testing.assert_array_equal(got, [[.1, .1, .1, .2, .3], [.1, .1, .3, .2, .2]])
    p = np.array([[.15, .15, .05, .25, .25]])
    actions = threshold_actions(p, np.array([[False]*4 + [True]]), got[:1], got[:1])
    np.testing.assert_array_equal(actions, [[D, D, R, D, R]])
