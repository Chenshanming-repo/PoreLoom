"""Chunk-arrival replay: arrival arithmetic, processing delays, and threshold rules."""
import numpy as np
import pytest

from poreloom.chunk_replay import (NO_DECISION, arrivals, available_at, calibrated_thresholds,
                                   replay, threshold_actions)
from poreloom.contracts import Action

A, R, D = int(Action.ACCEPT), int(Action.REJECT), int(Action.DEFER)


def test_available_at_and_arrivals():
    assert available_at(3000) == .75
    np.testing.assert_allclose(available_at([3000, 3001, 100], phase=200., chunk=1600.), [3400/4000, 3400/4000, 200/4000])
    np.testing.assert_allclose(available_at(3400, phase=200., chunk=1600.), 3400/4000)  # an arrival exactly at need
    grid = arrivals(np.array([0., 900.]), 1600., 4000)
    np.testing.assert_array_equal(grid[0], [0, 1600, 3200, 4800])
    np.testing.assert_array_equal(grid[1, :3], [900, 2500, 4100]) and np.isnan(grid[1, 3])


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


def test_calibrated_thresholds_take_the_nearest_lower_length():
    used = np.array([[432, 500, 527, 528, 6000], [100, 432, 9999, 600, 528]])
    got = calibrated_thresholds(used, np.array([432, 528, 6000]), np.array([.1, .2, .3]))
    np.testing.assert_array_equal(got, [[.1, .1, .1, .2, .3], [.1, .1, .3, .2, .2]])
    p = np.array([[.15, .15, .05, .25, .25]])
    actions = threshold_actions(p, np.array([[False]*4 + [True]]), got[:1], got[:1])
    np.testing.assert_array_equal(actions, [[D, D, R, D, R]])
