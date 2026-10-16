import numpy as np
import pytest
from poreloom.hostfloor import hostfloor_key, hard_components, paired_summary


def test_host_floor_overrides_lower_target_error_and_ties_use_time():
    def m(host, target, time):
        return dict(host_effective_reject_rate=host, target_command_frr=target,
                    host_mean_pore_seconds=time, mean_compute_seconds=.01)
    assert hostfloor_key(m(.93, .03, 2), .93) < hostfloor_key(m(.929, 0, 1), .93)
    assert hostfloor_key(m(.93, .02, 3), .93) < hostfloor_key(m(.95, .03, 1), .93)
    assert hostfloor_key(m(.93, .02, 2), .93) < hostfloor_key(m(.95, .02, 3), .93)


def test_unreachable_windows_end_naturally_and_defer_uses_later_branch():
    branches = {k: np.zeros((3, 2, 2)) for k in ('command_reject', 'effective_reject', 'pore_seconds')}
    branches['command_reject'][:, :, 1] = 1
    branches['effective_reject'][:, :, 1] = 1
    branches['pore_seconds'][:, :, 0] = 20
    branches['pore_seconds'][:, 0, 1] = .85
    branches['pore_seconds'][:, 1, 1] = 1.23
    ends = {'command_reject': np.zeros(3), 'effective_reject': np.zeros(3), 'pore_seconds': np.array([.5, 20, 20])}
    actions = np.array([[1, 1], [2, 1], [0, 1]])
    reachable = np.array([[False, False], [True, True], [True, True]])
    got = hard_components(actions, reachable, branches, ends)
    np.testing.assert_allclose(got['pore_seconds'], [.5, 1.23, 20])
    np.testing.assert_array_equal(got['command_reject'], [0, 1, 0])


def test_pairing_separates_command_rescue_from_effective_rescue():
    before = [dict(read_id='h', is_target=False, effective_reject=False, command_reject=False),
              dict(read_id='t', is_target=True, effective_reject=False, command_reject=True)]
    after = [dict(read_id='t', is_target=True, effective_reject=False, command_reject=False),
             dict(read_id='h', is_target=False, effective_reject=True, command_reject=True)]
    got = paired_summary(before, after, bootstrap=100, seed=1)
    assert got['host_gained'] == 1 and got['host_lost'] == 0
    assert got['target_command_rescued'] == 1 and got['target_effective_rescued'] == 0
    assert got['target_command_delta'] == -1 and got['host_delta'] == 1
    assert got['target_command_improvement'] is False
    with pytest.raises(ValueError):
        paired_summary(before, after[:-1])


def test_no_discordant_hosts_is_inconclusive_not_proof_of_noninferiority():
    rows = [dict(read_id='h', is_target=False, effective_reject=True, command_reject=True),
            dict(read_id='t', is_target=True, effective_reject=False, command_reject=False)]
    got = paired_summary(rows, rows, bootstrap=100)
    assert got['host_noninferiority_delta0'] is False
    assert got['host_interval_degenerate'] is True
