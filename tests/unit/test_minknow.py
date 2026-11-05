"""poreloom.minknow: the MinKNOW live-read adapter (template, validated on Icarust only).

Signal mapping, read bookkeeping, the live client against a scripted in-process stream, position selection, and
equality of the packaged PoreLoom policy with the Icarust wrapper used for the paper's closed-loop runs."""
import importlib.util
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from poreloom.minknow.reads import ReadBook
from poreloom.minknow.signal import adc_mapping, to_model_adc

MODEL = dict(offset=13.464453946637898, range=1536.598389, digitisation=8192.)


# ---------------------------------------------------------------- signal mapping
def test_mapped_adc_reproduces_each_channels_picoamperes():
    device = dict(offsets=[-240., 10., 3.5], pa_ranges=[1400., 1536.6, 1700.], digitisation=2048.)
    gain, shift = adc_mapping(device, MODEL)
    raw = np.random.default_rng(0).integers(-500, 2000, (3, 200)).astype(np.int16)
    step = MODEL['range']/MODEL['digitisation']
    for c in range(3):
        mapped = to_model_adc(raw[c], gain[c], shift[c])
        assert mapped.dtype == np.int16
        pa_model = (mapped + MODEL['offset'])*step
        pa_device = (raw[c] + device['offsets'][c])*device['pa_ranges'][c]/device['digitisation']
        assert np.max(np.abs(pa_model - pa_device)) <= step/2 + 1e-9


def test_uniform_calibration_matches_the_simloop_client():
    src = dict(offset=4., range=1500., digitisation=8192.)
    gain, shift = adc_mapping(dict(offsets=[src['offset']]*4, pa_ranges=[src['range']]*4,
                                   digitisation=src['digitisation']), MODEL)
    # experiments/20261011_simloop/closed_loop.py Client.__init__ / on_packet
    g = (src['range']/src['digitisation'])/(MODEL['range']/MODEL['digitisation'])
    s = src['offset']*g - MODEL['offset']
    raw = np.arange(-300, 900, 7, dtype=np.int16)
    for c in range(4):
        np.testing.assert_array_equal(to_model_adc(raw, gain[c], shift[c]), np.rint(raw*g + s).astype(np.int16))


def test_out_of_range_samples_saturate_instead_of_wrapping():
    gain, shift = np.array([8.]), np.array([0.])
    out = to_model_adc(np.array([5000, -5000], np.int16), gain[0], shift[0])
    assert out.tolist() == [32767, -32768]


# ---------------------------------------------------------------- read bookkeeping
def first_window(s):
    return 400 if s['k'] == 0 else None


def chunk(start, n):
    return np.arange(start, start + n, dtype=np.int16)


def test_read_becomes_ready_once_its_window_has_arrived():
    book = ReadBook(first_window, buffer_samples=1000)
    assert book.add(1, 'r1', 1, 0, 0, chunk(0, 300), (83,), .1) is None
    s = book.add(1, 'r1', 1, 0, 300, chunk(300, 300), (83,), .2)
    assert s is book.current(1) and s['received'] == 600 and s['pending'] and s['ready_at'] == .2
    np.testing.assert_array_equal(s['buf'][:600], np.arange(600))
    assert book.add(1, 'r1', 1, 0, 600, chunk(600, 300), (83,), .3) is None       # already queued


def test_overlapping_chunk_is_trimmed():
    book = ReadBook(first_window, buffer_samples=1000)
    book.add(2, 'r', 5, 100, 100, chunk(0, 300), (83,), 0.)
    s = book.add(2, 'r', 5, 100, 300, chunk(200, 300), (83,), 1.)
    assert s['received'] == 500 and book.stats['overlaps'] == 1
    np.testing.assert_array_equal(s['buf'][:500], np.arange(500))


def test_gap_marks_the_read_and_never_queues_it():
    book = ReadBook(first_window, buffer_samples=1000)
    book.add(3, 'r', 1, 0, 0, chunk(0, 300), (83,), 0.)
    assert book.add(3, 'r', 1, 0, 400, chunk(400, 600), (83,), 1.) is None
    assert book.current(3)['decided'] == 'gap' and book.stats['gaps'] == 1


def test_new_read_closes_the_previous_one_and_stale_packets_are_dropped():
    closed = []
    book = ReadBook(first_window, buffer_samples=1000, on_close=lambda s, why: closed.append((s['read_id'], why)))
    book.add(4, 'old', 1, 0, 0, chunk(0, 300), (83,), 0.)
    book.add(4, 'new', 2, 900, 900, chunk(0, 300), (83,), 1.)
    assert closed == [('old', 'replaced')] and book.current(4)['read_id'] == 'new'
    assert book.add(4, 'old', 1, 0, 300, chunk(300, 300), (83,), 2.) is None
    assert book.stats['stale_packets'] == 1 and book.current(4)['received'] == 300


def test_reads_outside_the_accepted_classifications_are_left_alone():
    book = ReadBook(first_window, buffer_samples=1000, accepted={83})
    assert book.add(5, 'pore', 1, 0, 0, chunk(0, 600), (80,), 0.) is None
    assert book.current(5)['decided'] == 'skipped_class' and book.stats['skipped_class'] == 1
    assert book.add(6, 'strand', 1, 0, 0, chunk(0, 600), (83,), 0.) is not None


def test_buffer_keeps_the_prefix_and_counts_every_sample():
    book = ReadBook(lambda s: None, buffer_samples=500)
    book.add(7, 'r', 1, 0, 0, chunk(0, 300), (83,), 0.)
    s = book.add(7, 'r', 1, 0, 300, chunk(300, 300), (83,), 1.) or book.current(7)
    assert s['received'] == 600 and len(s['buf']) == 500
    np.testing.assert_array_equal(s['buf'], np.arange(500))


# ---------------------------------------------------------------- live client against a scripted stream
class FakeCall:
    """In-process stand-in for the bidirectional get_live_reads call: consumes the client's requests on a thread,
    yields scripted responses (or packets until cancelled when script is None), then one ack per action."""

    def __init__(self, requests, script, wait_actions):
        import grpc
        self.grpc, self.script, self.wait_actions = grpc, script, wait_actions
        self.setup = next(requests).setup
        self.actions, self.cancelled = [], threading.Event()
        self._requests = requests
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        for r in self._requests:
            self.actions.extend(r.actions.actions)

    def cancel(self):
        self.cancelled.set()

    def __iter__(self):
        from minknow_api import data_pb2
        if self.script is None:
            n = 0
            while not self.cancelled.is_set():
                yield packet({1: ('r', 1, 0, n*300, 300)}); n += 1; time.sleep(.01)
            raise self.grpc.RpcError('cancelled')
        for response in self.script:
            if self.cancelled.is_set():
                raise self.grpc.RpcError('cancelled')
            yield response; time.sleep(.01)
        deadline = time.time() + 5
        while len(self.actions) < self.wait_actions and time.time() < deadline:
            if self.cancelled.is_set():
                raise self.grpc.RpcError('cancelled')
            time.sleep(.01)
        acks = [data_pb2.GetLiveReadsResponse.ActionResponse(action_id=a.action_id, response=0) for a in self.actions]
        yield data_pb2.GetLiveReadsResponse(action_responses=acks)


class FakeConnection:
    def __init__(self, script, wait_actions=0):
        from types import SimpleNamespace
        self.script, self.wait_actions, self.call = script, wait_actions, None
        self.data = SimpleNamespace(get_live_reads=self._open)

    def _open(self, requests):
        self.call = FakeCall(requests, self.script, self.wait_actions)
        return self.call


def packet(reads, classification=83):
    """reads: {channel: (read_id, number, start_sample, chunk_start_sample, n)}; samples count up from 0."""
    from minknow_api import data_pb2
    ReadData = data_pb2.GetLiveReadsResponse.ReadData
    channels = {}
    for ch, (rid, number, start, chunk_start, n) in reads.items():
        raw = np.arange(chunk_start - start, chunk_start - start + n, dtype='<i2').tobytes()
        channels[ch] = ReadData(id=rid, number=number, start_sample=start, chunk_start_sample=chunk_start,
                                chunk_length=n, chunk_classifications=[classification], raw_data=raw)
    return data_pb2.GetLiveReadsResponse(channels=channels)


class OddEven:
    """Reject odd channels, accept even ones, once 400 samples have arrived; records what it saw."""
    max_samples = 1000

    def __init__(self, fail=False):
        self.fail, self.seen, self.released = fail, {}, []

    def needed(self, s):
        return 400 if s['k'] == 0 else None

    def decide(self, jobs):
        if self.fail:
            raise ValueError('policy failure')
        out = []
        for s in jobs:
            s['k'] = 1; self.seen[s['channel']] = s['buf'][:s['received']].copy()
            out.append((s, 'reject' if s['channel'] % 2 else 'accept', dict(margin=0.)))
        return out

    def release(self, s):
        self.released.append(s['read_id'])


def two_reads():
    return [packet({1: ('a', 1, 0, 0, 300), 2: ('b', 1, 50, 50, 300)}),
            packet({1: ('a', 1, 0, 300, 300), 2: ('b', 1, 50, 350, 300)})]


def run_client(tmp_path, connection, policy, **kw):
    from poreloom.minknow.client import LiveClient
    pytest.importorskip('minknow_api')
    kw.setdefault('gain', np.ones(2)); kw.setdefault('shift', np.zeros(2))
    client = LiveClient(connection, policy, tmp_path, first_channel=1, last_channel=2, **kw)
    return client.run()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_reject_sends_unblock_and_accept_sends_stop_further_data(tmp_path):
    pytest.importorskip('minknow_api')
    from minknow_api import data_pb2
    conn = FakeConnection(two_reads(), wait_actions=2)
    info = run_client(tmp_path, conn, OddEven(), unblock_duration=.1)
    setup = conn.call.setup
    assert (setup.first_channel, setup.last_channel) == (1, 2)
    assert setup.raw_data_type == data_pb2.GetLiveReadsRequest.UNCALIBRATED
    actions = {a.channel: a for a in conn.call.actions}
    assert set(actions) == {1, 2}
    assert actions[1].WhichOneof('action') == 'unblock' and actions[1].unblock.duration == pytest.approx(.1)
    assert actions[2].WhichOneof('action') == 'stop_further_data'
    assert actions[1].number == 1 and actions[2].number == 1
    decisions = {d['channel']: d for d in jsonl(tmp_path/'decisions.jsonl')}
    assert decisions[1]['action'] == 'reject' and decisions[2]['action'] == 'accept'
    assert decisions[1]['action_id'] == actions[1].action_id and decisions[1]['sent'] is not None
    assert {a['action_id'] for a in jsonl(tmp_path/'acks.jsonl')} == {a.action_id for a in conn.call.actions}
    assert info['status'] == 'stream_ended' and info['stats']['actions_sent'] == 2


def test_dry_run_decides_and_logs_without_sending(tmp_path):
    conn = FakeConnection(two_reads())
    run_client(tmp_path, conn, OddEven(), dry_run=True)
    assert conn.call.actions == []
    decisions = jsonl(tmp_path/'decisions.jsonl')
    assert sorted(d['action'] for d in decisions) == ['accept', 'reject']
    assert not any('sent' in d or 'action_id' in d for d in decisions)     # analyze_live: unsent = no command


def test_policy_sees_each_channel_on_the_model_grid(tmp_path):
    policy = OddEven()
    run_client(tmp_path, FakeConnection(two_reads()), policy, dry_run=True,
               gain=np.array([1., 2.]), shift=np.array([0., 10.]))
    np.testing.assert_array_equal(policy.seen[1], np.arange(600))
    np.testing.assert_array_equal(policy.seen[2], np.arange(600)*2 + 10)


def test_replaced_and_open_reads_are_released_and_logged(tmp_path):
    script = two_reads() + [packet({1: ('c', 2, 900, 900, 300)})]
    policy = OddEven()
    run_client(tmp_path, FakeConnection(script), policy, dry_run=True)
    assert sorted(policy.released) == ['a', 'b', 'c']
    closed = {r['read_id']: r['closed'] for r in jsonl(tmp_path/'reads.jsonl')}
    assert closed == {'a': 'replaced', 'b': 'run_end', 'c': 'run_end'}


def test_run_stops_after_its_duration(tmp_path):
    t0 = time.monotonic()
    info = run_client(tmp_path, FakeConnection(None), OddEven(), dry_run=True, duration=.5)
    assert info['status'] == 'duration_reached' and time.monotonic() - t0 < 5


def test_policy_failure_ends_the_run_with_an_error(tmp_path):
    with pytest.raises(RuntimeError, match='policy failure'):
        run_client(tmp_path, FakeConnection(two_reads(), wait_actions=2), OddEven(fail=True))


# ---------------------------------------------------------------- packaged policy == Icarust wrapper
def icarust_live_module():
    repo = Path(__file__).resolve().parents[2]
    path = repo/'experiments/20261014_final/icarust_live.py'
    if not path.exists() or not (repo/'experiments/20261011_simloop/closed_loop.py').exists():
        pytest.skip('experiments/ is not part of this checkout')
    spec = importlib.util.spec_from_file_location('minknow_test_final_live', path)
    live = importlib.util.module_from_spec(spec); spec.loader.exec_module(live)
    return live


def tiny_bundle(tmp_path, budget=7500, fs=4000):
    import torch
    from poreloom.stream_controller import BucketController
    from test_method_runtime_config import checkpoint, trajectories
    _, encoder = checkpoint(tmp_path, fs=float(fs), budget=budget)
    z = trajectories(budget=budget, fs=fs, size=1)
    config = {k: z[k] for k in ('budget', 'skip', 'stride', 'min_samples', 'bucket_samples', 'n_buckets',
                                'feature_fs', 'feature_max_samples')}
    torch.manual_seed(3)
    ctrl = BucketController(6, config['n_buckets'], 8, 1.)
    with torch.no_grad():               # several looks per read; accept, reject and defer all occur
        for head in ctrl.heads[:-1]:
            head[-1].weight.mul_(4.)
            head[-1].bias[:3] = torch.tensor([1., .6, -1.5])     # u: accept, reject, defer
            head[-1].bias[5] = -2.                               # r: defer
        ctrl.heads[-1][-1].bias[:2] = torch.tensor([0., -1.])
    path = tmp_path/'bundle.pt'
    torch.save(dict(encoder=str(encoder), head_state=ctrl.state_dict(), beta=1., fs=float(fs), max_samples=budget,
                    head=dict(input_dim=6, n_buckets=ctrl.n_buckets, hidden=8, look_config=config)), path)
    return path


def replay(policy, reads, seed):
    """Feed reads as growing chunks of 50-1200 samples. Returns the (read, samples, action, margin, next needed)
    trace and the head inputs of every look, by bucket."""
    rng = np.random.default_rng(seed)
    looks, choose = [], policy.controller.actions

    def recorded(x, b):
        looks.append((b, x.cpu().numpy().copy()))
        return choose(x, b)
    policy.controller.actions = recorded
    states = [dict(channel=i, read_id=f'r{i}', read_number=1, received=0, k=0, buf=r) for i, r in enumerate(reads)]
    active, trace = list(states), []
    for r in range(10_000):
        if not active: break
        for s in active:
            s['received'] = min(len(s['buf']), s['received'] + int(rng.integers(50, 1200)))
        # even rounds: only reads at their next decision point (the live client); odd rounds: every active read,
        # so repeated looks inside one bucket must be ignored
        jobs = active if r % 2 else [s for s in active if (n := policy.needed(s)) is not None and s['received'] >= n]
        if jobs:
            for s, action, info in policy.decide(jobs):
                m = info.get('margin')
                trace.append((s['read_id'], s['received'], action, None if m is None or not np.isfinite(m) else round(m, 5),
                              policy.needed(s)))
        active = [s for s in active if policy.needed(s) is not None and not any(
            t[0] == s['read_id'] and t[2] != 'defer' for t in trace)]
    return trace, looks


def test_packaged_policy_matches_the_icarust_wrapper(tmp_path, monkeypatch):
    from poreloom.minknow.policy import PoreLoomLivePolicy
    live = icarust_live_module()
    bundle = tiny_bundle(tmp_path)
    monkeypatch.setitem(live.MODELS, 'PoreLoom', str(tmp_path))
    reference = live.PoreLoomPolicy('cpu', MODEL, capacity=16, bundle_path=bundle)
    packaged = PoreLoomLivePolicy(bundle, 'cpu', MODEL, capacity=16)
    assert packaged.max_samples == reference.max_samples == 7500
    reads = [np.random.default_rng(10 + i).integers(-200, 900, 7500).astype(np.int16) for i in range(12)]
    (expected, expected_looks), (actual, actual_looks) = replay(reference, reads, seed=4), replay(packaged, reads, seed=4)
    assert actual == expected
    assert {t[2] for t in actual} == {'accept', 'reject', 'defer'}
    assert [b for b, _ in actual_looks] == [b for b, _ in expected_looks]
    assert sum(len(x) for _, x in actual_looks) >= 2*len(reads)
    for (_, x), (_, y) in zip(actual_looks, expected_looks):
        np.testing.assert_allclose(x, y, rtol=0, atol=1e-6)
    assert packaged.stats['steps'] == reference.stats['steps']


def endpoint_needs(policy, signal, positions):
    """Next decision point after each look of one read whose head defers until the forced endpoint head."""
    import torch
    choose = policy.controller.actions
    policy.controller.actions = lambda x, b: torch.full((len(x),), 2 if b < policy.n_buckets else 0, dtype=torch.long)
    state, out = dict(channel=0, read_id='e', read_number=1, received=0, k=0), []
    try:
        for n in positions:
            state.update(buf=signal[:n], received=n)
            (_, action, _), = policy.decide([state])
            out.append((action, policy.needed(state)))
    finally:
        policy.controller.actions = choose
    return out


def test_packaged_policy_waits_for_the_endpoint_like_the_icarust_wrapper(tmp_path, monkeypatch):
    from poreloom.minknow.policy import PoreLoomLivePolicy
    live = icarust_live_module()
    bundle = tiny_bundle(tmp_path)
    monkeypatch.setitem(live.MODELS, 'PoreLoom', str(tmp_path))
    signal = np.random.default_rng(7).integers(-200, 900, 7500).astype(np.int16)
    positions = [1932, 2700, 4500, 6800, 7499, 7500]
    expected = endpoint_needs(live.PoreLoomPolicy('cpu', MODEL, capacity=4, bundle_path=bundle), signal, positions)
    actual = endpoint_needs(PoreLoomLivePolicy(bundle, 'cpu', MODEL, capacity=4), signal, positions)
    assert actual == expected
    assert actual[-2] == ('defer', 7500) and actual[-1][0] == 'accept'


# ---------------------------------------------------------------- command line helpers
def test_position_is_chosen_by_name_or_when_it_is_the_only_one():
    from types import SimpleNamespace
    from poreloom.minknow.cli import pick_position
    a, b = SimpleNamespace(name='MN1'), SimpleNamespace(name='MN2')
    assert pick_position([a, b], 'MN2') is b and pick_position([a], None) is a
    with pytest.raises(ValueError, match='MN1.*MN2'):
        pick_position([a, b], None)
    with pytest.raises(ValueError, match='X1'):
        pick_position([a, b], 'X1')


def test_classification_names_resolve_to_ids():
    from poreloom.minknow.cli import classification_ids
    table = {83: 'strand', 80: 'pore', 84: 'adapter'}
    assert classification_ids(table, ['strand', 'adapter']) == {83, 84}
    assert classification_ids(table, ['all']) is None
    with pytest.raises(ValueError, match='pore'):
        classification_ids(table, ['strand', 'tether'])


def test_device_calibration_is_read_per_channel():
    from types import SimpleNamespace
    from poreloom.minknow.cli import device_calibration
    calls = []

    def get_calibration(first_channel, last_channel):
        calls.append((first_channel, last_channel))
        n = last_channel - first_channel + 1
        return SimpleNamespace(offsets=[float(i) for i in range(n)], pa_ranges=[1400.]*n, digitisation=2048.,
                               has_calibration=True)
    conn = SimpleNamespace(device=SimpleNamespace(get_calibration=get_calibration))
    cal = device_calibration(conn, 5, 8)
    assert calls == [(5, 8)] and cal['offsets'] == [0., 1., 2., 3.] and cal['digitisation'] == 2048.
    uncalibrated = SimpleNamespace(device=SimpleNamespace(get_calibration=lambda **k: SimpleNamespace(
        offsets=[0.], pa_ranges=[1.], digitisation=1., has_calibration=False)))
    with pytest.raises(ValueError, match='calibrat'):
        device_calibration(uncalibrated, 1, 1)


def test_state_bank_holds_every_channel_twice():
    from poreloom.minknow.cli import bank_capacity
    assert bank_capacity(512) == 2048 and bank_capacity(3000) == 6000
