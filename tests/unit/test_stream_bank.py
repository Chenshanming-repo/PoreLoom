"""StreamR0Bank (plan §10 M3) computes StreamR0's function: equal to the offline forward and to StreamR0.step under
any chunking, batch composition and slot reuse (random weights)."""
import copy

import numpy as np
import pytest
import torch

from poreloom.stream_bank import StreamR0Bank
from poreloom.stream_encoder import build_stream_r0
from test_stream_encoder import BUDGET, SKIP, TOL, check_r0, randomize_batchnorm, squiggle


@pytest.fixture(scope='module')
def r0():
    torch.manual_seed(2)
    return randomize_batchnorm(build_stream_r0(median=90., mad=10., skip=SKIP, max_samples=BUDGET, time_features=True))


def feed(bank, plans):
    """plans: {slot: (signal, cut points)}; round j sends every slot its j-th chunk (slots whose cuts ran out get
    nothing). Returns {slot: [(received, logits), ...]} after every round in which the slot took a chunk."""
    out = {slot: [] for slot in plans}
    rounds = max(len(cuts) - 1 for _, cuts in plans.values())
    for j in range(rounds):
        slots, starts, chunks = [], [], []
        for slot, (signal, cuts) in plans.items():
            if j + 1 < len(cuts):
                slots.append(slot); starts.append(cuts[j]); chunks.append(signal[cuts[j]:cuts[j + 1]])
        bank.step(slots, starts, chunks)
        for slot, row in zip(slots, bank.readout(slots).cpu()):
            out[slot].append((int(bank.received[slot]), row))
    return out


def schedules(seed):
    rng = np.random.default_rng(seed)
    signals = [squiggle(n, seed + n) for n in (7700, 3100, 5300, 1800, 6400)]
    cuts = [[0, *range(1600, len(signals[0]) + 1600, 1600)],
            [0, *sorted(rng.choice(np.arange(1, 3100), 9, replace=False)), 3100],
            [0, 1500, 1548, 1932, 1980, 2556, 2604, 5300],
            [0, 1000, 1800],
            list(range(0, 6400, 480)) + [6400]]
    return signals, cuts


def test_bank_matches_offline_r0(r0):
    bank = StreamR0Bank(r0, 8)
    signals, cuts = schedules(3)
    slots = [bank.allocate() for _ in signals]
    got = feed(bank, {s: (x, c) for s, x, c in zip(slots, signals, cuts)})
    for slot, signal in zip(slots, signals):
        check_r0(r0, signal, got[slot])
    assert torch.isnan(got[slots[3]][-1][1]).all()          # 300 samples after skip: no decision
    assert bank.used[slots[0]] == BUDGET - SKIP and bank.received[slots[0]] == 7700


def test_bank_matches_stream_step(r0):
    """Same chunks through StreamR0.step and through the bank, all reads in one batch per round."""
    bank = StreamR0Bank(r0, 8)
    signals, cuts = schedules(4)
    slots = [bank.allocate() for _ in signals]
    states = [r0.init_state() for _ in signals]
    for j in range(max(map(len, cuts)) - 1):
        live = [i for i, c in enumerate(cuts) if j + 1 < len(c)]
        chunks = [(cuts[i][j], signals[i][cuts[i][j]:cuts[i][j + 1]]) for i in live]
        r0.step([states[i] for i in live], chunks)
        bank.step([slots[i] for i in live], [c[0] for c in chunks], [c[1] for c in chunks])
        want, got = r0.readout([states[i] for i in live]), bank.readout([slots[i] for i in live])
        torch.testing.assert_close(got, want, equal_nan=True, **TOL)
        for i in live:
            s = states[i]
            assert (bank.received[slots[i]], bank.length[slots[i]], bank.frames[slots[i]], bank.tail_frames[slots[i]],
                    bank.used[slots[i]]) == (s.received, s.length, s.frames, s.tail_frames, s.used)
            torch.testing.assert_close(bank.z[slots[i], :s.length], s.z[:s.length], **TOL)


def test_bank_edges_and_slot_reuse(r0):
    bank = StreamR0Bank(r0, 2)
    signal = squiggle(9000, 15)
    a = bank.allocate()
    for start, stop in [(0, 3000), (0, 1000), (3000, 3000), (2000, 4000), (3999, 9000)]:  # repeats and overlaps
        bank.step([a], [start], [signal[start:stop]])
    ref = bank.allocate()
    bank.step([ref], [0], [signal])
    torch.testing.assert_close(bank.readout([a]), bank.readout([ref]), **TOL)
    assert bank.received[a] == 9000 and bank.length[a] == BUDGET - SKIP
    with pytest.raises(RuntimeError): bank.allocate()
    bank.release(ref)
    gap = bank.allocate()                                    # reused slot starts clean
    assert gap == ref and bank.received[gap] == 0 and bank.frames[gap] == 0
    bank.step([gap], [0], [signal[:3000]]); before = bank.readout([gap])
    bank.step([gap], [3100], [signal[3100:5000]])
    assert bank.failed[gap] and bank.received[gap] == 3000 and torch.equal(bank.readout([gap]), before)
    bad = signal[3000:4000].copy(); bad[5] = np.nan
    bank.release(gap); slot = bank.allocate()
    bank.step([slot], [0], [signal[:3000]]); bank.step([slot], [3000], [bad])
    assert bank.failed[slot]
    with pytest.raises(ValueError): bank.step([a, a], [0, 0], [signal[:10], signal[:10]])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_bank_on_cuda_matches_offline(r0):
    model = copy.deepcopy(r0).cuda()
    bank = StreamR0Bank(model, 8)
    signals, cuts = schedules(5)
    slots = [bank.allocate() for _ in signals]
    got = feed(bank, {s: (x, c) for s, x, c in zip(slots, signals, cuts)})
    assert bank.z.is_cuda
    for slot, signal in zip(slots, signals):
        check_r0(r0, signal, got[slot])


def test_ragged_forward_equals_per_row_forward(r0):
    from poreloom.stream_encoder import _ragged, ragged_lengths
    layers = copy.deepcopy(r0.backbone.layers).double()  # float64: random-weight features reach ~3e3, where float32
    lengths = [432, 433, 480, 527, 1008, 1056, 1300]       # batch-vs-single noise alone is ~2e-3
    x = torch.zeros(len(lengths), 1, max(lengths), dtype=torch.float64)
    rows = [torch.from_numpy(squiggle(n, 70 + n).astype(np.float64) - 90.)/10. for n in lengths]
    for i, row in enumerate(rows): x[i, 0, :len(row)] = row
    with torch.no_grad():
        out, count = _ragged(layers, x, torch.tensor(lengths))
        assert ragged_lengths(layers, np.array(lengths)).tolist() == count.tolist()
        for i, row in enumerate(rows):
            want = layers(row[None, None])[0]
            assert count[i] == want.shape[-1]
            torch.testing.assert_close(out[i, :, :want.shape[-1]], want, rtol=0, atol=1e-9)
