"""Streaming-prefix correctness: causality, chunking and batch invariance, and boundary cases."""
import numpy as np
import pytest
import torch

from poreloom.model import build_model
from poreloom.stream_encoder import build_stream_r0
from poreloom.streaming import CausalNormalizer, fresh_samples, to_pa

TOL = dict(rtol=0, atol=1e-5)
SKIP, BUDGET = 1500, 7500


def squiggle(n, seed):
    """Piecewise-constant pA levels + noise + sparse spikes (outliers)."""
    rng = np.random.default_rng(seed)
    x = np.repeat(rng.uniform(60, 120, n), rng.integers(4, 13, n))[:n] + rng.normal(0, 2, n)
    spikes = rng.choice(n, n//200, replace=False)
    x[spikes] += rng.choice([-1, 1], len(spikes))*rng.uniform(60, 100, len(spikes))
    return x.astype(np.float32)


def stream(enc, signal, cuts):
    """Feed signal[cuts[i]:cuts[i+1]]; return the state and (n, readout) after every chunk."""
    state, out = enc.init_state(), []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        enc.step([state], [(lo, signal[lo:hi])])
        out.append((hi, enc.readout([state])[0]))
    return state, out


def test_normalizer_blocks_use_only_earlier_samples():
    norm = CausalNormalizer(median=90., mad=10., prior_weight=400., block=480)
    x = torch.from_numpy(squiggle(2000, 11))[None]
    z = norm(x)
    m0, s0 = 90., 10.
    first = (x[0, :480] - m0)/(1.4826*s0)
    inlier = first.abs() <= 3.5
    torch.testing.assert_close(z[0, :480][inlier], first[inlier])
    med = x[0, :480].median(); mad = (x[0, :480] - med).abs().median()
    m, s = (480*med + 400*m0)/880, (480*mad + 400*s0)/880
    second = (x[0, 480:960] - m)/(1.4826*s)
    torch.testing.assert_close(z[0, 480:960][second.abs() <= 3.5], second[second.abs() <= 3.5])
    changed = x.clone(); changed[0, 1000:] += 50
    assert torch.equal(norm(changed)[0, :1000], z[0, :1000])


def test_normalizer_holds_last_inlier():
    norm = CausalNormalizer(median=0., mad=1/1.4826, prior_weight=1e9, block=480)  # z == x in block 0
    z = norm(torch.tensor([[9., 1., 2., 9., -9., 3.]]))
    torch.testing.assert_close(z, torch.tensor([[0., 1., 2., 2., 2., 3.]]))


def test_normalizer_online_matches_offline():
    norm = CausalNormalizer(median=90., mad=10.)
    x = torch.from_numpy(squiggle(3000, 12))
    table, last, parts = torch.zeros(0, 2), 0., []
    for lo, hi in [(0, 1), (1, 479), (479, 481), (481, 2000), (2000, 3000)]:
        (z,), (table,) = norm.extend([x[:hi]], [lo], [table], [last]); last = float(z[-1]); parts.append(z)
    torch.testing.assert_close(torch.cat(parts), norm(x[None])[0], rtol=0, atol=1e-6)


def test_chunk_intake_and_calibration():
    signal = np.arange(10.)
    assert list(fresh_samples(4, 2, signal[2:8])) == [4, 5, 6, 7]
    assert len(fresh_samples(8, 2, signal[2:8])) == 0
    for received, offset, chunk in ((4, 5, signal[5:]), (0, -1, signal), (0, 0, signal[None]), (0, 0, [1., np.inf])):
        with pytest.raises(ValueError): fresh_samples(received, offset, chunk)
    np.testing.assert_allclose(to_pa([0, 100], offset=10., range_=1400., digitisation=8192.), [10*1400/8192, 110*1400/8192], rtol=1e-6)


# Streaming prefix encoder

def randomize_batchnorm(model):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                module.running_mean.uniform_(-.1, .1); module.running_var.uniform_(.5, 2.)
    return model


@pytest.fixture(scope='module')
def r0():
    torch.manual_seed(1)
    return randomize_batchnorm(build_stream_r0(median=90., mad=10., skip=SKIP, max_samples=BUDGET))


def r0_definition(model, signal, n):
    """R0 by definition: stage A layers (two-sided padding) over the normalized stream [skip, skip + L'), mean of
    all frames, projection, head."""
    used = (min(n, BUDGET, len(signal)) - SKIP)//48*48
    with torch.no_grad():
        z = model.normalizer(torch.from_numpy(signal[SKIP:SKIP + used])[None])
        return model.logits(model.backbone.layers(z[:, None]).mean(-1))[0]


def r0_offline(model, signal, ns):
    with torch.no_grad():
        logits, valid = model(torch.from_numpy(signal)[None], [len(signal)], torch.tensor([ns]))
    return logits[0], valid[0]


def r0_stream(model, signal, cuts, keep=60):
    """Feed signal[cuts[i]:cuts[i+1]]; return the state and (n, readout) after up to `keep` evenly spaced chunks
    (always including the last)."""
    state, out = model.init_state(), []
    pairs = list(zip(cuts[:-1], cuts[1:]))
    kept = set(np.linspace(0, len(pairs) - 1, min(keep, len(pairs))).round().astype(int).tolist())
    for i, (lo, hi) in enumerate(pairs):
        model.step([state], [(lo, signal[lo:hi])])
        if i in kept: out.append((hi, model.readout([state])[0]))
    return state, out


def check_r0(model, signal, readouts):
    logits, valid = r0_offline(model, signal, [n for n, _ in readouts])
    for (n, got), want, ok in zip(readouts, logits, valid):
        if ok: torch.testing.assert_close(got, want, **TOL)
        else: assert torch.isnan(got).all() and torch.isnan(want).all(), n


def test_r0_geometry(r0):
    assert r0.min_samples == 432 and r0.context == 528 and r0.backbone.alignment == 'delayed'
    for used in (1056, 1104, 2016, 6000):
        begin, first = r0._tail_window(used, r0._final(used))
        assert (used - begin, first) == (1056, 11)
    for used in (432, 528, 1008):
        assert r0._tail_window(used, r0._final(used)) == (0, r0._final(used))


def test_r0_offline_is_r0(r0):
    signal = squiggle(8000, 11)
    ns = [SKIP + 431, SKIP + 432, SKIP + 500, SKIP + 1055, SKIP + 1056, SKIP + 1100, 3000, 4321, BUDGET, 8000]
    logits, valid = r0_offline(r0, signal, ns)
    assert valid.tolist() == [False] + [True]*9 and torch.isnan(logits[0]).all()
    for n, row in zip(ns[1:], logits[1:]):
        torch.testing.assert_close(row, r0_definition(r0, signal, n), **TOL)
    torch.testing.assert_close(logits[-1], logits[-2], **TOL)  # budget


def test_r0_is_causal(r0):
    signal = squiggle(7500, 13)
    ns = [SKIP + 432, SKIP + 1000, 3000, 5123]
    base, _ = r0_offline(r0, signal, ns)
    for j, n in enumerate(ns):
        changed = signal.copy(); changed[n:] = squiggle(len(signal) - n, 14)
        assert torch.equal(r0_offline(r0, changed, ns)[0][:j + 1], base[:j + 1])


@pytest.mark.parametrize('schedule', ['chunks_400ms', 'random_cuts', 'block_and_frame_edges', 'single_samples', 'one_big'])
def test_r0_any_chunking_matches_offline(r0, schedule):
    rng = np.random.default_rng(17)
    signal = squiggle(8000, 12)  # longer than the budget
    if schedule == 'chunks_400ms':
        cuts = list(range(int(rng.integers(300, 1600)), len(signal), 1600))
    elif schedule == 'random_cuts':
        cuts = sorted(rng.choice(np.arange(1, len(signal)), 60, replace=False).tolist())
    elif schedule == 'block_and_frame_edges':
        cuts = sorted({SKIP, SKIP + 48, SKIP + 431, SKIP + 432, SKIP + 480, SKIP + 1056, SKIP + 1057, 4500, BUDGET, BUDGET + 1})
    elif schedule == 'single_samples':
        cuts = list(range(SKIP + 400, SKIP + 1200)) + [4000, 4001, 6100]
    else:
        cuts = [4000]
    state, readouts = r0_stream(r0, signal, [0, *cuts, len(signal)])
    check_r0(r0, signal, readouts)
    assert state.used == BUDGET - SKIP and state.frames == r0._final(BUDGET - SKIP) and state.tail_frames == 11


def test_r0_batched_reads_match_single_reads(r0):
    signals = [squiggle(n, 20 + n) for n in (7500, 2900, 5000, 1700)]
    schedules = [list(range(0, 7500, 1600)), list(range(0, 2900, 333)), list(range(0, 5000, 700)), [0, 1000]]
    states = [r0.init_state() for _ in signals]
    readouts = [[] for _ in signals]
    for round_ in range(max(map(len, schedules))):
        chunks = []
        for signal, cuts in zip(signals, schedules):
            lo = cuts[round_] if round_ < len(cuts) else None
            chunks.append(None if lo is None else (lo, signal[lo:cuts[round_ + 1] if round_ + 1 < len(cuts) else len(signal)]))
        r0.step(states, chunks)
        for i, row in enumerate(r0.readout(states)):
            readouts[i].append((states[i].received, row))
    for signal, rows in zip(signals, readouts):
        check_r0(r0, signal, rows)
    assert torch.isnan(readouts[3][-1][1]).all()  # 200 samples after skip: no decision


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_r0_device_resident_state_matches_offline(r0):
    import copy
    model = copy.deepcopy(r0).cuda()
    signals = [squiggle(n, 40 + n) for n in (7600, 3100, 5200)]
    states = [model.init_state(signal_device='cuda') for _ in signals]
    rows = [[] for _ in signals]
    for lo in range(0, 7600, 1600):
        model.step(states, [None if lo >= len(s) else (lo, s[lo:lo + 1600]) for s in signals])
        for i, row in enumerate(model.readout(states).cpu()):
            rows[i].append((states[i].received, row))
    assert states[0].z.is_cuda and states[0].stats.is_cuda
    for signal, readouts in zip(signals, rows):
        check_r0(r0, signal, readouts)


def test_r0_edges(r0):
    signal = squiggle(9000, 15)
    state, readouts = r0_stream(r0, signal, [0, 1000, SKIP + 47, SKIP + 400, SKIP + 433, 7000, 9000])
    assert [torch.isnan(row).all().item() for _, row in readouts] == [True, True, True, False, False, False]
    torch.testing.assert_close(readouts[-1][1], r0_definition(r0, signal, BUDGET), **TOL)
    assert state.received == 9000 and state.length == BUDGET - SKIP
    repeated = r0.init_state()
    for chunk in [(0, signal[:3000]), (0, signal[:1000]), (3000, signal[3000:3000]), (2000, signal[2000:4000]),
                  (3999, signal[3999:9000])]:
        r0.step([repeated], [chunk])
    torch.testing.assert_close(r0.readout([repeated]), r0.readout([state]), **TOL)  # same frames, other summation order
    gap = r0.init_state()
    r0.step([gap], [(0, signal[:3000])]); before = r0.readout([gap])
    r0.step([gap], [(3100, signal[3100:5000])])
    assert gap.failed and gap.received == 3000 and torch.equal(r0.readout([gap]), before)


def test_r0_time_term_starts_at_zero_and_streams(r0):
    torch.manual_seed(5)
    timed = randomize_batchnorm(build_stream_r0(median=90., mad=10., skip=SKIP, max_samples=BUDGET, time_features=True))
    timed.load_state_dict({**r0.state_dict(), **{f'time_head.{k}': v for k, v in timed.time_head.state_dict().items()}})
    signal = squiggle(8000, 16)
    ns = [SKIP + 432, 3000, BUDGET]
    torch.testing.assert_close(r0_offline(timed, signal, ns)[0], r0_offline(r0, signal, ns)[0], rtol=0, atol=0)
    with torch.no_grad():
        timed.time_head.weight.normal_(); timed.time_head.bias.normal_()
    _, readouts = r0_stream(timed, signal, [0, *range(1700, 8000, 1600), 8000])
    check_r0(timed, signal, readouts)
    assert not torch.allclose(r0_offline(timed, signal, ns)[0], r0_offline(r0, signal, ns)[0])


def test_r0_checkpoint_transfers_backbone_projection_and_head(tmp_path):
    torch.manual_seed(4)
    source, head = build_model(), torch.nn.Linear(128, 2)
    torch.save({'model_state': source.state_dict(), 'temporary_head': head.state_dict()}, tmp_path/'stage_A.pt')
    model = build_stream_r0(tmp_path/'stage_A.pt', median=90., mad=10.)
    for ours, theirs in ((model.backbone.layers, source.encoder), (model.projection, source.projection), (model.head, head)):
        a, b = ours.state_dict(), theirs.state_dict()
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    assert not model.training and not any(m.training for m in model.modules())
