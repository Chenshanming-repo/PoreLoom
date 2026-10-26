"""Prefix caches, sampled-length training, and classification metrics on synthetic data."""
import json

import numpy as np
import pytest
import torch

from poreloom.data import file_sha256
from poreloom.model import build_model
from poreloom.prefix_data import FILES, PrefixSet, load_prefix_cache
from poreloom.stream_evaluation import auroc, ece, frr_at_host_reject, summarize
from poreloom.stream_encoder import build_stream_r0
from poreloom.stream_training import (R0_DEFAULTS, batch_signal, load_stream_r0, normalizer_prior, r0_loss,
                                      sample_lengths, score_r0, train_stream_r0)
from poreloom.streaming import to_pa


def synthetic_prefix(n=12, length=9000, seed=0):
    rng = np.random.default_rng(seed)
    is_target = np.arange(n) % 2 == 1
    levels = np.where(is_target[:, None], 520, 480) + rng.integers(-40, 40, (n, length//10)).repeat(10, 1)
    adc = (levels + rng.normal(0, 8, (n, length))).astype(np.int16)
    raw_length = rng.integers(2500, 12000, n); raw_length[0] = 1800
    adc[np.arange(length) >= raw_length[:, None]] = 0
    calibration = np.stack([rng.uniform(-60, 30, n), np.full(n, 1536.598389), np.full(n, 8192.)], 1)
    return PrefixSet(adc, is_target, raw_length, calibration, np.array([f'r{i}' for i in range(n)]), np.full(n, 'x'))


def test_prefix_set_converts_to_pa_and_zeroes_past_length():
    data = synthetic_prefix()
    x = data.pa([0, 3], 1000, 2600)
    offset, range_, digitisation = data.calibration[3]
    np.testing.assert_allclose(x[1], to_pa(data.adc[3, 1000:2600], offset, range_, digitisation), rtol=1e-6)
    assert (x[0, 1800 - 1000:] == 0).all() and (x[0, :800] != 0).any()


def test_prefix_cache_round_trip_and_hash_check(tmp_path):
    data = synthetic_prefix()
    np.save(tmp_path/'adc.npy', data.adc)
    np.savez_compressed(tmp_path/'metadata.npz', read_id=data.read_id, is_target=data.is_target,
                        raw_length=data.raw_length, offset=data.calibration[:, 0], range=data.calibration[:, 1],
                        digitisation=data.calibration[:, 2])
    (tmp_path/'manifest.json').write_text(json.dumps(dict(files={f: file_sha256(tmp_path/f) for f in FILES})))
    loaded = load_prefix_cache(tmp_path)
    assert np.array_equal(loaded.adc, data.adc) and np.array_equal(loaded.is_target, data.is_target)
    np.save(tmp_path/'adc.npy', data.adc + 1)
    with pytest.raises(ValueError): load_prefix_cache(tmp_path)


def test_batches_cover_budget_and_augment_only_valid_samples():
    data, config = synthetic_prefix(), dict(R0_DEFAULTS)
    pa, lengths = batch_signal(data, np.arange(4), config)
    assert pa.shape == (4, 7500) and list(lengths) == list(np.minimum(data.raw_length[:4], 7500))
    np.testing.assert_allclose(pa.numpy(), data.pa(np.arange(4), 0, 7500), rtol=1e-6)
    rng = np.random.default_rng(1)
    pa, lengths = batch_signal(data, np.arange(12), {**config, 'jitter_p': 1.}, rng)
    assert (lengths <= 7500).all() and (lengths <= torch.as_tensor(data.raw_length)).all()
    for row, n in zip(pa, lengths): assert (row[int(n):] == 0).all()


def test_normalizer_prior_is_median_of_read_statistics():
    data = synthetic_prefix()
    median, mad = normalizer_prior(data, np.arange(1, 12), 1500, 7500)
    x = data.pa([1], 1500, 7500)[0][:data.raw_length[1] - 1500]
    assert 50 < median < 150 and 0 < mad < 30 and abs(np.median(x) - median) < 30


def test_metrics():
    y = np.array([0, 0, 1, 1], bool)
    assert auroc(np.array([.1, .4, .35, .8]), y) == .75
    p = np.r_[np.linspace(0, .5, 100), np.linspace(.3, 1, 100)]; t = np.r_[np.zeros(100, bool), np.ones(100, bool)]
    assert frr_at_host_reject(p, t, .95) == pytest.approx(np.mean(p[100:] < np.quantile(p[:100], .95)))
    assert ece(np.array([.2, .2, .8, .8]), np.array([0, 0, 1, 1], bool)) == pytest.approx(.2)
    out = summarize(np.r_[p, np.nan], np.r_[t, True])
    assert out['reads'] == 200 and {'frr_host90', 'frr_host95', 'frr_host99'} <= out.keys()


def test_r0_lengths_are_aligned_and_usable():
    model = build_stream_r0(median=90., mad=10.)
    lengths = np.array([7500, 1500 + 432, 1500 + 431, 4000, 9000])
    n = sample_lengths(lengths, model, 200, np.random.default_rng(0))
    assert (n[2] == -1).all() and (n[1] == 1500 + 432).all()
    used = n[[0, 3, 4]] - 1500
    assert (used % 48 == 0).all() and used.min() == 432 and (used[0] <= 6000).all() and (used[1] <= 2496).all()
    assert used[0].max() == 6000 and used[1].max() == 2496  # the top length is reachable


def test_r0_loss_ignores_invalid_lengths():
    logits = torch.tensor([[[2., 0.], [float('nan')]*2], [[0., 1.], [1., 0.]], [[float('nan')]*2]*2])
    valid = torch.tensor([[True, False], [True, True], [False, False]])
    labels = torch.tensor([0, 1])
    got = r0_loss(logits, torch.tensor([0, 1, 0]), valid)
    ce = torch.nn.functional.cross_entropy
    want = (ce(logits[0, :1], labels[:1]) + (ce(logits[1, :1], labels[1:]) + ce(logits[1, 1:], labels[1:]))/2)/2
    torch.testing.assert_close(got, want)


def test_r0_smoke_training_starts_from_r0_and_saves_a_loadable_checkpoint(tmp_path):
    torch.manual_seed(0)
    torch.save({'model_state': build_model().state_dict(), 'temporary_head': torch.nn.Linear(128, 2).state_dict()},
               tmp_path/'stage_A.pt')
    data = synthetic_prefix()
    best = train_stream_r0(data, data, tmp_path/'stage_A.pt', tmp_path/'run', device='cpu', log=lambda *a: None,
                           epochs=1, batch_size=4, eval_batch=6, lengths_per_read=2, dev_lengths=(432, 1488))
    history = json.loads((tmp_path/'run'/'training_history.json').read_text())
    assert [h['epoch'] for h in history] == [0, 1] and all(np.isfinite(h['dev_ce']) for h in history)
    model, payload = load_stream_r0(best)
    assert payload['epoch'] in (0, 1) and model.time_head is not None
    untrained = build_stream_r0(tmp_path/'stage_A.pt', median=payload['normalizer_prior']['median_pa'],
                                mad=payload['normalizer_prior']['mad_pa'])
    config = {**R0_DEFAULTS, 'eval_batch': 6}
    p0, _ = score_r0(untrained, data, np.arange(len(data)), config, 'cpu', (432, 1488))
    if payload['epoch'] == 0:  # epoch 0 is R0 itself (zero time term)
        p1, _ = score_r0(model, data, np.arange(len(data)), config, 'cpu', (432, 1488))
        np.testing.assert_allclose(p1, p0, rtol=0, atol=1e-6)
    assert np.isnan(p0[0, 0]) and np.isfinite(p0[1:, 0]).all()  # read 0 has 1800 samples
