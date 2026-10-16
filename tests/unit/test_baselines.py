import json

import numpy as np
import pytest
import torch

from poreloom.baselines import CONTRACTS, build_model, host_margin, prepare_windows, score_source
from poreloom.baselines import training
from poreloom.baselines.models import PARAMETERS
from poreloom.baselines.native_source import NativeSource
from poreloom.baselines.tiles import prepare_tiles, tile_count, train_spans
from poreloom.paths import THIRD_PARTY

# Upstream baseline sources are local-only (git-ignored); skip on a fresh clone.
pytestmark = pytest.mark.skipif(
    not all((THIRD_PARTY / d).is_dir() for d in ('CAFES', 'NanoDeep', 'SquiggleNet', 'ReadCurrent_patched')),
    reason='third_party baseline sources not present (see third_party/README.md)')


@pytest.fixture(autouse=True)
def restore_determinism():
    before = torch.are_deterministic_algorithms_enabled()
    yield
    torch.use_deterministic_algorithms(before)


def make_native(root):
    """Tiny native export: 8 TRAIN + 8 DEV parents, balanced, with full-read caches."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    split = np.array(['train'] * 8 + ['dev'] * 8)
    labels = np.array([0, 1] * 8)
    lengths = rng.integers(7000, 13000, 16)
    lengths[8] = 4200  # long enough for raw[1000:4000] only
    reads = [rng.normal(500 + 40 * label, 30, length).astype(np.int16) for label, length in zip(labels, lengths)]
    reads[0][1500:4500] = 480  # zero-MAD first tile
    meta = dict(split=split, labels=labels, raw_length=lengths, cache_length=lengths,
                cache_offset=np.full(16, -1), read_id=np.array([f'r{i}' for i in range(16)]))
    for key, minimum in (('adc_1000_4000', 4000), ('adc_1500_4500', 4500), ('pa_1500_4500', 4500)):
        meta[f'eligible_{key}'] = lengths >= minimum
        start = int(key.split('_')[1])
        dtype = np.float32 if key.startswith('pa') else np.int16
        windows = np.zeros((16, 3000), dtype)
        for i, read in enumerate(reads):
            if lengths[i] >= minimum:
                windows[i] = read[start:start + 3000] * (np.float32(.15) if dtype == np.float32 else 1)
        np.save(root / f'{key}.npy', windows)
    for label, name in ((0, 'microbe'), (1, 'host')):
        offset, parts = 0, []
        for i in np.flatnonzero((split == 'train') & (labels == label)):
            meta['cache_offset'][i] = offset
            offset += lengths[i]
            parts.append(reads[i])
        np.save(root / f'train_{name}_adc.npy', np.concatenate(parts))
    np.savez(root / 'metadata.npz', **meta)
    return root


@pytest.mark.parametrize('tool', ['nanodeep', 'cafes', 'readcurrent', 'squigglenet'])
def test_models_build_with_upstream_parameter_counts(tool):
    model = build_model(tool).eval()
    assert sum(p.numel() for p in model.parameters()) == PARAMETERS[tool]
    windows = np.random.default_rng(0).normal(500, 30, (2, 3000))
    windows = windows.astype(np.float32 if tool == 'squigglenet' else np.int16)
    x, valid = prepare_windows(windows, tool)
    assert valid.all()
    with torch.no_grad():
        assert model(torch.from_numpy(x)).shape == (2, 2)


@pytest.mark.parametrize('tool', ['nanodeep', 'readcurrent', 'squigglenet'])
def test_constant_window_is_invalid(tool):
    windows = np.full((1, 3000), 500, np.float32 if tool == 'squigglenet' else np.int16)
    _, valid = prepare_windows(windows, tool)
    assert not valid.any()


def test_host_margin_follows_each_tools_label_order():
    logits = np.array([[0., 1.]])
    assert host_margin(logits, 'nanodeep')[0] == 1 and host_margin(logits, 'squigglenet')[0] == 1
    assert host_margin(logits, 'readcurrent')[0] == -1 and host_margin(logits, 'cafes')[0] == -1
    assert {c['key'] for c in CONTRACTS.values()} == {'adc_1000_4000', 'adc_1500_4500', 'pa_1500_4500'}


def test_train_spans_are_offset_major_full_read_tiles():
    assert list(train_spans(9000)) == [(1500, 4500), (4500, 7500), (2500, 5500), (5500, 8500), (3500, 6500)]


def test_prepare_tiles_layout_and_counts(tmp_path):
    source = NativeSource(make_native(tmp_path / 'native'))
    summary = prepare_tiles(source.root, tmp_path / 'tiles')
    for label, group in ((0, 'pos'), (1, 'neg')):
        for split, stem in (('train', 'train'), ('dev', 'valid')):
            total = tile_count(source, label, split)
            cafes = np.load(tmp_path / f'tiles/cafes/data/{group}/{stem}_preprocessed.npy')
            rc = np.load(tmp_path / f'tiles/readcurrent/data/{group}/{stem}_preprocessed.npy')
            side = np.load(tmp_path / f'tiles/readcurrent/data/{group}/{stem}_metadata.npz')
            assert cafes.dtype == np.int16 and len(cafes) == total == summary[f'{group}/{stem}']['candidate_tiles']
            assert rc.dtype == np.float32 and len(rc) == len(side['row_index']) == summary[f'{group}/{stem}']['readcurrent_valid']
            assert np.all(source.labels[side['row_index']] == label)
    # Zero-MAD microbe TRAIN tiles (raw 1500 and 2500 starts, >= 2/3 constant) are
    # kept for CAFES and dropped for ReadCurrent.
    assert summary['pos/train']['invalid_MAD'] == 2
    # The short DEV microbe read has no raw[1500:4500] window.
    assert summary['pos/valid']['candidate_tiles'] == 3
    protocol = json.loads((tmp_path / 'tiles/readcurrent/protocol.json').read_text())
    assert protocol['label_map']['microbe']['folder'] == 'pos'


def test_nanodeep_and_squigglenet_smoke_fit_and_score(tmp_path):
    native = make_native(tmp_path / 'native')
    nd = training.train_nanodeep(native, tmp_path / 'nd', device='cpu', min_epochs=1, max_epochs=2)
    assert nd['epochs_completed'] == 2 and (tmp_path / 'nd/best.pt').exists()
    sq = training.train_squigglenet(native, tmp_path / 'sq', device='cpu', epochs=1, batch_size=4)
    assert sq['updates_completed'] == 2 and (tmp_path / 'sq/best.pt').exists()
    for tool, fit in (('nanodeep', 'nd'), ('squigglenet', 'sq')):
        rows = NativeSource(native).rows('dev')
        out = score_source(tool, tmp_path / fit / 'best.pt', native, rows=rows)
        eligible = NativeSource(native).eligible(CONTRACTS[tool]['key'])[rows]
        assert np.array_equal(np.isfinite(out['margin']), eligible)


# The patched ReadCurrent trainer calls torch.load(weights_only=...), added in torch 1.13.
@pytest.mark.skipif(tuple(int(v) for v in torch.__version__.split('.')[:2]) < (1, 13),
                    reason='patched ReadCurrent trainer needs torch >= 1.13')
@pytest.mark.parametrize('tool', ['readcurrent', 'cafes'])
def test_tile_trainer_smoke_fit_and_score(tmp_path, tool):
    native = make_native(tmp_path / 'native')
    prepare_tiles(native, tmp_path / 'tiles')
    result = training.train_tiles(tool, tmp_path / 'tiles', tmp_path / 'smoke', device='cpu', batch_size=4,
                                  smoke_steps=2)
    assert result['step'] == 2 and (tmp_path / 'smoke/best.pt').exists()
    out = score_source(tool, tmp_path / 'smoke/best.pt', native, rows=NativeSource(native).rows('dev'))
    assert np.isfinite(out['margin']).sum() == 7
