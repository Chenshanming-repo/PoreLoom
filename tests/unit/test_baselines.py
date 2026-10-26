import numpy as np
import pytest
import torch

from poreloom.baselines import CONTRACTS, build_model, host_margin, prepare_windows
from poreloom.baselines.models import PARAMETERS
from poreloom.paths import THIRD_PARTY

# Upstream baseline sources are local-only (git-ignored); skip on a fresh clone.
pytestmark = pytest.mark.skipif(
    not all((THIRD_PARTY / d).is_dir() for d in ('ReadCurrent', 'NanoDeep', 'SquiggleNet')),
    reason='third_party baseline sources not present (see third_party/README.md)')


@pytest.fixture(autouse=True)
def restore_determinism():
    before = torch.are_deterministic_algorithms_enabled()
    yield
    torch.use_deterministic_algorithms(before)


@pytest.mark.parametrize('tool', ['nanodeep', 'readcurrent', 'squigglenet'])
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
    assert host_margin(logits, 'readcurrent')[0] == -1
    assert {c['key'] for c in CONTRACTS.values()} == {'adc_1000_4000', 'adc_1500_4500', 'pa_1500_4500'}
