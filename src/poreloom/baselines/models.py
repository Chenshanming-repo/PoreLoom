"""Upstream baseline architectures, imported by path from ``third_party``."""
import importlib.util
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

from ..paths import THIRD_PARTY

TOOLS = ('nanodeep', 'cafes', 'readcurrent', 'squigglenet')
NANODEEP_SOURCE = THIRD_PARTY / 'NanoDeep/read_deep/model/nanodeep.py'
READCURRENT_SOURCE = THIRD_PARTY / 'ReadCurrent/models/ReadCurrent.py'
READCURRENT_TRAINER = THIRD_PARTY / 'ReadCurrent_patched'
CAFES_ROOT = THIRD_PARTY / 'CAFES'
SQUIGGLENET_ROOT = THIRD_PARTY / 'SquiggleNet'
CAFES_MODEL = dict(n_conv_neurons=[32, 64, 128, 256, 512], n_fc_neurons=2048, depth=29, n_classes=2,
                   shortcut=True, dropout_rate=.2, use_SEBlock=True, n_input_channels=6, use_first_bn=True)
PARAMETERS = {'nanodeep': 55686, 'cafes': 16917706, 'readcurrent': 16817282, 'squigglenet': 66916}

_loaded = {}


def load_module(path, cache=True):
    """Import a vendored file under a private module name (once, unless ``cache=False``)."""
    path = Path(path).resolve()
    if cache and path in _loaded:
        return _loaded[path]
    name = '_poreloom_vendored_' + '_'.join(path.relative_to(THIRD_PARTY.resolve()).with_suffix('').parts)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if cache:
        _loaded[path] = module
    return module


def build_nanodeep(stochastic_eval=False):
    """Upstream NanoDeep with dropout disabled in eval unless ``stochastic_eval``.

    Upstream applies functional dropout unconditionally; the state-dict layout and
    the training computation are unchanged.
    """
    upstream = load_module(NANODEEP_SOURCE)

    class CausalNanoDeep(upstream.model):
        def __init__(self):
            super().__init__(kernel_length=5, class_num=2, kernal_num=32)
            self.stochastic_eval = bool(stochastic_eval)

        def forward(self, x, **kwargs):
            x = self.conv1d(x)
            x = self.se(x)
            x = self.conv1d2(x)
            x = self.res1d1(x)
            x = self.pool1(x)
            x = self.res1d2(x)
            x = F.dropout(x, p=.2, training=self.training or self.stochastic_eval)
            x = self.pool2(x)
            x = self.res1d3(x)
            x = F.dropout(x, p=.2, training=self.training or self.stochastic_eval)
            x = self.avgpool(x)
            return self.liner(x)

    return CausalNanoDeep()


def build_model(tool):
    if tool == 'nanodeep':
        model = build_nanodeep()
    elif tool == 'readcurrent':
        model = load_module(READCURRENT_SOURCE).ReadCurrent([32, 64, 128, 256, 512], n_fc_neurons=2048,
                                                             depth=29, shortcut=True)
    elif tool == 'cafes':
        model = load_module(CAFES_ROOT / 'models/CAFES.py').CAFES(**CAFES_MODEL)
    elif tool == 'squigglenet':
        module = load_module(SQUIGGLENET_ROOT / 'model.py')
        model = module.ResNet(module.Bottleneck, [2, 2, 2, 2])
    else:
        raise ValueError(f'unknown baseline tool: {tool}')
    count = sum(p.numel() for p in model.parameters())
    if count != PARAMETERS[tool]:
        raise ValueError(f'{tool} has {count} parameters; expected {PARAMETERS[tool]}')
    return model


def torch_load(path):
    """Load a trusted local checkpoint (ours carry optimizer and RNG state)."""
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only
        return torch.load(path, map_location='cpu')


def checkpoint_state(value):
    """Model weights from a bare state dict or a trainer checkpoint."""
    state = value.get('model_state', value)
    return {key.removeprefix('module.'): tensor for key, tensor in state.items()}


def load_model(tool, checkpoint, device='cpu'):
    model = build_model(tool)
    model.load_state_dict(checkpoint_state(torch_load(checkpoint)), strict=True)
    return model.to(device).eval()


def cafes_preprocessor():
    return load_module(CAFES_ROOT / 'preprocessor.py')


def readcurrent_trainer():
    """A fresh patched ReadCurrent trainer module (callers may override its hooks).

    Its bare ``from upstream_training import ...`` is bound to the vendored controller.
    """
    controller = load_module(READCURRENT_TRAINER / 'upstream_training.py')
    previous = sys.modules.get('upstream_training')
    path = list(sys.path)
    try:
        sys.modules['upstream_training'] = controller
        return load_module(READCURRENT_TRAINER / 'trainer.py', cache=False)
    finally:
        if previous is None:
            sys.modules.pop('upstream_training', None)
        else:
            sys.modules['upstream_training'] = previous
        sys.path[:] = path

