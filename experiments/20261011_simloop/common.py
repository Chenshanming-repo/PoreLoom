"""Shared paths and the frozen strategy registry for the simulator experiment.

Everything reuses the signal_explore implementation of the earlier
deferred-decision host-depletion study (same generator, genomic split, ADC
windows, timing model, B0 scoring and paired bootstrap), so numbers stay
comparable with that study.
"""
from pathlib import Path
import hashlib
import json
import sys

SE = Path('/path/to/poreloom_work')
OUT = SE / 'results/poreloom_simloop_20261011'
PYTHON = SE / '.venv-nanodeep/bin/python'
PARENT = SE / 'results/poreloom_resource_ablation_20261010/validation_retry1'
PREFIX = SE / 'results/poreloom_control_20261009/shared/prefix_best.pt'
TIMING = SE / 'results/poreloom_20261009/baseline_timing.json'
REGISTRATION = SE / 'results/native_window_retest_20261009/registration.json'

# Offline technical cohort: three new molecule/waveform shards, class-balanced.
SEEDS = (20331011, 20331012, 20331013)
COUNT_PER_RUN = 20000

# PoreLoom candidates: family name in the parent frozen_selection -> report id.
POLICIES = {
    'B_diagnostic': 'PoreLoom-B',
    'A': 'PoreLoom-A',
    'prior_gru': 'GRU-threshold',
    'repr_gru': 'SameRepr-GRU',
    'repr_fixed4': 'F-class',
}
F_ACTOR = 'F-actor'          # fixed W4 with A's terminal actor; built here, not in parent
PRIMARY_POLICY = 'PoreLoom-B'
SECONDARY_POLICY = 'PoreLoom-A'

# Native baselines, archived simulation fits (native_window_retest sim_baseline job).
BASELINES = {
    'readcurrent': dict(id='ReadCurrent', role='primary', window=(1500, 4500), units='ADC'),
    'nanodeep': dict(id='NanoDeep', role='key_secondary', window=(1000, 4000), units='ADC'),
    'cafes': dict(id='CAFES', role='extended', window=(1500, 4500), units='ADC'),
    'squigglenet': dict(id='SquiggleNet', role='extended', window=(1500, 4500), units='pA'),
}
PRIMARY_BASELINE = 'ReadCurrent'


def setup_imports():
    for p in (SE / 'src', SE / 'scripts'):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    for site in sorted((SE / '.venv-raw/lib').glob('python*/site-packages')):
        if str(site) not in sys.path:
            sys.path.append(str(site))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def pin(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')


def read(path):
    return json.loads(Path(path).read_text())
