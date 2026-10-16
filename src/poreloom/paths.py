"""Repository, vendored-source and external data/result locations.

Data and results live outside the repository. Lookup order for each root:
environment variable, then ``.env`` at the repository root, then the default.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = REPO_ROOT / 'third_party'
DEFAULTS = {'PORELOOM_DATA': str(REPO_ROOT / 'data'),
            'PORELOOM_RESULTS': str(REPO_ROOT / 'results')}


def _dotenv(path=REPO_ROOT / '.env'):
    values = {}
    try:
        lines = Path(path).read_text().splitlines()
    except FileNotFoundError:
        return values
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip().strip('"\'')
    return values


def _root(name):
    value = os.environ.get(name) or _dotenv().get(name) or DEFAULTS[name]
    return Path(value).expanduser()


def data_root():
    return _root('PORELOOM_DATA')


def results_root():
    return _root('PORELOOM_RESULTS')


def results_path(*parts):
    return results_root().joinpath(*parts)
