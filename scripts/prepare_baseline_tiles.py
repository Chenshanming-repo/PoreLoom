#!/usr/bin/env python3
"""Build ReadCurrent/CAFES tile datasets from a native training export."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from poreloom.baselines.tiles import prepare_tiles
from poreloom.paths import results_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-source', type=Path, default=results_path('defer_simulation/native_source'))
    parser.add_argument('--output', type=Path, required=True, help='writes <output>/{cafes,readcurrent}')
    args = parser.parse_args(argv)
    print(json.dumps(prepare_tiles(args.native_source, args.output), indent=2))


if __name__ == '__main__':
    main()
