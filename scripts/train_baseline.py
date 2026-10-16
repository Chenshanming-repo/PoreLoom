#!/usr/bin/env python3
"""Fit one native baseline (NanoDeep, CAFES, ReadCurrent or SquiggleNet) on TRAIN / model_DEV.

NanoDeep and SquiggleNet read the native export directly. ReadCurrent and CAFES
read tile datasets from scripts/prepare_baseline_tiles.py (--tiles).
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from poreloom.baselines import training
from poreloom.paths import results_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', choices=('nanodeep', 'cafes', 'readcurrent', 'squigglenet'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-source', type=Path, default=results_path('defer_simulation/native_source'))
    parser.add_argument('--tiles', type=Path, default=results_path('defer_simulation/native_baselines'),
                        help='directory holding <tool>/data/{pos,neg} (readcurrent, cafes)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=training.SEED)
    parser.add_argument('--smoke', action='store_true',
                        help='tiny run: 2 epochs on 64 rows per split, or 20 trainer steps for tiles')
    args = parser.parse_args(argv)
    if args.tool == 'nanodeep':
        smoke = dict(min_epochs=1, max_epochs=2, limit=64) if args.smoke else {}
        result = training.train_nanodeep(args.native_source, args.output, device=args.device, seed=args.seed, **smoke)
    elif args.tool == 'squigglenet':
        smoke = dict(epochs=2, batch_size=32, limit=64) if args.smoke else {}
        result = training.train_squigglenet(args.native_source, args.output, device=args.device, seed=args.seed, **smoke)
    else:
        result = training.train_tiles(args.tool, args.tiles, args.output, device=args.device, seed=args.seed,
                                      smoke_steps=20 if args.smoke else None)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
