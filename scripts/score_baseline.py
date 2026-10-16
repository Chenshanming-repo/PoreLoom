#!/usr/bin/env python3
"""Score a native export with a fitted baseline; writes margin/labels/rows to an .npz.

The host margin is host logit minus microbe logit (> 0: the tool's argmax host call).
Parents without a valid window get NaN.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from poreloom.baselines import score_source
from poreloom.baselines.native_source import NativeSource
from poreloom.paths import results_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', choices=('nanodeep', 'cafes', 'readcurrent', 'squigglenet'), required=True)
    parser.add_argument('--checkpoint', type=Path, help='default: the archived native fit best.pt')
    parser.add_argument('--source', type=Path, default=results_path('defer_simulation/native_development/source'))
    parser.add_argument('--role', help='only rows with this split/role (e.g. dev, C_screen)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint or results_path(
        'defer_simulation/native_baselines', args.tool, 'models/seed20260929/best.pt')
    rows = NativeSource(args.source).rows(args.role) if args.role else None
    result = score_source(args.tool, checkpoint, args.source, rows=rows, device=args.device,
                          batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **result)
    margin = result['margin']
    print(json.dumps(dict(tool=args.tool, checkpoint=str(checkpoint), source=str(args.source),
                          scored=int(np.isfinite(margin).sum()), rows=len(margin), output=str(args.output))))


if __name__ == '__main__':
    main()
