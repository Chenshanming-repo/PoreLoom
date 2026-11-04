"""Fixed symmetric-threshold controls for manuscript Table 3 and Table S2.

python -m poreloom.method.thresholds --data DATA --run RUN --out OUTPUT --fs 4000

Uses the frozen encoder in RUN/stream/best.pt and all test reads, with the
same observation phases, processing delay and ejection delay as FACP.
No threshold is selected on training, validation or test outcomes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from . import data as D, head as H
from ..chunk_replay import replay, threshold_actions
from ..stream_config import sampling_rate
from ..stream_training import load_stream_r0

DELTAS = (0.01, 0.05, 0.1, 0.2, 0.3)


def evaluate(z, fs, delta, service=H.HEAD['service'], eject=H.HEAD['eject']):
    """Reject p < delta, accept p >= 1-delta, and force p=0.5 at budget."""
    if not np.isfinite(delta) or not 0 < delta < 0.5:
        raise ValueError('delta must be finite and between 0 and 0.5')
    looks = H.looks(z, fs)
    times, valid, forced = looks.t, looks.valid, looks.forced & looks.valid
    probabilities = np.where(valid, expit(z['ell']), np.nan)
    actions = threshold_actions(probabilities, forced, delta, 0.5, 1-delta)
    out = replay(times + service, actions, times, service,
                 np.asarray(z['raw_length'])/fs, eject)
    rows = np.flatnonzero(out['command'])
    endpoint = np.zeros(len(out['command']), bool)
    endpoint[rows] = forced[rows, out['steps'][rows]]
    return out, endpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fs', type=float, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    fs = sampling_rate(args.fs)
    if (args.run/'run.json').exists():
        config = json.loads((args.run/'run.json').read_text())
        if config.get('interval', H.HEAD['interval']) != H.HEAD['interval']:
            parser.error('manuscript controls require a run trained with 0.4-s observations')
        if config['fs'] != fs:
            parser.error('--fs differs from the frozen encoder run replay rate')
    encoder, _ = load_stream_r0(args.run/'stream'/'best.pt', args.device)
    data = D.load_split(args.data, 'test', verify=True)
    z = H.extract(encoder, data, fs, H.HEAD['interval'], H.HEAD['phase_seed'], args.device)
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for delta in DELTAS:
        out, endpoint = evaluate(z, fs, delta)
        rows.append(dict(delta=delta, **H.metrics(out, z['is_target']),
                         forced_decision_fraction_all_reads=float(endpoint.mean())))
        np.savez(args.out/f'outcomes_delta_{delta:g}.npz', read_id=z['read_id'],
                 is_target=z['is_target'], forced_decision=endpoint, **out)
    (args.out/'metrics.json').write_text(json.dumps(dict(
        protocol='manuscript_20261104', fs=fs, interval=H.HEAD['interval'],
        budget=encoder.max_samples-encoder.skip, final_probability_threshold=0.5,
        forced_decision_denominator='all test reads', rows=rows), indent=2)+'\n')


if __name__ == '__main__':
    main()
