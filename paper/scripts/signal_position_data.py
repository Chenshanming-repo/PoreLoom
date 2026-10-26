"""Derive fixed-prefix curves and observed decision positions from completed results.

No inference or replay: the final arrival is recovered independently from the
saved completion time and from the seeded callback phase plus terminal step.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

from result_archive import ROOT, ARCHIVE
SERVICE = .0158467092551291
INTERVAL = .4
PHASE_SEED = 20261012
SKIP, ENDPOINT = 1500, 7500


def family(ds):
    return 'Simulated' if ds.startswith('sim') else 'R9' if ds == 'r9' else 'R10 mixed' if ds.startswith('r10') else 'R10 sputum'


def prepare():
    sources = set()
    fixed_path = ARCHIVE/'results/final_20261014/fixed_window/report.json'
    sources.add(str(fixed_path.relative_to(ARCHIVE)))
    report = json.loads(fixed_path.read_text())
    prefix, cdfs, summaries, checks = [], [], [], []
    for unit, r in report.items():
        ds, rep = unit.split('/')
        fam = family(ds)
        assert len(r['grid']) == len(r['acc']) == 32 and r['limit'] in (None, 0)
        for length, acc in zip(r['grid'], r['acc']):
            prefix.append(dict(unit=unit, family=fam, usable_samples=length, acc=acc, n_test=r['n_test']))
        base = ARCHIVE/'results/final_20261014/runs/main'/ds/f'poreloom_{rep}'
        run_path, outcome_path = base/'run.json', base/'test/outcomes_0.4_nominal.npz'
        sources.update(str(p.relative_to(ARCHIVE)) for p in [run_path, outcome_path])
        fs = float(json.loads(run_path.read_text())['fs'])
        assert fs == r['fs']
        with np.load(outcome_path, allow_pickle=False) as z:
            decided, target, rejected = (z[k].astype(bool) for k in ['command', 'is_target', 'command_reject'])
            t, steps = z['decision_seconds'], z['steps']
            assert len(t) == r['n_test']
            assert np.array_equal(decided, np.isfinite(t)) and np.array_equal(decided, steps >= 0)
            phase = np.random.default_rng(PHASE_SEED).random(len(t))*INTERVAL
            arrival = phase[decided] + steps[decided]*INTERVAL
            error = float(np.max(np.abs(t[decided] - (arrival+SERVICE))))
            assert error < 1e-12
            sample = np.floor(arrival*fs).astype(np.int64)
            inverse = np.floor((t[decided]-SERVICE)*fs).astype(np.int64)
            assert np.array_equal(sample, inverse)
            observed = np.minimum(sample, ENDPOINT)-SKIP
            assert observed.min() >= 432 and observed.max() <= 6000
            usable = np.full(len(t), -1, dtype=np.int64)
            usable[decided] = observed
            for category, mask in [('all', np.ones(len(t), bool)), ('host', ~target), ('target', target)]:
                selected = mask & decided
                values = np.sort(usable[selected])
                n, n_decided = int(mask.sum()), len(values)
                assert n_decided > 0
                summary = dict(unit=unit, family=fam, category=category, fs=fs,
                               n_reads=n, n_decided=n_decided, n_no_decision=n-n_decided,
                               n_accept=int((selected & ~rejected).sum()), n_reject=int((selected & rejected).sum()),
                               no_decision_fraction=(n-n_decided)/n,
                               median_samples=float(np.median(values)),
                               p10_samples=float(np.quantile(values, .1)), p90_samples=float(np.quantile(values, .9)),
                               fraction_by_1200=float(np.mean(values <= 1200)),
                               fraction_by_3000=float(np.mean(values <= 3000)),
                               fraction_at_6000=float(np.mean(values == 6000)))
                summaries.append(summary)
                grid = np.unique(np.r_[0, values, ENDPOINT-SKIP])
                for length, count in zip(grid, np.searchsorted(values, grid, side='right')):
                    cdfs.append(dict(unit=unit, family=fam, category=category, usable_samples=int(length),
                                     n_by_length=int(count), n_decided=n_decided, n_reads=n,
                                     cdf=count/n_decided, fraction_all_reads=count/n))
            checks.append(dict(unit=unit, n_reads=len(t), n_decided=int(decided.sum()),
                               timing_reconstruction_max_error_seconds=error,
                               sample_positions_match_two_reconstructions=True))
    for name, rows in [('fixed_prefix', prefix), ('decision_position_cdf', cdfs), ('decision_position_summary', summaries)]:
        pd.DataFrame(rows).to_csv(ROOT/'figure_data'/f'{name}.csv', index=False, float_format='%.15g')
    provenance = dict(source_files=[dict(path=p, sha256=hashlib.sha256((ARCHIVE/p).read_bytes()).hexdigest()) for p in sorted(sources)],
                      fixed_prefix_rows=len(prefix), units=len(checks),
                      cdf_rows=len(cdfs),
                      cdf_support='Every observed integer sample position per unit/category, plus 0 and 6000 anchors; no fixed plotting grid.',
                      outcome_rows=sum(r['n_reads'] for r in checks), decided_rows=sum(r['n_decided'] for r in checks),
                      interval_seconds=INTERVAL, service_seconds=SERVICE, phase_seed=PHASE_SEED,
                      reconstruction='min(floor((decision_seconds - service_seconds) * sampling_rate), 7500) - 1500',
                      cross_check='Reconstruct arrival independently as seeded phase + saved terminal step * 0.4 s; require equal integer sample positions.',
                      definition='Observed usable input length, before encoder stride alignment; capped at 6000. Both Accept and Reject decisions included. Natural endings without a decision are excluded from the conditional CDF and counted separately.',
                      aggregation='Equal mean of three unit CDFs per family, aligned by carrying each CDF forward on the union of observed positions; units are not independent biological acquisitions.', checks=checks)
    (ROOT/'review/signal_position_audit.json').write_text(json.dumps(provenance, indent=2)+'\n')
    return sources


if __name__ == '__main__':
    prepare()
