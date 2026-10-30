#!/usr/bin/env python3
"""Check rebuilt outputs against the current manuscript and unit-level evidence."""
from pathlib import Path
from statistics import mean, stdev
import csv
import hashlib
import json
import math
import re

ROOT = Path(__file__).resolve().parents[1]


def read_csv(name):
    with (ROOT / 'figure_data' / name).open(newline='') as handle:
        return list(csv.DictReader(handle))


def close(a, b):
    assert math.isclose(float(a), float(b), abs_tol=1e-12, rel_tol=1e-12), (a, b)


def main():
    manifest = json.loads((ROOT / 'provenance/publication_manifest.json').read_text())
    for record in manifest['tables']:
        path = ROOT / record['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], record['label']
    macros = dict(re.findall(r'^\\newcommand\{\\(\w+)\}\{(.*)\}$', (ROOT / 'numbers.tex').read_text(), re.M))
    assert macros == manifest['numeric_macros'], 'Numeric macros differ from the manuscript'
    for figure in manifest['figures']:
        path = ROOT / figure['file']
        assert path.exists() and path.stat().st_size > 1000, figure['label']

    budget = read_csv('budget.csv')
    audit = json.loads((ROOT / 'provenance/budget_endpoint_audit.json').read_text())
    assert len(budget) == 24
    for condition in audit['conditions']:
        rows = [r for r in budget if r['dataset'] == condition['dataset'] and int(r['budget']) == condition['budget']]
        assert {r['replicate'] for r in rows} == {'0', '1', '2'}
        assert len({r['n_host'] for r in rows}) == len({r['n_target'] for r in rows}) == 1
        values = [float(r['host_forced_fraction']) for r in rows]
        assert all(0 <= v <= 1 for v in values)
        close(mean(values), condition['host_forced_fraction']['mean'])
        close(stdev(values), condition['host_forced_fraction']['sd'])

    beta = read_csv('beta_by_unit.csv')
    beta_summary = read_csv('beta_summary.csv')
    assert len(beta) == 492 and len(beta_summary) == 164
    for row in beta_summary:
        units = [r for r in beta if r['dataset'] == row['dataset'] and r['beta'] == row['beta']]
        assert len({r['unit'] for r in units}) == len(units) == 3
        for metric in ['accuracy_pct', 'target_rejection_pct', 'host_rejection_pct', 'host_pore_time_s']:
            values = [float(r[metric]) for r in units]
            close(mean(values), row[metric + '_mean'])
            close(stdev(values), row[metric + '_sd'])

    units = read_csv('stopping_matched_units.csv')
    summaries = read_csv('stopping_matched_summary.csv')
    assert len(units) == 1092 and len(summaries) == 364
    for row in units:
        weight = float(row['weight_high'])
        assert 0 <= weight <= 1
        close((1-weight)*float(row['low_frr']) + weight*float(row['high_frr']), row['target_rejection'])
        close((1-weight)*float(row['low_host_pore']) + weight*float(row['high_host_pore']), row['host_pore'])
    for row in summaries:
        values = [float(r['host_pore']) for r in units if all(r[k] == row[k] for k in ['family', 'method', 'target_rejection'])]
        assert len(values) == 3
        close(mean(values), row['host_pore'])
        close(stdev(values), row['host_pore_sd'])
    readme = (ROOT.parent / 'README.md').read_text()
    assert not re.search(r'`[^`]+` sheet|Supplementary Section S', readme)
    report = {'tables_match_manuscript': len(manifest['tables']), 'numeric_macros_match': len(macros),
              'figures_present': len(manifest['figures']), 'budget_unit_rows': len(budget),
              'beta_candidates': len(beta), 'matched_unit_interpolations': len(units),
              'sample_sd_ddof': 1, 'status': 'passed'}
    (ROOT / 'review').mkdir(exist_ok=True)
    (ROOT / 'review/publication_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
