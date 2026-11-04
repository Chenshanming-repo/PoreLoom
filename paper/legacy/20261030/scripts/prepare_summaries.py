#!/usr/bin/env python3
"""Rebuild the current Bioinformatics tables and numeric macros from saved CSVs."""
from pathlib import Path
import csv
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    for directory in ['tables', 'review', 'provenance']:
        (ROOT / directory).mkdir(exist_ok=True)
    for name in ['aggregate_results.py', 'prepare_supplement_tables.py', 'prepare_summary_numbers.py']:
        subprocess.run([sys.executable, str(ROOT / 'scripts' / name)], check=True)
    audit = json.loads((ROOT / 'review/numerical_mean_sd_audit.json').read_text())
    (ROOT / 'provenance/numerical_claims.json').write_text(json.dumps({
        'aggregation': 'equal-weight mean +/- sample SD; n=3; ddof=1',
        'source': 'figure_data/', 'macros_with_sd': audit,
    }, indent=2) + '\n')
    # Export compact, machine-readable mean/SD summaries in addition to LaTeX.
    summary = json.loads((ROOT / 'review/mean_sd_summary.json').read_text())
    budget = json.loads((ROOT / 'review/supp_experiment_tables_audit.json').read_text())['budget']
    rows = []
    for condition in budget:
        row = {'dataset': condition['dataset'], 'budget': condition['budget'], 'n_training_repeats': 3}
        for key in ['acc', 'host_reject', 'frr', 'host_pore', 'host_forced_fraction']:
            row[key + '_mean'] = condition[key]['mean']
            row[key + '_sd'] = condition[key]['sd']
        rows.append(row)
    write_csv('budget_summary.csv', rows)
    rows = []
    for condition in summary['timing']:
        row = {key: condition[key] for key in ['scenario', 'chunk', 'batch', 'prefix_samples', 'method']}
        row.update(n=3, mean_ms=condition['seconds']['mean'] * 1000,
                   sd_ms=condition['seconds']['sd'] * 1000)
        rows.append(row)
    write_csv('timing_summary.csv', rows)
    print('Rebuilt 3 main tables, 8 supplementary tables, and current numeric macros.')


def write_csv(name, rows):
    with (ROOT / 'figure_data' / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
