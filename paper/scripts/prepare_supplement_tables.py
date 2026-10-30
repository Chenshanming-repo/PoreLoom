#!/usr/bin/env python3
"""Rebuild supplementary tables from saved publication CSVs."""
from pathlib import Path
from statistics import mean, stdev
import csv
import hashlib
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = ROOT / 'figure_data/data_partitions.csv'


def partitions():
    with PARTITIONS.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    for row in rows:
        assert int(row['test']) == 2 * int(row['per_class'])
    lines = [r'\begin{tabular}{@{}lrrrr@{}}\toprule',
             r'Dataset & TRAIN & DEV-A & DEV-B & TEST\\\midrule']
    audit = []
    for family, prefix in [('Simulated', 'Simulated '), ('R9', 'R9 '), ('R10 mixed', 'R10 mixed '), ('R10 sputum', 'Sputum ')]:
        group = [r for r in rows if r['group'].startswith(prefix)]
        assert len(group) == (1 if family == 'R9' else 3)
        result = {'family': family, 'n_distinct_partitions': len(group)}
        cells = []
        for key in ['train', 'dev_a', 'dev_b', 'test']:
            values = [int(r[key]) for r in group]
            result[key] = {'values': values, 'mean': mean(values), 'sd': stdev(values) if len(group) > 1 else None}
            cells.append(f'{mean(values):.0f}' + (r' $\pm$ ' + f'{stdev(values):.0f}' if len(group) > 1 else ''))
        lines.append(family + ' & ' + ' & '.join(cells) + r'\\')
        audit.append(result)
    lines += [r'\bottomrule\end{tabular}', '']
    (ROOT / 'tables/supp_partitions_summary.tex').write_text('\n'.join(lines))
    (ROOT / 'review/supp_partitions_audit.json').write_text(json.dumps({'rows': audit, 'source': str(PARTITIONS.relative_to(ROOT)), 'sha256': hashlib.sha256(PARTITIONS.read_bytes()).hexdigest()}, indent=2) + '\n')


if __name__ == '__main__':
    partitions()
    for name in ['prepare_supp_control_tables.py', 'prepare_supp_experiment_tables.py']:
        subprocess.run([sys.executable, str(ROOT / 'scripts' / name)], check=True)
