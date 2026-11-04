#!/usr/bin/env python3
"""Restore concise threshold and length-control summaries from archived records.

The output contains equal-weight means and sample SDs across distinct units.
Threshold differences are calculated within each unit before aggregation.
"""
from pathlib import Path
from statistics import mean, stdev
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT
FAMILIES = ['Simulated', 'R9', 'R10 mixed', 'R10 sputum']
METHODS = [('poreloom', 'PoreLoom', 'streaming'),
           ('readcurrent', 'ReadCurrent', 'w3000'),
           ('readcurrent', 'ReadCurrent', 'w6000'),
           ('nanodeep_upstream', 'NanoDeep', 'w3000'),
           ('nanodeep_upstream', 'NanoDeep', 'w6000'),
           ('squigglenet', 'SquiggleNet', 'w3000'),
           ('squigglenet', 'SquiggleNet', 'w6000')]
METRICS = ['acc', 'host_reject', 'frr', 'host_pore']
HASHES = {}


def load_csv(name):
    path = ARCHIVE / 'figure_data' / name
    HASHES[str(path.relative_to(ARCHIVE))] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def load_json(path):
    HASHES[str(path.relative_to(ARCHIVE))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def stats(values):
    values = list(values)
    assert len(values) in (1, 3) and all(math.isfinite(x) for x in values)
    return {'n': len(values), 'mean': mean(values),
            'sd': stdev(values) if len(values) > 1 else None, 'values': values}


def cell(stat, scale=1, digits=2):
    left = f"{stat['mean'] * scale:.{digits}f}"
    if float(left) == 0:
        left = left.lstrip('-')
    if stat['sd'] is None:
        return left
    return left + r' $\pm$ ' + f"{stat['sd'] * scale:.{digits}f}"


def main():
    source = load_csv('length_metrics.csv')
    paired = []
    comparisons = 0
    for row in load_csv('threshold_comparison.csv'):
        selected = next(value for value in source
                        if value['scope'] == 'all' and value['unit'] == row['unit']
                        and value['method'] == row['method'] and value['variant'] == 'w' + row['window'])
        entry = {key: row[key] for key in ['family', 'unit', 'method']}
        entry['variant'] = 'w' + row['window']
        for metric in METRICS:
            assert math.isclose(float(selected[metric]), float(row[metric + '_selected']), abs_tol=1e-12)
            entry[metric] = float(row[metric + '_selected']) - float(row[metric + '_default'])
            comparisons += 1
        paired.append(entry)
    assert len(paired) == 72

    threshold_rows = []
    parts = [r'\begin{tabular}{@{}llrrrr@{}}\toprule',
             r'Dataset & Method (samples) & $\Delta$Accuracy (pp) & $\Delta$Host rejection (pp) & $\Delta$Target rejection (pp) & $\Delta$Host time (s)\\\midrule']
    for family in FAMILIES:
        for index, (method, label, variant) in enumerate(METHODS[1:]):
            rows = [r for r in paired if (r['family'], r['method'], r['variant']) == (family, method, variant)]
            assert len(rows) == len({r['unit'] for r in rows}) == 3
            record = {'family': family, 'method': method, 'variant': variant,
                      'units': [r['unit'] for r in rows], **{k: stats(r[k] for r in rows) for k in METRICS}}
            threshold_rows.append(record)
            cells = [cell(record[k], 1 if k == 'host_pore' else 100, 3 if k == 'host_pore' else 2) for k in METRICS]
            parts.append(' & '.join([family if index == 0 else '', f'{label} ({variant[1:]})', *cells]) + r'\\')
        if family != FAMILIES[-1]:
            parts.append(r'\addlinespace')
    parts += [r'\bottomrule\end{tabular}', '']
    (ROOT / 'tables/supp_threshold_summary.tex').write_text('\n'.join(parts))

    parts = [r'\begingroup\setlength{\tabcolsep}{3.5pt}',
             r'\begin{tabular}{@{}lllrrrr@{}}\toprule',
             r'Dataset & Method & Observation & Accuracy (\%) & Hosts rejected (\%) & Targets rejected (\%) & Host pore time (s)\\\midrule']
    long_rows = []
    for family in FAMILIES:
        family_rows = [r for r in source if r['family'] == family and r['scope'] == 'ge7500']
        for index, (method, label, variant) in enumerate(METHODS):
            rows = [r for r in family_rows if (r['method'], r['variant']) == (method, variant)]
            assert len(rows) == len({r['unit'] for r in rows}) == 3
            # Every method uses the same class counts and exact read-subset hash in each unit.
            for row in rows:
                matches = [r for r in family_rows if r['unit'] == row['unit']]
                assert len(matches) == 7
                for key in ['n_reads', 'n_host', 'n_target', 'subset_sha256']:
                    assert len({r[key] for r in matches}) == 1
            record = {'family': family, 'method': method, 'variant': variant,
                      'units': [r['unit'] for r in rows], **{k: stats(float(r[k]) for r in rows) for k in METRICS}}
            long_rows.append(record)
            cells = [cell(record[k], 1 if k == 'host_pore' else 100, 3 if k == 'host_pore' else 2) for k in METRICS]
            observation = 'Budget 6000' if variant == 'streaming' else 'Window ' + variant[1:]
            method_label = '' if variant == 'w6000' else label
            parts.append(' & '.join([family if index == 0 else '', method_label, observation, *cells]) + r'\\')
        if family != FAMILIES[-1]:
            parts.append(r'\addlinespace')
    parts += [r'\bottomrule\end{tabular}', r'\endgroup', '']
    (ROOT / 'tables/supp_length_summary.tex').write_text('\n'.join(parts))

    report = {'aggregation': 'Equal-weight mean +/- sample SD, ddof=1; threshold changes paired within each unit.',
              'threshold_pairs': len(paired), 'threshold_rows': threshold_rows,
              'length_rows': long_rows, 'source_sha256': HASHES,
              'checks': {'selected_metrics_reconciled': comparisons,
                         'long_read_subset_identity_across_methods': True,
                         'mean_accuracy_improved': sum(r['acc']['mean'] > 0 for r in threshold_rows)}}
    assert report['checks']['mean_accuracy_improved'] == 21
    (ROOT / 'review/supp_control_tables_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'threshold_rows': len(threshold_rows), 'length_rows': len(long_rows), **report['checks']}, indent=2))


if __name__ == '__main__':
    main()
