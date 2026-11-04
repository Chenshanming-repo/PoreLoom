#!/usr/bin/env python3
"""Recover concise experimental summaries for the supplement from archived runs.

Each condition has three original units. Rates are displayed as percentages;
paired changes use percentage points. SD is the sample SD (ddof=1). No model,
operating point, or read-level outcome is changed by this presentation step.
"""
from pathlib import Path
from statistics import mean, stdev
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'figure_data'
FAMILIES = ['Simulated', 'R9', 'R10 mixed', 'R10 sputum']
METHODS = [('poreloom', 'PoreLoom'), ('readcurrent', 'ReadCurrent'),
           ('nanodeep_upstream', 'NanoDeep'), ('squigglenet', 'SquiggleNet')]
METRICS = ['acc', 'host_reject', 'frr', 'host_pore']
HASHES = {}


def read(name):
    path = SOURCE / name
    HASHES[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def stat(values, units):
    assert len(values) == len(units) == len(set(units)) == 3
    assert all(math.isfinite(value) for value in values)
    return {'mean': mean(values), 'sd': stdev(values), 'n': 3,
            'units': units, 'values': values}


def group(rows, unit_key, metrics):
    return {key: stat([float(row[key]) for row in rows], [row[unit_key] for row in rows])
            for key in metrics}


def cell(value, scale=100, digits=2, signed=False):
    pattern = f'{{:{"+" if signed else ""}.{digits}f}}'
    return '$' + pattern.format(value['mean'] * scale) + r' \pm ' + f"{value['sd'] * scale:.{digits}f}" + '$'


def line(fields):
    return ' & '.join(fields) + r'\\'


def write_table(name, spec, header, body):
    content = [r'\begin{tabular}{@{}' + spec + r'@{}}', r'\toprule',
               line(header) + r'\midrule', *body, r'\bottomrule', r'\end{tabular}', '']
    (ROOT / 'tables' / name).write_text('\n'.join(content))


def run():
    audit = {'aggregation': 'equal-weight mean +/- sample SD; n=3; ddof=1; rates stored as fractions',
             'encoder_direction': 'streaming minus sliding-window, paired within unit',
             'lso_direction': 'taxon holdout minus full training, paired within group'}

    encoder = read('encoder.csv')
    encoder_summary, body = [], []
    for family in FAMILIES:
        pairs = {variant: {row['unit']: row for row in encoder if row['family'] == family and row['variant'] == variant}
                 for variant in ['main', 'window_cp']}
        assert set(pairs['main']) == set(pairs['window_cp'])
        units = sorted(pairs['main'])
        record = {'family': family}
        for key in METRICS:
            record[key] = stat([float(pairs['main'][unit][key]) - float(pairs['window_cp'][unit][key]) for unit in units], units)
        for unit in units:
            for metric, archived_delta in [('acc', 'dacc'), ('host_pore', 'dpore')]:
                expected = float(pairs['window_cp'][unit][metric]) - float(pairs['main'][unit][metric])
                assert math.isclose(expected, float(pairs['window_cp'][unit][archived_delta]), abs_tol=1e-12)
        encoder_summary.append(record)
        body.append(line([family] + [cell(record[key], 1 if key == 'host_pore' else 100, 3 if key == 'host_pore' else 2, True) for key in METRICS]))
    write_table('supp_encoder_summary.tex', 'lrrrr',
                ['Dataset', r'$\Delta$Accuracy (pp)', r'$\Delta$Host rejection (pp)',
                 r'$\Delta$Target rejection (pp)', r'$\Delta$Host pore time (s)'], body)
    audit['encoder'] = encoder_summary

    budget = read('budget.csv')
    for row in budget:
        assert 0 <= float(row['host_forced_fraction']) <= 1
    budget_metrics = METRICS + ['host_forced_fraction']
    budget_summary, body = [], []
    for i, (dataset, label) in enumerate([('r10_run01', 'R10 mixed, group 1'), ('sputum_g1', 'R10 sputum, group 1')]):
        budgets = sorted({int(row['budget']) for row in budget if row['dataset'] == dataset})
        for j, maximum in enumerate(budgets):
            rows = [row for row in budget if row['dataset'] == dataset and int(row['budget']) == maximum]
            assert all(row['interval'] == '0.4' and row['service'] == 'nominal' for row in rows)
            assert {row['replicate'] for row in rows} == {'0', '1', '2'}
            record = {'dataset': dataset, 'budget': maximum, **group(rows, 'replicate', budget_metrics)}
            budget_summary.append(record)
            body.append(line([label if j == 0 else '', str(maximum)] + [cell(record[key], 1 if key == 'host_pore' else 100, 3 if key == 'host_pore' else 2) for key in budget_metrics]))
        if i == 0:
            body.append(r'\addlinespace')
    write_table('supp_budget_summary.tex', 'llrrrrr',
                ['Data group', 'Budget', r'\shortstack{Accuracy\\(\%)}',
                 r'\shortstack{Hosts rejected\\(\%)}', r'\shortstack{Targets rejected\\(\%)}',
                 r'\shortstack{Host pore time\\(s)}',
                 r'\shortstack{Hosts decided\\at budget (\%)}'], body)
    audit['budget'] = budget_summary
    audit['budget_host_forced_definition'] = ('Fraction of all TEST host reads with a completed Accept or Reject '
                                            'decision at the budget-forced observation; three training repeats '
                                            'on the same fixed data group and TEST set.')

    simulation = read('icarust_extended.csv')
    sim_methods = [('No-AS', 'Nonselective'), ('PoreLoom', r'PoreLoom (6000)'),
                   ('readcurrent', 'ReadCurrent (3000)'), ('readcurrent_w6000', 'ReadCurrent (6000)'),
                   ('nanodeep', 'NanoDeep (3000)'), ('nanodeep_w6000', 'NanoDeep (6000)'),
                   ('squigglenet', 'SquiggleNet (3000)'), ('squigglenet_w6000', 'SquiggleNet (6000)')]
    sim_keys = ['operational_acc', 'host_reject', 'target_effective_reject', 'host_mean_pore_seconds']
    simulation_summary, body = [], []
    for fraction in [10, 100, 300]:
        for index, (strategy, label) in enumerate(sim_methods):
            rows = [row for row in simulation if int(row['permille']) == fraction and row['strategy'] == strategy]
            for row in rows:
                n_host, n_target = float(row['host_reads']), float(row['target_reads'])
                expected = (n_host * float(row['host_reject']) + n_target * (1 - float(row['target_effective_reject']))) / (n_host + n_target)
                assert math.isclose(expected, float(row['operational_acc']), abs_tol=1e-12)
            record = {'permille': fraction, 'strategy': strategy, **group(rows, 'seed', sim_keys)}
            simulation_summary.append(record)
            body.append(line([str(fraction // 10) + r'\%' if index == 0 else '', label] + [cell(record[key], 1 if key == 'host_mean_pore_seconds' else 100, 3 if key == 'host_mean_pore_seconds' else 2) for key in sim_keys]))
        if fraction != 300:
            body.append(r'\addlinespace')
    write_table('supp_simulation_summary.tex', 'llrrrr',
                ['Targets', 'Method (samples)', r'Accuracy (\%)', r'Hosts ejected (\%)',
                 r'Targets ejected (\%)', 'Host pore time (s)'], body)
    audit['simulation'] = simulation_summary

    lso = read('lso.csv')
    folds = [('Prevotella', r'\textit{Prevotella}'), ('Streptococcus', r'\textit{Streptococcus}'),
             ('Fusobacterium', r'\textit{Fusobacterium}'), ('Actinomycetota', 'Four genera')]
    lso_keys = ['frr_main', 'frr_fold', 'dfrr', 'dhost']
    lso_summary, counts, body = [], [], []
    for fold, label in folds:
        count_rows = [row for row in lso if row['fold'] == fold and row['method'] == 'poreloom']
        counts.append({'fold': fold, 'n_held': stat([float(row['n_held']) for row in count_rows], [row['group'] for row in count_rows])})
        for index, (method, method_label) in enumerate(METHODS):
            rows = [row for row in lso if row['fold'] == fold and row['method'] == method]
            for row in rows:
                for delta, before, after in [('dfrr', 'frr_main', 'frr_fold'), ('dhost', 'host_main', 'host_fold')]:
                    assert math.isclose(float(row[delta]), float(row[after]) - float(row[before]), abs_tol=1e-12)
                assert float(row['n_held']) == next(float(r['n_held']) for r in count_rows if r['group'] == row['group'])
            record = {'fold': fold, 'method': method, **group(rows, 'group', lso_keys)}
            lso_summary.append(record)
            body.append(line([label if index == 0 else '', method_label] + [cell(record[key], signed=key in ['dfrr', 'dhost']) for key in lso_keys]))
        if fold != 'Actinomycetota':
            body.append(r'\addlinespace')
    write_table('supp_lso_summary.tex', 'llrrrr',
                ['Held-out taxa', 'Method', r'\shortstack{Target rejection (\%)\\Full training}',
                 r'\shortstack{Target rejection (\%)\\Taxon holdout}',
                 r'\shortstack{$\Delta$Target rejection\\(pp)}',
                 r'\shortstack{$\Delta$Host rejection\\(pp)}'], body)
    audit['lso'] = lso_summary
    audit['lso_held_target_counts'] = counts

    previous = json.loads((ROOT / 'review/mean_sd_summary.json').read_text())
    reconciled = 0
    for section, rows, keys, metrics in [('budget', budget_summary, ['dataset', 'budget'], METRICS),
                                        ('simulation', simulation_summary, ['permille', 'strategy'], sim_keys),
                                        ('lso', lso_summary, ['fold', 'method'], lso_keys)]:
        for row in rows:
            candidates = [old for old in previous[section] if all(str(old[key]) == str(row[key]) for key in keys)]
            assert len(candidates) == 1
            for metric in metrics:
                for field in ['mean', 'sd']:
                    assert math.isclose(row[metric][field], candidates[0][metric][field], abs_tol=1e-12)
                reconciled += 1
    audit['source_sha256'] = HASHES
    audit['checks'] = {'sample_sd_ddof': 1, 'n_per_condition': 3,
                       'absolute_or_paired_stats_reconciled': reconciled,
                       'encoder_rows': 4, 'budget_rows': 8, 'simulation_rows': 24,
                       'lso_rows': 16, 'numeric_cells': 216,
                       'budget_endpoint_unit_rows': len(budget),
                       'encoder_archived_within_unit_differences_verified': True,
                       'effective_simulation_accuracy_formula_verified': True,
                       'lso_within_group_differences_verified': True}
    (ROOT / 'review/supp_experiment_tables_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'checks': audit['checks'], 'held_out_counts': counts}, indent=2))


if __name__ == '__main__':
    run()
