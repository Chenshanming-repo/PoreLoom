#!/usr/bin/env python3
"""Aggregate archived experimental units without refitting or pooling reads.

All displayed dispersions are sample standard deviations (n=3, ddof=1).
Source measurements are the saved figure_data CSVs. The JSON report keeps
full-precision summaries and unit values for verification.
"""
from pathlib import Path
from statistics import mean, stdev
import csv
import hashlib
import json
import math
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT
FAMILIES = ['Simulated', 'R9', 'R10 mixed', 'R10 sputum']
METRICS = ['acc', 'host_reject', 'frr', 'host_pore']
METHODS = [('poreloom', 'streaming', 'PoreLoom', r'6000'),
           ('readcurrent', 'w3000', 'ReadCurrent', '3000'),
           ('readcurrent', 'w6000', 'ReadCurrent', '6000'),
           ('nanodeep_upstream', 'w3000', 'NanoDeep', '3000'),
           ('nanodeep_upstream', 'w6000', 'NanoDeep', '6000'),
           ('squigglenet', 'w3000', 'SquiggleNet', '3000'),
           ('squigglenet', 'w6000', 'SquiggleNet', '6000')]
STOPS = [('main', 'FACP'), ('fixed_len', 'Fixed length'),
         ('const_thr', 'Constant threshold'), ('calib_thr', 'Length-calibrated')]
HASHES = {}


def read(name):
    path = SOURCE / 'figure_data' / name
    HASHES[str(path.relative_to(SOURCE))] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def summarize(rows, keys, metrics, unit_key=None):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, []).append(row)
    output = []
    for key, group in groups.items():
        assert len(group) == 3, (keys, key, len(group))
        if unit_key:
            assert len({row[unit_key] for row in group}) == 3, (key, unit_key)
        entry = dict(zip(keys, key))
        entry['n'] = 3
        if unit_key:
            entry['units'] = [row[unit_key] for row in group]
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            assert all(math.isfinite(value) for value in values), (key, metric)
            entry[metric] = {'mean': mean(values), 'sd': stdev(values), 'values': values}
        output.append(entry)
    return output


def metric_cell(value, metric, bold=False):
    scale, precision = (1, 3) if metric == 'host_pore' else (100, 2)
    left = f"{value['mean'] * scale:.{precision}f}"
    right = f"{value['sd'] * scale:.{precision}f}"
    text = left + r' $\pm$ ' + right
    return r'\textbf{' + text + '}' if bold else text


def table(rows, methods, include_observation=False, bold_best=True):
    columns = 'lllrrrr' if include_observation else 'llrrrr'
    parts = ([r'\begingroup\setlength{\tabcolsep}{3.5pt}'] if include_observation else [])
    parts += [r'\begin{tabular}{@{}' + columns + r'@{}}', r'\toprule',
              r'Dataset & Method' + (r' & Observation' if include_observation else '') + r' & Accuracy (\%) & Hosts rejected (\%) & Targets rejected (\%) & Host pore time (s)\\\midrule']
    for family in FAMILIES:
        family_rows = [row for row in rows if row['family'] == family]
        best = {}
        for metric in METRICS:
            scale, digits = (1, 3) if metric == 'host_pore' else (100, 2)
            values = [round(row[metric]['mean'] * scale, digits) for row in family_rows]
            best[metric] = (min if metric in ('frr', 'host_pore') else max)(values)
        for i, (key, label) in enumerate(methods):
            row = next(row for row in family_rows if all(row[k] == v for k, v in key.items()))
            values = []
            for metric in METRICS:
                scale, digits = (1, 3) if metric == 'host_pore' else (100, 2)
                is_best = bold_best and round(row[metric]['mean'] * scale, digits) == best[metric]
                values.append(metric_cell(row[metric], metric, is_best))
            fields = [family if i == 0 else '', label]
            if include_observation:
                if i > 0 and label == methods[i - 1][1]:
                    fields[1] = ''
                variant = key['variant']
                fields.append('Budget 6000' if variant == 'streaming' else 'Window ' + variant[1:])
            parts.append(' & '.join(fields + values) + r'\\')
        if family != FAMILIES[-1]:
            parts.append(r'\addlinespace')
    parts += [r'\bottomrule', r'\end{tabular}']
    if include_observation:
        parts.append(r'\endgroup')
    parts.append('')
    return '\n'.join(parts)


def run():
    length = read('length_metrics.csv')
    overall = summarize([r for r in length if r['scope'] == 'all'], ['family', 'method', 'variant'], METRICS, 'unit')
    stops = summarize(read('stopping.csv'), ['family', 'variant'], METRICS, 'unit')
    facp = summarize([r for r in read('native_cp.csv') if r['state'] == 'C-p'], ['family', 'method'], METRICS, 'unit')
    main_methods = [({'method': method, 'variant': variant}, label) for method, variant, label, window in METHODS]
    stopping_methods = [({'variant': key}, label) for key, label in STOPS]
    facp_methods = [({'method': method}, label) for method, variant, label, window in METHODS if variant == 'w3000']
    simulations = read('icarust_extended.csv')
    simulation = summarize(simulations, ['strategy', 'permille'], ['relative_yield', 'operational_acc', 'host_reject', 'target_effective_reject', 'frr', 'host_mean_pore_seconds'], 'seed')
    previous_simulation = read('icarust_summary.csv')
    for old in previous_simulation:
        row = next(row for row in simulation if row['strategy'] == old['strategy'] and row['permille'] == old['permille'])
        for metric, old_metric in [('relative_yield', 'relative_yield'), ('frr', 'target_command_frr'), ('host_mean_pore_seconds', 'host_pore')]:
            assert math.isclose(row[metric]['mean'], float(old[old_metric]), abs_tol=1e-10)

    (ROOT / 'tables/main_overall.tex').write_text(table(overall, main_methods, True))
    (ROOT / 'tables/supp_stopping.tex').write_text(table(stops, stopping_methods))
    (ROOT / 'tables/supp_baseline_facp.tex').write_text(table(facp, facp_methods, bold_best=False))
    sim_methods = [('PoreLoom', 'PoreLoom', r'6000'), ('readcurrent', 'ReadCurrent', '3000'),
                   ('readcurrent_w6000', '', '6000'), ('nanodeep', 'NanoDeep', '3000'),
                   ('nanodeep_w6000', '', '6000'), ('squigglenet', 'SquiggleNet', '3000'),
                   ('squigglenet_w6000', '', '6000')]
    sim_parts = [r'\begingroup\setlength{\tabcolsep}{3.5pt}', r'\begin{tabular}{@{}llrrr@{}}\toprule', r'Method & Observation & 1\% & 10\% & 30\%\\\midrule']
    for strategy, label, window in sim_methods:
        cells = []
        for permille in ['10', '100', '300']:
            row = next(row for row in simulation if row['strategy'] == strategy and row['permille'] == permille)
            value = row['relative_yield']
            cell = f"{value['mean']:.2f}" + r' $\pm$ ' + f"{value['sd']:.2f}"
            if strategy == 'PoreLoom':
                cell = r'\textbf{' + cell + '}'
            cells.append(cell)
        observation = ('Budget ' if strategy == 'PoreLoom' else 'Window ') + window
        sim_parts.append(' & '.join([label, observation, *cells]) + r'\\')
    sim_parts += [r'\bottomrule\end{tabular}\endgroup', '']
    (ROOT / 'tables/closed_loop.tex').write_text('\n'.join(sim_parts))

    timing = [dict(row, prefix_samples='-1' if row['scenario'] == 'mixed' else row['n']) for row in read('timing_raw.csv')]
    summary = {
        'aggregation': 'Equal-weight mean and sample standard deviation; n=3 experimental units; ddof=1. Rates remain fractions in JSON.',
        'overall': overall, 'stopping': stops, 'baseline_facp': facp, 'simulation': simulation,
        'budget': summarize(read('budget.csv'), ['dataset', 'budget'], METRICS, 'replicate'),
        'encoder': summarize(read('encoder.csv'), ['family', 'variant'], METRICS, 'unit'),
        'length_strata': summarize([r for r in length if r['scope'] != 'all'], ['family', 'method', 'variant', 'scope'], METRICS, 'unit'),
        'lso': summarize(read('lso.csv'), ['fold', 'method', 'scope'], ['frr_main', 'frr_fold', 'dfrr', 'host_main', 'host_fold', 'dhost', 'n_held'], 'group'),
        'decision_positions': summarize(read('decision_position_summary.csv'), ['family', 'category'], ['no_decision_fraction', 'fraction_by_1200', 'fraction_by_3000', 'fraction_at_6000'], 'unit'),
        'fixed_prefix': summarize(read('fixed_prefix.csv'), ['family', 'usable_samples'], ['acc'], 'unit'),
        'timing': summarize(timing, ['scenario', 'chunk', 'batch', 'prefix_samples', 'method'], ['seconds'], 'rep'),
        'source_sha256': HASHES,
        'checks': {'all_summary_groups_n': 3, 'sample_sd_ddof': 1,
                   'closed_loop_archived_means_match': True,
                   'main_overall_rows': len(overall), 'stopping_rows': len(stops), 'baseline_facp_rows': len(facp), 'closed_loop_rows': 7},
    }
    (ROOT / 'review/mean_sd_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['checks'], indent=2))


if __name__ == '__main__':
    run()
