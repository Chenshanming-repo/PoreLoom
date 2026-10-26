#!/usr/bin/env python3
"""Verify the archived FACP beta sweep and render the compact supplementary figure.

No models are fitted and no new operating points are selected. The default
command plots saved figure_data CSVs; --prepare verifies the completed archive.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
from statistics import mean

from result_archive import ROOT, ARCHIVE
FAMILIES = {
    'Simulated': ['sim_g1/r0', 'sim_g2/r0', 'sim_g3/r0'],
    'R9': ['r9/r0', 'r9/r1', 'r9/r2'],
    'R10 mixed': ['r10_run01/r0', 'r10_run02/r0', 'r10_run03/r0'],
    'R10 sputum': ['sputum_g1/r0', 'sputum_g2/r0', 'sputum_g3/r0'],
}
METRICS = {
    'accuracy_pct': ('acc', 'acc', 100.0, 'Accuracy (%)', (48, 100)),
    'target_rejection_pct': ('frr', 'target_command_frr', 100.0, 'Target rejection (%)', (0, 68)),
    'host_rejection_pct': ('host_reject', 'host_command_reject', 100.0, 'Host rejection (%)', (0, 102)),
    'host_pore_time_s': ('host_pore', 'host_mean_pore_seconds', 1.0, 'Mean host pore time (s)', (0, 16.5)),
}

def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)

def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

def prepare():
    records, selected, provenance = [], [], []
    for dataset, units in FAMILIES.items():
        for unit in units:
            relative = f'results/final_20261014/ablation/{unit}/main/metrics.json'
            source = ARCHIVE / relative
            archived = json.loads(source.read_text())
            points = sorted(archived['test_curve'], key=lambda p: p['beta'])
            assert len(points) == 41 and len({p['beta'] for p in points}) == 41
            provenance.append({'source': relative, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
            for i, point in enumerate(points):
                beta = float(point['beta'])
                assert close(beta, 0.03 * 10 ** (i / 10))
                is_selected = close(beta, archived['locked']['beta'])
                row = {'dataset': dataset, 'unit': unit, 'beta': beta, 'selected_on_dev_b': is_selected}
                for name, (_, archive_key, scale, _, _) in METRICS.items():
                    value = float(point[archive_key])
                    assert math.isfinite(value)
                    if is_selected:
                        assert close(value, archived['locked'][archive_key])
                    row[name] = value * scale
                assert abs(row['accuracy_pct'] - (row['host_rejection_pct'] + 100 - row['target_rejection_pct']) / 2) < 1e-8
                row['source'] = relative
                records.append(row)
                if is_selected:
                    selected.append(row)
    assert len(records) == 492 and len(selected) == 12
    summary = []
    for dataset in FAMILIES:
        for beta in sorted({r['beta'] for r in records}):
            points = [r for r in records if r['dataset'] == dataset and r['beta'] == beta]
            assert len(points) == 3
            row = {'dataset': dataset, 'beta': beta, 'n_units': 3}
            for name in METRICS:
                values = [p[name] for p in points]
                row.update({name + '_mean': mean(values), name + '_min': min(values), name + '_max': max(values)})
            summary.append(row)
    endpoints = [r for r in summary if r['beta'] in (0.03, 300.0)]
    (ROOT / 'review').mkdir(exist_ok=True)
    write_csv(ROOT / 'figure_data/beta_by_unit.csv', records)
    write_csv(ROOT / 'figure_data/beta_summary.csv', summary)
    write_csv(ROOT / 'figure_data/beta_selected.csv', selected)
    write_csv(ROOT / 'figure_data/beta_endpoints.csv', endpoints)
    audit = {
        'candidate_rows': len(records), 'units': 12, 'betas_per_unit': 41,
        'selected_points': 12, 'verified_against_archived_metrics': True,
        'sources': provenance,
        'interpretation': 'Diagnostic TEST sweep of separately fitted policies with frozen encoders; original DEV-B selections retained.',
        'aggregation': 'Equal-weight mean and min-max of 3 experimental units at each common beta; bands are not confidence intervals.',
    }
    (ROOT / 'provenance/beta_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return records, summary, selected

def render(records, summary, selected):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/poreloom-bioinformatics-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FixedFormatter, MaxNLocator
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .65, 'pdf.fonttype': 42, 'savefig.facecolor': 'white',
    })
    colors = ['#437FBF', '#A656B9', '#008E8B', '#D66639']
    styles = ['-', '--', '-.', ':']
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.65))
    fig.subplots_adjust(left=.095, right=.985, bottom=.105, top=.88, hspace=.50, wspace=.30)
    legend = [Line2D([], [], color=c, linestyle=s, linewidth=1.5, label=f) for f, c, s in zip(FAMILIES, colors, styles)]
    legend.append(Line2D([], [], marker='*', color='none', markerfacecolor='#777777', markeredgecolor='#222222', markersize=8, label='DEV-B selection'))
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.53, 1.005), ncol=3,
               frameon=False, fontsize=8.5, columnspacing=1.25, handlelength=2)
    for panel, (ax, (name, (_, _, _, label, ylim))) in enumerate(zip(axes.flat, METRICS.items())):
        for dataset, color, style in zip(FAMILIES, colors, styles):
            points = [p for p in summary if p['dataset'] == dataset]
            beta = [p['beta'] for p in points]
            ax.fill_between(beta, [p[name + '_min'] for p in points], [p[name + '_max'] for p in points], color=color, alpha=.10, linewidth=0)
            ax.plot(beta, [p[name + '_mean'] for p in points], color=color, linestyle=style, linewidth=1.45)
            for point in [p for p in selected if p['dataset'] == dataset]:
                ax.scatter(point['beta'], point[name], marker='*', s=36, facecolor=color, edgecolor='#222222', linewidth=.45, zorder=7)
        ax.set_xscale('log')
        ax.set_xlim(.026, 345)
        ax.set_ylim(*ylim)
        ax.set_ylabel(label, labelpad=4)
        ax.set_xlabel(r'$\beta$ (log scale)', labelpad=3)
        ax.xaxis.set_major_locator(FixedLocator([.03, .3, 3, 30, 300]))
        ax.xaxis.set_major_formatter(FixedFormatter(['0.03', '0.3', '3', '30', '300']))
        ax.tick_params(axis='x', which='minor', bottom=False)
        ax.tick_params(labelsize=8, length=3, width=.6, pad=3)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(axis='y', color='#E3E6E8', linewidth=.55)
        ax.set_axisbelow(True)
        ax.text(-.16, 1.04, chr(97 + panel), transform=ax.transAxes, fontweight='bold', fontsize=11, va='bottom')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            bbox = text.get_window_extent(renderer)
            if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > fig.bbox.x1 + 1 or bbox.y1 > fig.bbox.y1 + 1:
                outside.append(text.get_text())
    assert not outside, outside
    for extension in ['pdf', 'svg', 'png']:
        fig.savefig(ROOT / f'figures/figS_beta.{extension}', dpi=300)
    plt.close(fig)

def saved():
    """Read the checked-in publication data without requiring original archives."""
    import pandas as pd
    return tuple(pd.read_csv(ROOT / f'figure_data/{name}.csv').to_dict('records')
                 for name in ['beta_by_unit', 'beta_summary', 'beta_selected'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', help='Verify the completed archive and rebuild CSVs first')
    args = parser.parse_args()
    render(*(prepare() if args.prepare else saved()))
    print('Rendered figS_beta from 492 candidate outcomes and 12 DEV-B-selected policies.')
