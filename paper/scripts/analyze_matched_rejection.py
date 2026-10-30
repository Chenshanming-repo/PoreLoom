"""Plot matched-rejection TEST tradeoffs from archived candidates, without refitting."""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
from statistics import mean, stdev

from result_archive import ROOT, ARCHIVE
FAMILIES = {
    'Simulated': [f'sim_g{i}/r0' for i in (1, 2, 3)],
    'R9': [f'r9/r{i}' for i in (0, 1, 2)],
    'R10 mixed': [f'r10_run0{i}/r0' for i in (1, 2, 3)],
    'R10 sputum': [f'sputum_g{i}/r0' for i in (1, 2, 3)],
}
METHODS = ['main', 'fixed_len', 'const_thr', 'calib_thr']
GRID_STEP = 0.005
MAX_REJECTION = 0.20


def frontier(points):
    ordered = sorted(enumerate(points), key=lambda item: (
        item[1]['target_command_frr'], item[1]['host_mean_pore_seconds'], item[0]))
    edge, best = [], float('inf')
    for index, point in ordered:
        x, y = point['target_command_frr'], point['host_mean_pore_seconds']
        if edge and x == edge[-1][1]:
            continue
        if y < best:
            edge.append((index, x, y))
            best = y
    return edge


def write_csv(path, records):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


def prepare():
    curves, provenance, support_by_family, levels_by_family = {}, [], {}, {}
    for family, units in FAMILIES.items():
        for unit in units:
            for method in METHODS:
                relative = f'results/final_20261014/ablation/{unit}/{method}/metrics.json'
                source = ARCHIVE / relative
                points = json.loads(source.read_text())['test_curve']
                edge = frontier(points)
                assert len(edge) >= 2
                curves[unit, method] = (edge, relative)
                provenance.append({'family': family, 'unit': unit, 'method': method,
                                   'source': relative, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                                   'candidates': len(points), 'frontier_points': len(edge),
                                   'support': [edge[0][1], edge[-1][1]]})
        supports = [curves[u, m][0] for u in units for m in METHODS]
        low = max(edge[0][1] for edge in supports)
        high = min(edge[-1][1] for edge in supports)
        support_by_family[family] = [low, high]
        first = math.ceil(low / GRID_STEP - 1e-12)
        last = math.floor(min(high, MAX_REJECTION) / GRID_STEP + 1e-12)
        levels_by_family[family] = [round(i * GRID_STEP, 8) for i in range(first, last+1)]
    assert [len(v) for v in levels_by_family.values()] == [30, 23, 17, 21]
    rows, exact_hits = [], []
    for family, units in FAMILIES.items():
        for unit in units:
            for method in METHODS:
                edge, relative = curves[unit, method]
                for level in levels_by_family[family]:
                    exact = [p for p in edge if math.isclose(p[1], level, rel_tol=0, abs_tol=1e-12)]
                    if exact:
                        assert len(exact) == 1
                        low = high = exact[0]
                        weight, pore = 0.0, low[2]
                        exact_hits.append({'unit': unit, 'method': method, 'target_rejection': level,
                                           'candidate_index': low[0]})
                    else:
                        brackets = [(a, b) for a, b in zip(edge, edge[1:]) if a[1] < level < b[1]]
                        assert len(brackets) == 1, (unit, method, level)
                        low, high = brackets[0]
                        weight = (level-low[1]) / (high[1]-low[1])
                        pore = (1-weight)*low[2] + weight*high[2]
                    assert 0 <= weight <= 1
                    assert math.isclose((1-weight)*low[1]+weight*high[1], level, rel_tol=0, abs_tol=1e-12)
                    assert min(low[2], high[2]) <= pore <= max(low[2], high[2])
                    rows.append({'family': family, 'unit': unit, 'method': method,
                                 'target_rejection': level, 'host_pore': pore,
                                 'low_frr': low[1], 'high_frr': high[1],
                                 'low_host_pore': low[2], 'high_host_pore': high[2],
                                 'weight_high': weight, 'low_candidate_index': low[0],
                                 'high_candidate_index': high[0], 'source': relative})
    assert len(rows) == 1092 and len(exact_hits) == 6
    summary, winners = [], {}
    for family in FAMILIES:
        winners[family] = {m: 0 for m in METHODS}
        for level in levels_by_family[family]:
            means = {}
            for method in METHODS:
                values = [r['host_pore'] for r in rows if
                          (r['family'], r['target_rejection'], r['method']) == (family, level, method)]
                assert len(values) == 3
                means[method] = mean(values)
                summary.append({'family': family, 'target_rejection': level, 'method': method,
                                'n_units': 3, 'host_pore': means[method],
                                'minimum': min(values), 'maximum': max(values), 'host_pore_sd': stdev(values)})
            winners[family][min(means, key=means.get)] += 1
        for method in METHODS:
            values = [r['host_pore'] for r in summary if (r['family'], r['method']) == (family, method)]
            assert all(b <= a+1e-12 for a, b in zip(values, values[1:]))
    assert len(summary) == 364
    write_csv(ROOT / 'figure_data/stopping_matched_units.csv', rows)
    write_csv(ROOT / 'figure_data/stopping_matched_summary.csv', summary)
    audit = {'analysis': 'Retrospective interpolated TEST tradeoff curves',
             'grid_step': GRID_STEP, 'maximum_displayed_target_rejection': MAX_REJECTION,
             'grid_rule': "Within each dataset's common candidate support, take every 0.5-percentage-point level not exceeding 20%; no ranking-based filtering.",
             'levels_by_family': levels_by_family, 'support_by_family': support_by_family,
             'unit_results': len(rows), 'summary_results': len(summary), 'exact_hits': exact_hits,
             'maximum_bracket_width': max(r['high_frr']-r['low_frr'] for r in rows),
             'extrapolation': False, 'original_dev_b_operating_points_changed': False,
             'candidate_scope': 'All 41 FACP penalties; archived DEV-B accuracy-time frontier candidates for each threshold rule.',
             'method': 'Remove TEST target-rejection/time dominated candidates within each unit and method; retain exact grid hits or interpolate adjacent points; equally average three units and report sample SD (ddof=1).',
             'lowest_mean_counts_on_grid': winners, 'sources': provenance, 'summary': summary}
    (ROOT / 'provenance/stopping_matched_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    return summary, audit


def render(summary, audit):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/poreloom-bioinformatics-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, FuncFormatter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .65, 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    style = {
        'main': ('FACP', '#008B87', '-', 'o'),
        'fixed_len': ('Fixed length', '#7C8795', '--', 's'),
        'const_thr': ('Constant threshold', '#D88A21', '-.', '^'),
        'calib_thr': ('Length-calibrated', '#A25BAB', ':', 'D'),
    }
    ticks = [[5.5, 10, 15, 20], [9, 12, 16, 20], [5.5, 8, 11, 13.5], [10, 12.5, 15, 17.5, 20]]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
    fig.subplots_adjust(left=.095, right=.97, bottom=.10, top=.86, hspace=.50, wspace=.28)
    handles = [Line2D([], [], label=label, color=color, linestyle=line, marker=marker,
                      linewidth=1.5, markersize=3) for label, color, line, marker in style.values()]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .995),
               ncol=2, frameon=False, fontsize=8.5, columnspacing=2.2, handlelength=2.5)
    for panel, (ax, family) in enumerate(zip(axes.flat, FAMILIES)):
        for method, (label, color, line, marker) in style.items():
            points = [r for r in summary if (r['family'], r['method']) == (family, method)]
            ax.fill_between([100*r['target_rejection'] for r in points],
                            [r['host_pore']-r['host_pore_sd'] for r in points],
                            [r['host_pore']+r['host_pore_sd'] for r in points],color=color,alpha=.10,linewidth=0)
            ax.plot([100*r['target_rejection'] for r in points], [r['host_pore'] for r in points],
                    color=color, linestyle=line, marker=marker, markersize=2.8,
                    markeredgewidth=.45, linewidth=1.55 if method == 'main' else 1.25)
        levels = audit['levels_by_family'][family]
        ax.set_xlim(100*levels[0], 100*levels[-1])
        ax.set_xticks(ticks[panel])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f'{value:g}'))
        ax.set_ylabel('Mean host pore time (s)', labelpad=4)
        ax.set_xlabel('Target rejection (%)', labelpad=3)
        title = 'Sputum' if family == 'R10 sputum' else family
        ax.set_title(f'{chr(97+panel)}  {title}', loc='left', fontsize=10, fontweight='bold', pad=6)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(labelsize=8, length=3, width=.6, pad=3)
        ax.grid(axis='y', color='#E3E6E8', linewidth=.55)
        ax.set_axisbelow(True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.x1+1 or box.y1 > fig.bbox.y1+1:
                outside.append(text.get_text())
    assert not outside, outside
    (ROOT / 'figures').mkdir(exist_ok=True)
    for extension in ['pdf', 'svg', 'png']:
        fig.savefig(ROOT / f'figures/figS_stopping_matched.{extension}', dpi=300)
    plt.close(fig)
    audit['figure'] = {'panels': 4, 'curves': 16, 'mean_points': len(summary),
                       'font_minimum_pt': 8, 'text_bounds_passed': True, 'bands': 'sample SD across three experimental units (ddof=1)'}


def saved():
    """Read saved interpolation results and their candidate provenance."""
    import pandas as pd
    summary = pd.read_csv(ROOT / 'figure_data/stopping_matched_summary.csv').to_dict('records')
    audit = json.loads((ROOT / 'provenance/stopping_matched_audit.json').read_text())
    return summary, audit


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', help='Rebuild interpolation from the completed archive')
    args = parser.parse_args()
    summary, audit = prepare() if args.prepare else saved()
    render(summary, audit)
    (ROOT / 'review/stopping_matched_render.json').write_text(json.dumps(audit['figure'], indent=2)+'\n')
    print(f"Rendered {len(summary)} means from {audit['unit_results']} unit-level interpolations.")
