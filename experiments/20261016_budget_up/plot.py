#!/usr/bin/env python3
"""Figure: host reject, target false rejection and host pore time vs the decision budget (r10_run01, sputum_g1).

TEST 0.4 s, nominal service, max DEV-B ACC beta; line = mean of replicates r0-r2, dots = the replicates.

    plot.py [--root $SE/results/budget_up_20261016/runs] [--out $SE/results/budget_up_20261016/figures/budget_rates]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path('/path/to/poreloom_work/results/budget_up_20261016/runs')
DATASETS = [('r10_run01', 'R10 mix (HG002 / Zymo, 4 kHz)', (6000, 9000, 12000, 15000, 18000), '#2a78d6'),
            ('sputum_g1', 'Sputum (5 kHz)', (6000, 9000, 12000), '#eb6834')]
PANELS = [('host_command_reject', 'Host reject rate (%)', 100.), ('target_command_frr', 'Target false rejection (%)', 100.),
          ('host_mean_pore_seconds', 'Host mean pore time (s)', 1.)]
INK, INK2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#fcfcfb'


def load(root, ds, budgets):
    out = {}
    for b in budgets:
        rows = []
        for r in (0, 1, 2):
            m = json.loads((root/ds/f'b{b}_r{r}'/'test'/'metrics.json').read_text())
            rows.append(next(x for x in m['rows'] if x['interval'] == .4 and x['service'] == 'nominal'))
        out[b] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--out', type=Path, default=ROOT.parent/'figures'/'budget_rates')
    a = ap.parse_args()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'savefig.dpi': 300, 'pdf.fonttype': 42})
    data = {ds: load(a.root, ds, budgets) for ds, _, budgets, _ in DATASETS}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), facecolor=SURFACE)
    for ax, (key, title, scale) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        ax.axvline(6000, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
        for ds, label, budgets, color in DATASETS:
            reps = np.array([[row[key]*scale for row in data[ds][b]] for b in budgets])
            x = np.array(budgets, float)
            for j in range(reps.shape[1]):
                ax.scatter(x, reps[:, j], s=14, color=color, alpha=.35, linewidths=0, zorder=2)
            ax.plot(x, reps.mean(1), color=color, lw=2, zorder=3, solid_capstyle='round')
            ax.scatter(x, reps.mean(1), s=48, color=color, edgecolors=SURFACE, linewidths=2, zorder=4, label=label)
            ax.annotate(f'{reps.mean(1)[-1]:.2f}' if scale == 1 else f'{reps.mean(1)[-1]:.1f}', (x[-1], reps.mean(1)[-1]),
                        xytext=(7, 0), textcoords='offset points', va='center', fontsize=8, color=INK2)
            ax.annotate(f'{reps.mean(1)[0]:.2f}' if scale == 1 else f'{reps.mean(1)[0]:.1f}', (x[0], reps.mean(1)[0]),
                        xytext=(-7, 0), textcoords='offset points', va='center', ha='right', fontsize=8, color=INK2)
        ax.set_title(title, loc='left', fontsize=10, color=INK, fontweight='bold')
        ax.set_xticks([6000, 9000, 12000, 15000, 18000]); ax.set_xticklabels(['6k\n(default)', '9k', '12k', '15k', '18k'])
        ax.set_xlim(3800, 19600)
        ax.grid(axis='y', color=GRID, lw=.8); ax.set_axisbelow(True)
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        ax.tick_params(length=0)
    axes[1].set_xlabel('Decision budget (signal samples after the 1500-sample skip)', color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.5, 1.0), fontsize=9,
               labelcolor=INK)
    fig.text(.5, -.04, 'TEST, 0.4 s chunks, operating point locked on DEV-B (max ACC). '
                       'Large dots + line: mean of 3 training replicates; small dots: replicates.',
             ha='center', fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, .9))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'): fig.savefig(a.out.with_suffix('.' + ext), bbox_inches='tight', facecolor=SURFACE)
    print(a.out.with_suffix('.png'))


if __name__ == '__main__':
    main()
