#!/usr/bin/env python3
"""Plot the pre-holdout share of bacterial reads for each sputum group and split."""
import csv
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/poreloom-recomb-mpl")
from pathlib import Path
from shutil import copyfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('G1', 'G2', 'G3')
SPLITS = ('Training', 'Validation', 'Test')
FOLDS = {
    'Prevotella': ('Prevotella',),
    'Streptococcus': ('Streptococcus',),
    'Fusobacterium': ('Fusobacterium',),
    'Four genera': ('Rothia', 'Schaalia', 'Actinomyces', 'Lancefieldella'),
}


def read_counts():
    source = ROOT / 'data/revision_20261104/figS3_composition.csv'
    splits = {'train': 'Training', 'dev_a (new)': 'Validation', 'test': 'Test'}
    rows = []
    with source.open(newline='') as src:
        for row in csv.DictReader(src):
            if row['partition'] not in splits:
                continue
            total, pct = int(row['n_bact']), float(row['pct'])
            fold = 'Four genera' if row['fold'] == 'Actinomycetota' else row['fold']
            rows.append((row['group'].upper(), splits[row['partition']], fold,
                         round(total * pct / 100), total, pct))
    assert len(rows) == len(GROUPS) * len(SPLITS) * len(FOLDS)
    return rows


def render(rows, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
    cmap = LinearSegmentedColormap.from_list('holdout_share', [
        '#F4F8F9', '#C9E1E5', '#91BDC4', '#478995', '#12515E'])
    norm = Normalize(0, 8)
    shares = {(group, split, fold): pct for group, split, fold, _, _, pct in rows}
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.52))
    fig.subplots_adjust(left=0.175, right=0.998, top=0.70, bottom=0.012, wspace=0.086)
    for panel, (split, ax) in enumerate(zip(SPLITS, axes)):
        values = np.array([[shares[group, split, fold] for group in GROUPS] for fold in FOLDS])
        ax.pcolormesh(range(4), range(5), values, cmap=cmap, norm=norm,
                      edgecolors='white', linewidths=1, rasterized=False)
        ax.set(xlim=(0, 3), ylim=(4, 0))
        ax.set_title(split, fontsize=10, fontweight='bold', pad=3)
        ax.set_xticks(np.arange(3) + .5, GROUPS, fontsize=8.5)
        ax.xaxis.tick_top()
        ax.set_yticks(np.arange(4) + .5, list(FOLDS) if panel == 0 else [], fontsize=9)
        for label in ax.get_yticklabels()[:3]:
            label.set_fontstyle('italic')
        ax.tick_params(length=0, pad=3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i, j in np.ndindex(values.shape):
            value = values[i, j]
            ax.text(j + .5, i + .5, f'{value:.2f}', ha='center', va='center',
                    fontsize=9.5, color='white' if value >= 6.2 else '#0C2027')
    pdf = ROOT / 'figures/figS_taxon_holdouts.pdf'
    pdf.parent.mkdir(exist_ok=True)
    fig.savefig(pdf, metadata={'Title': 'Sputum taxon holdout proportions', 'CreationDate': None})
    fig.savefig(pdf.with_suffix('.svg'))
    fig.savefig(pdf.with_suffix('.png'), dpi=240, facecolor='white')
    copyfile(pdf, output.with_suffix('.pdf'))
    fig.savefig(output.with_suffix('.png'), dpi=240, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    data = read_counts()
    destination = ROOT / 'output/sputum_taxon_holdout_proportions'
    destination.parent.mkdir(exist_ok=True)
    with destination.with_suffix('.csv').open('w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(('group', 'split', 'holdout_taxa', 'reads_from_holdout_taxa',
                         'all_bacterial_reads', 'percent_of_bacterial_reads'))
        for group, split, fold, n, total, pct in data:
            writer.writerow((group, split, fold, n, total, f'{pct:.6f}'))
    render(data, destination)
