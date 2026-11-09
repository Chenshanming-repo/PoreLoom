#!/usr/bin/env python3
"""Rebuild the current paper's figures and tables from its frozen input snapshot.

Run from any directory. This artifact-only build needs no GPU, raw sequencing
data, external result archive, or TeX installation.
"""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    for name in ('review', 'figures', 'tables', 'output'):
        (ROOT / name).mkdir(exist_ok=True)
    scripts = [
        'prepare_revision_tables.py', 'draw_overview.py', 'plot_mean_sd.py',
        'plot_encoding.py', 'plot_sputum_taxon_holdouts.py', 'plot_beta.py',
        'export_example_reads.py',
    ]
    with (ROOT / 'review/rebuild.log').open('w') as log:
        for script in scripts:
            print('Running:', script, flush=True)
            subprocess.run([sys.executable, str(ROOT / 'scripts' / script)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    shutil.copyfile(ROOT / 'figures/fig01_overview.png',
                    ROOT.parent / 'docs/poreloom_overview.png')
    subprocess.run([sys.executable, str(ROOT / 'scripts/verify_publication.py')],
                   cwd=ROOT, check=True)
    print('Rebuilt 8 figures (PDF/PNG/SVG), 10 tables, and the README overview.')


if __name__ == '__main__':
    main()
