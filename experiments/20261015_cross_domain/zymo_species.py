#!/usr/bin/env python3
"""Per-species breakdown of the cross-dataset test: which Zymo D6322 species the
sputum models falsely reject on the R10 HG002 + Zymo TEST, against the R10 cohort's own model on the same reads.

Labels: the 30000 Zymo TEST reads of r10_run01/02/03, basecalled with Guppy 6.5.7 (dna_r10.4.1_e8.2_400bps_hac_prom),
aligned with minimap2 -x map-ont to the Zymo bacteria + S. cerevisiae R64 + GRCh38 (contigs tagged '<species>|').
A read gets the species of its primary alignment when MAPQ >= 20 and the aligned query covers >= 50% of the read;
otherwise 'unassigned'. Pass and fail basecalls are both used; reads without a basecall are 'no_call'.

    zymo_species.py assign --paf <aln.paf.gz> --fastq <dir> --ids <ids.txt> --out <labels.tsv>      (local)
    zymo_species.py report --labels <labels.tsv> --main <runs>/main --cd <runs>/cross_domain [--md ..] (HPC)
"""
import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_domain import METHODS, NAMES, SOURCES, TARGETS, correct  # noqa: E402

MAPQ, COVER = 20, .5


def assign(a):
    ids = open(a.ids).read().split()
    length = {}
    for f in sorted(Path(a.fastq).rglob('*.fastq*')):
        with (gzip.open(f, 'rt') if f.suffix == '.gz' else open(f)) as h:
            for i, line in enumerate(h):
                if i % 4 == 0: rid = line[1:].split()[0]
                elif i % 4 == 1: length[rid] = len(line.strip())
    best = {}
    with (gzip.open(a.paf, 'rt') if str(a.paf).endswith('.gz') else open(a.paf)) as h:
        for line in h:
            x = line.split('\t')
            if 'tp:A:P' not in line: continue
            q, qlen, qs, qe, sp, mq = x[0], int(x[1]), int(x[2]), int(x[3]), x[5].split('|')[0], int(x[11])
            cov = (qe - qs)/qlen
            if q not in best or mq > best[q][1]: best[q] = (sp, mq, cov)
    with open(a.out, 'w') as o:
        o.write('read_id\tspecies\tmapq\tcover\tbases\n')
        for r in ids:
            if r not in length: o.write(f'{r}\tno_call\t\t\t\n'); continue
            sp, mq, cov = best.get(r, ('unassigned', 0, 0.))
            lab = sp if (mq >= MAPQ and cov >= COVER) else 'unassigned'
            o.write(f'{r}\t{lab}\t{mq}\t{cov:.3f}\t{length[r]}\n')
    c = defaultdict(int)
    for line in open(a.out).read().splitlines()[1:]: c[line.split('\t')[1]] += 1
    print(json.dumps(dict(sorted(c.items(), key=lambda kv: -kv[1])), indent=1))


def report(a):
    lab = {}
    for line in open(a.labels).read().splitlines()[1:]:
        r, sp = line.split('\t')[:2]; lab[r] = sp
    rows = []
    for v in ('at5k', 'native'):
        for s in SOURCES:
            for t in TARGETS:
                for m in METHODS:
                    d = a.cd/v/f'{s}_on_{t}'/f'{m}_r0'
                    if not (d/'test'/'metrics.json').exists(): continue
                    ids, x, _, host = correct(d, m); ids0, x0, _, _ = correct(a.main/t/f'{m}_r0', m)
                    assert np.array_equal(ids, ids0)
                    g = np.array([lab.get(r, 'missing') for r in ids])
                    for k in np.unique(g[~host]):
                        sel = (g == k) & ~host
                        rows.append(dict(variant=v, src=s, tgt=t, method=m, species=str(k), n=int(sel.sum()),
                                         frr=float(1 - x[sel].mean()), frr_indomain=float(1 - x0[sel].mean())))
    tot = defaultdict(int)
    for r in rows:
        if r['variant'] == 'at5k' and r['method'] == 'poreloom' and r['src'] == SOURCES[0]: tot[r['species']] += r['n']
    L = ['# Zymo 逐物种 target 误拒：痰液模型 → R10 HG002 + Zymo（跨数据集）vs R10 自己的模型（域内）', '',
         f'物种标签：Guppy 6.5.7 hac basecall + minimap2 map-ont；主比对 MAPQ ≥ {MAPQ} 且覆盖 read ≥ {int(COVER*100)}% 才计入该物种。'
         '每格 = 跨数据集误拒 %（括号 = 同一批 read 上 R10 域内模型的误拒 %），3 个源痰液组 × 3 个 R10 测试集合并；主结果 R10 重采样到 5 kHz。', '',
         '| 物种 | read 数 | ' + ' | '.join(NAMES[m] for m in METHODS) + ' |', '|---|---:|' + '---:|'*len(METHODS)]
    R = [r for r in rows if r['variant'] == 'at5k']
    for k in sorted(tot, key=lambda k: -tot[k]):
        cell = []
        for m in METHODS:
            rs = [r for r in R if r['species'] == k and r['method'] == m]; n = sum(r['n'] for r in rs)
            cell.append(f"{100*sum(r['frr']*r['n'] for r in rs)/n:.1f} ({100*sum(r['frr_indomain']*r['n'] for r in rs)/n:.1f})" if n else '–')
        L.append(f'| {k} | {tot[k]} | ' + ' | '.join(cell) + ' |')
    L += ['', 'read 数 = 3 个 R10 测试集合计的不重复 read。', '']
    text = '\n'.join(L) + '\n'; print(text)
    if a.md: a.md.write_text(text); a.md.with_suffix('.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('assign', 'report'))
    ap.add_argument('--paf', type=Path); ap.add_argument('--fastq', type=Path); ap.add_argument('--ids', type=Path)
    ap.add_argument('--out', type=Path); ap.add_argument('--labels', type=Path); ap.add_argument('--main', type=Path)
    ap.add_argument('--cd', type=Path); ap.add_argument('--md', type=Path)
    a = ap.parse_args()
    globals()[a.stage](a)
