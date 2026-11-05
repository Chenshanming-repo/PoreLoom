#!/usr/bin/env python3
"""PoreLoom example: replay PoreLoom and three baselines on 1,000 simulated test reads.

    python example/run_example.py [--device cpu] [--out example/output]

The example is a test of the installation: it runs the paper's trained models on a small subset of the paper's
simulated test set with the paper's offline protocol (an observation every 0.4 s, 0.1 s ejection), checks every
read's outcome against the paper's, and writes metrics.json, per_read.csv, example_reads.png and README.md to --out.
Baseline architectures come from their upstream repositories; missing files are downloaded into third_party/ at
the commits used in the paper and checked by sha256.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO/'src'))

FS, INTERVAL = 4000., .4
METHODS = ('poreloom', 'readcurrent', 'nanodeep', 'squigglenet')
NAMES = dict(poreloom='PoreLoom', readcurrent='ReadCurrent', nanodeep='NanoDeep', squigglenet='SquiggleNet')
KEYS = ('command', 'command_reject', 'effective_reject', 'pore_seconds', 'decision_seconds')
UPSTREAM = {   # third_party path: (repository, commit, sha256)
    'ReadCurrent/models/ReadCurrent.py': ('Ming-Ni-Group/ReadCurrent', '6db8921fdc04236dd23a11a951a25ccc0965ae72',
                                          '892f2798bc6fea50588ec33266a0785eaf9ed44f35f48c25c7b0b0629c726fb3'),
    'NanoDeep/read_deep/model/nanodeep.py': ('lysovosyl/NanoDeep', '1abba04d72379b93a78d45d50a705cdadc4ec773',
                                             'fece5a95075613f5e868850c821bcf05e9cb80d42bc1f508c477ed4188be495d'),
    'SquiggleNet/model.py': ('welch-lab/SquiggleNet', 'fb7b206e05935de48895cdbd16a7c1df73515202',
                             'b05d129eeb19ffd59d9b7ae6959da1e1b3eb3ee72569e806df7afabff2222a73'),
    'SquiggleNet/inference.py': ('welch-lab/SquiggleNet', 'fb7b206e05935de48895cdbd16a7c1df73515202',
                                 '279e5ef9965ef3a14d66f0ebda3ecbced3686b3218565de0d8adc13a89fa0cb5'),
}
INK, INK2, MUTED, SURFACE, ACCENT = '#0b0b0b', '#52514e', '#898781', '#fcfcfb', '#2a78d6'
BAND = '#efeee9'      # every other signal chunk


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def ensure_third_party():
    from poreloom.paths import THIRD_PARTY
    for rel, (repo, commit, digest) in UPSTREAM.items():
        path = THIRD_PARTY/rel
        if not path.exists():
            url = f'https://raw.githubusercontent.com/{repo}/{commit}/{rel.split("/", 1)[1]}'
            print(f'downloading {url}')
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            if sha256(data) != digest:
                raise ValueError(f'{url} does not match the pinned sha256')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        elif sha256(path.read_bytes()) != digest:
            raise ValueError(f'{path} differs from {repo}@{commit[:7]}; the paper used that commit')


def load_data():
    from poreloom.prefix_data import load_prefix_cache
    data = load_prefix_cache(HERE/'data'/'test', verify=True)
    with np.load(HERE/'data'/'test'/'metadata.npz', allow_pickle=False) as z:
        phase = z['phase']
    return data, phase


def run_poreloom(data, phase, device):
    import torch
    from poreloom.method import head as H
    from poreloom.stream_training import load_stream_r0
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    net = load_stream_r0(HERE/'models'/'poreloom'/'encoder.pt', device)[0].eval()
    controller = H.load_controller(HERE/'models'/'poreloom'/'head.pt', 1.0, device)
    z = H.extract(net, data, FS, INTERVAL, None, device, phase=phase)
    return H.evaluate(controller, z, FS, device, service=H.HEAD['service'])


def run_baseline(tool, data, phase, device):
    from poreloom.baselines.preprocessing import CONTRACTS
    from poreloom.method import native, native_policy as P
    P.FS = FS
    selection = json.loads((HERE/'models'/tool/'selection.json').read_text())
    scores = native.score(tool, HERE/'models'/tool/'best.pt', native.prepare(tool, data), device)
    meta = dict(read_id=data.read_id, is_target=np.asarray(data.is_target, bool), raw_length=data.raw_length)
    return P.native_policy(scores, meta, threshold=None, service=selection['service_seconds'], delta=INTERVAL,
                           raw_end=CONTRACTS[tool]['raw_end'], phase=phase)


def metrics(o, is_target):
    y = np.asarray(is_target, bool)
    reject = np.asarray(o['command_reject'], bool)
    return dict(acc=float(((reject & ~y).sum() + (~reject & y).sum())/len(y)), host_reject=float(reject[~y].mean()),
                target_reject=float(reject[y].mean()), host_pore_seconds=float(np.asarray(o['pore_seconds'])[~y].mean()))


def compare(outcomes, read_id):
    with np.load(HERE/'reference'/'paper_outcomes.npz', allow_pickle=False) as z:
        ref = {k: z[k] for k in z.files}
    if not np.array_equal(ref['read_id'], read_id):
        raise ValueError('reference read order differs from the example data')
    result = {}
    for name, o in outcomes.items():
        same = np.ones(len(read_id), bool)
        for k in KEYS:
            a, b = np.asarray(o[k]), ref[f'{name}__{k}']
            same &= (a == b) if a.dtype == bool else np.isclose(a, b, rtol=0, atol=1e-6, equal_nan=True)
        result[name] = int(same.sum())
    return result


def action(o, i):
    return ('reject' if o['command_reject'][i] else 'keep') if o['command'][i] else 'none'


def per_read_rows(data, outcomes):
    rows = []
    for i, read_id in enumerate(data.read_id):
        target = bool(data.is_target[i])
        row = dict(read_id=str(read_id), true_class='target' if target else 'host', raw_samples=int(data.raw_length[i]))
        for name, o in outcomes.items():
            row[f'{name}_action'] = action(o, i)
            row[f'{name}_correct'] = bool(o['command_reject'][i]) != target
            row[f'{name}_decision_s'] = round(float(o['decision_seconds'][i]), 6) if o['command'][i] else ''
            row[f'{name}_pore_s'] = round(float(o['pore_seconds'][i]), 6)
        rows.append(row)
    return rows


def pick_examples(data, outcomes):
    """Three reads where PoreLoom does better: a host it rejects that no baseline rejects, a target it keeps that the
    most baselines reject, and a host every method rejects where PoreLoom is earliest by the widest margin."""
    y = np.asarray(data.is_target, bool)
    p = outcomes['poreloom']
    base = [outcomes[t] for t in METHODS[1:]]
    t_p = np.where(p['command'], p['decision_seconds'], np.inf)
    rejected = np.stack([np.asarray(o['command_reject'], bool) for o in base])
    picks = []
    host_only = np.flatnonzero(~y & p['effective_reject'] & ~rejected.any(0))
    if len(host_only):
        picks.append((int(host_only[np.argmin(t_p[host_only])]), 'PoreLoom rejects this host read; no baseline does'))
    kept = np.flatnonzero(y & p['command'] & ~p['command_reject'] & (rejected.sum(0) >= 2))
    if len(kept):
        best = kept[np.lexsort((t_p[kept], -rejected.sum(0)[kept]))[0]]
        picks.append((int(best), f'PoreLoom keeps this target read; {rejected.sum(0)[best]} of 3 baselines reject it'))
    t_b = np.stack([np.where(o['command'], o['decision_seconds'], np.inf) for o in base]).min(0)
    all_reject = np.flatnonzero(~y & p['effective_reject'] & np.stack([o['effective_reject'] for o in base]).all(0))
    if len(all_reject):
        best = all_reject[np.argmax(t_b[all_reject] - t_p[all_reject])]
        picks.append((int(best), f'every method rejects this host read; PoreLoom is {t_b[best] - t_p[best]:.2f} s earlier'))
    return picks


def plot(data, outcomes, picks, path, phase):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK2, 'xtick.color': INK2,
                         'ytick.color': INK2, 'text.color': INK})
    end = data.adc.shape[1]/FS
    fig = plt.figure(figsize=(9.5, 3.4*len(picks)), facecolor=SURFACE)
    outer = fig.add_gridspec(len(picks), 1, hspace=.45, top=.93)
    for row, (i, title) in enumerate(picks):
        inner = outer[row].subgridspec(2, 1, height_ratios=[1.4, 1], hspace=.08)
        ax, lanes = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
        # signal arrives in chunks ending at phase*0.4 s + k*0.4 s (the replay of every method); commands follow a chunk
        ends = (phase[i] + np.arange(int(end/INTERVAL) + 2))*INTERVAL
        edges = np.concatenate([[0.], ends[ends < end], [end]])
        for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            if k % 2 == 0:
                for a in (ax, lanes):
                    a.axvspan(lo, hi, color=BAND, lw=0, zorder=0)
            if hi - lo > .1:
                ax.text((lo + hi)/2, .97, str(k + 1), transform=ax.get_xaxis_transform(), ha='center', va='top',
                        fontsize=7.5, color=MUTED)
        n = int(min(data.raw_length[i], data.adc.shape[1]))
        t = np.arange(n)/FS
        ax.plot(t, data.pa([i], 0, n)[0], color=INK2, lw=.5)
        if outcomes['poreloom']['command'][i]:          # where PoreLoom decided
            ax.axvline(float(outcomes['poreloom']['decision_seconds'][i]), color=ACCENT, lw=1.2, ls='--')
        ax.set_xlim(0, end); ax.set_ylabel('current (pA)')
        low, high = ax.get_ylim(); ax.set_ylim(low, high + .22*(high - low))     # room for the chunk numbers
        ax.set_title(f'{"Target" if data.is_target[i] else "Host"} read {data.read_id[i]}: {title}', loc='left',
                     fontsize=9.5, color=INK)
        ax.tick_params(labelbottom=False)
        for name_row, name in enumerate(METHODS):
            o = outcomes[name]
            color = ACCENT if name == 'poreloom' else MUTED
            y = len(METHODS) - 1 - name_row
            correct = bool(o['command_reject'][i]) != bool(data.is_target[i])
            if o['command'][i]:
                when = float(o['decision_seconds'][i])
                lanes.plot([0, when], [y, y], color=color, lw=5, solid_capstyle='round')
                lanes.plot([when], [y], 'o', ms=8, color=color, markeredgecolor=SURFACE, markeredgewidth=2)
                label = f'{action(o, i).capitalize()} at {when:.2f} s ({"correct" if correct else "wrong"})'
                lanes.text(when + .04, y, label, va='center', fontsize=8.5,
                           color=INK if name == 'poreloom' else INK2, fontweight='bold' if name == 'poreloom' else None)
            else:
                lanes.text(.04, y, 'no command; the read is kept', va='center', fontsize=8.5, color=INK2)
        lanes.set_yticks(range(len(METHODS)), [NAMES[m] for m in METHODS[::-1]])
        lanes.set_ylim(-.7, len(METHODS) - .3); lanes.set_xlim(0, end)
        lanes.set_xlabel('time since read start (s)' if row == len(picks) - 1 else '')
        for a in (ax, lanes):
            a.set_facecolor(SURFACE); a.set_axisbelow(True)
            for side in ('top', 'right'):
                a.spines[side].set_visible(False)
        lanes.tick_params(axis='y', length=0)
    fig.suptitle('Shaded, numbered bands: the chunks in which the signal arrives, one every 0.4 s from a random phase '
                 '(the first is shorter).\nA method can act only after a chunk has arrived; its decision time adds '
                 'its compute time.', x=.125, ha='left', va='top', y=.985, fontsize=8.5, color=INK2, linespacing=1.4)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def percent(x):
    return f'{100*x:.2f}%'


def write_readme(path, results, paper, matches, picks, seconds, device):
    n = results['reads']
    device_name = 'a CPU' if device == 'cpu' else device
    lines = [
        '# Example: PoreLoom and baselines on 1,000 simulated reads',
        '',
        'This example is for testing. It runs the paper\'s trained PoreLoom model and three baselines on a small',
        'subset of the simulated test set and reports the metrics used in the paper, so an installation can be',
        'checked in about a minute. The results below come from the command in',
        '[Running the example](#running-the-example).',
        '',
        '## Results',
        '',
        f'{n:,} test reads ({results["targets"]} target, {n - results["targets"]} host) from simulated group 1, replayed with an observation',
        'every 0.4 s. Each baseline classifies one 3000-sample window at the position its original code uses:',
        'NanoDeep skips the first 1000 samples (samples 1000-4000), ReadCurrent and SquiggleNet skip 1500',
        '(samples 1500-4500); each uses its native probability threshold of 0.5. PoreLoom also skips the first',
        '1500 samples and then decides from the growing signal.',
        '',
        '| Method | Accuracy | Hosts rejected | Targets rejected | Pore time per host read |',
        '|---|---|---|---|---|',
    ]
    for m in METHODS:
        r = results['metrics'][m]
        lines.append(f'| {NAMES[m]} | {percent(r["acc"])} | {percent(r["host_reject"])} | '
                     f'{percent(r["target_reject"])} | {r["host_pore_seconds"]:.3f} s |')
    lines += [
        '',
        f'The same models on all {paper["reads"]:,} test reads of simulated group 1 in the paper (Table 1 averages',
        'this group with two others):',
        '',
        '| Method | Accuracy | Hosts rejected | Targets rejected | Pore time per host read |',
        '|---|---|---|---|---|',
    ]
    for m in METHODS:
        r = paper['metrics'][m]
        lines.append(f'| {NAMES[m]} | {percent(r["acc"])} | {percent(r["host_reject"])} | '
                     f'{percent(r["target_reject"])} | {r["host_pore_seconds"]:.3f} s |')
    same = ', '.join(f'{NAMES[m]} {matches[m]:,}/{n:,}' for m in METHODS)
    lines += [
        '',
        f'Per-read outcomes identical to the paper\'s: {same}.',
        '',
        '## Example reads',
        '',
        'Reads where PoreLoom does better than the baselines. Each panel shows the raw signal and, for every method,',
        'how long it observed the read before its command; the dashed line marks PoreLoom\'s decision. The',
        'shaded, numbered bands are the chunks in which the signal arrives, one every 0.4 s from a random phase',
        '(so the first chunk is shorter); a method can act only after a chunk has arrived. ReadCurrent and',
        'SquiggleNet need 4500 samples (1.125 s) and NanoDeep 4000 (1.000 s), so NanoDeep decides one chunk',
        'earlier when a chunk ends between those two times.',
        '',
        '![Example reads](example_reads.png)',
        '',
        '## Running the example',
        '',
        '```bash',
        'pip install -e . matplotlib',
        'python example/run_example.py --device cpu       # or --device cuda:0',
        '```',
        '',
        f'This run took {seconds:.0f} s on {device_name}. The script downloads four upstream baseline files into',
        '`third_party/` on first use (ReadCurrent `6db8921`, NanoDeep `1abba04`, SquiggleNet `fb7b206`, all MIT) and',
        'writes `metrics.json`, `per_read.csv`, `example_reads.png` and this `README.md` to `--out`',
        '(default `example/output/`). The files in this folder were written with `--out example`.',
        '',
        '## Contents',
        '',
        '| Path | Content |',
        '|---|---|',
        f'| `data/test/` | {n:,} reads in the prefix-cache layout of `poreloom.method.data`: int16 ADC of the first 9000 samples, calibration, class, and each read\'s replay phase in the paper (`phase`) |',
        '| `models/poreloom/` | streaming encoder (`encoder.pt`) and FACP head (`head.pt`, beta = 1) |',
        '| `models/readcurrent/`, `models/nanodeep/`, `models/squigglenet/` | 3000-sample baselines (`best.pt`) and their selection files with the service time used in replay |',
        '| `reference/` | the paper\'s per-read outcomes for these reads and its metrics on all group-1 test reads |',
        '| `run_example.py` | the script that produced this page |',
        '',
        'The simulated reads were generated with Squigulator (`dna-r10-min`, 4 kHz) from GRCh38.p14 (host) and eight',
        'bacterial references (target). `data/test/manifest.json` and `models/manifest.json` record the selection',
        'and the source files.',
        '',
    ]
    path.write_text('\n'.join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', type=Path, default=HERE/'output')
    a = ap.parse_args(argv)
    import torch
    if a.device.startswith('cuda'):
        torch.cuda.set_device(a.device)
    start = time.time()
    ensure_third_party()
    data, phase = load_data()
    outcomes = {'poreloom': run_poreloom(data, phase, a.device)}
    for tool in METHODS[1:]:
        outcomes[tool] = run_baseline(tool, data, phase, a.device)
    seconds = time.time() - start
    results = dict(reads=len(data), targets=int(np.sum(data.is_target)), device=a.device,
                   metrics={m: metrics(o, data.is_target) for m, o in outcomes.items()})
    matches = compare(outcomes, data.read_id)
    results['identical_to_paper'] = matches
    paper = json.loads((HERE/'reference'/'paper_metrics.json').read_text())
    picks = pick_examples(data, outcomes)
    results['example_reads'] = [dict(read_id=str(data.read_id[i]), why=why) for i, why in picks]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/'metrics.json').write_text(json.dumps(results, indent=2) + '\n')
    rows = per_read_rows(data, outcomes)
    with open(a.out/'per_read.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    plot(data, outcomes, picks, a.out/'example_reads.png', phase)
    write_readme(a.out/'README.md', results, paper, matches, picks, seconds, a.device)
    print(f'{"method":<12} {"accuracy":>9} {"hosts rej.":>11} {"targets rej.":>13} {"host pore s":>12}  same as paper')
    for m in METHODS:
        r = results['metrics'][m]
        print(f'{NAMES[m]:<12} {percent(r["acc"]):>9} {percent(r["host_reject"]):>11} {percent(r["target_reject"]):>13} '
              f'{r["host_pore_seconds"]:>12.3f}  {matches[m]}/{len(data)}')
    print(f'wrote {a.out} in {seconds:.0f} s')


if __name__ == '__main__':
    main()
