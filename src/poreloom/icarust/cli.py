"""Closed-loop simulation of selective sequencing with Icarust.

    python -m poreloom.icarust setup --icarust DIR                build-side files: TLS certificate, VBZ plugin
    python -m poreloom.icarust references --fasta F ... --out DIR/references
    python -m poreloom.icarust check --models M.json --data DATASET [--device cuda:0]
    python -m poreloom.icarust run --strategy S --model PATH --icarust DIR --out RUN --seed N --target-permille P --port Q
    python -m poreloom.icarust matrix --models M.json --icarust DIR --out OUT --lanes cuda:0:25100 [...]
    python -m poreloom.icarust aggregate OUT
    python -m poreloom.icarust analyze RUN [RUN ...]

Strategies: No-AS, PoreLoom, readcurrent, nanodeep, squigglenet (3000-sample windows) and readcurrent_w6000,
nanodeep_w6000, squigglenet_w6000 (6000-sample windows). See the repository README for building Icarust.
"""
import argparse
import json
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

SEEDS = (20331101, 20331102, 20331103)
PERMILLE = (10, 100, 300)
CHANNELS, DURATION, FOLLOW = 512, 1200, 20


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n')


def icarust_args(ap):
    g = ap.add_argument_group('Icarust')
    g.add_argument('--icarust', type=Path, required=True, metavar='DIR', help='Icarust directory (see poreloom.icarust.simulator)')
    g.add_argument('--icarust-src', type=Path, help='patched source tree (default DIR/src_tree)')
    g.add_argument('--icarust-bin', type=Path, help='release binary (default DIR/target/release/icarust)')
    g.add_argument('--ld-library-path', help="Icarust's shared libraries (HDF5, OpenSSL, zstd); default: $LD_LIBRARY_PATH")
    g.add_argument('--tmp', type=Path, help='simulator output, deleted after each run except the audit log (default /dev/shm/poreloom_icarust)')


def locate(a):
    from .simulator import Icarust
    return Icarust.locate(a.icarust, a.icarust_src, a.icarust_bin, a.ld_library_path, a.tmp).check()


def run_args(ap):
    ap.add_argument('--channels', type=int, default=CHANNELS)
    ap.add_argument('--duration', type=int, default=DURATION, help='cohort window T in seconds (default %(default)s)')
    ap.add_argument('--follow', type=int, default=FOLLOW, help='seconds of streaming after T for decisions in flight (default %(default)s)')


# ------------------------------------------------------------------ setup / references
def cmd_setup(a):
    from .simulator import setup
    print(json.dumps(setup(a.icarust, a.icarust_src), indent=2))


def cmd_references(a):
    from .references import build
    manifest = build(a.fasta, a.out, check=not a.no_check)
    print(json.dumps(manifest['samples'], indent=2))


# ------------------------------------------------------------------ check
def cmd_check(a):
    from .check import check
    from .strategies import STRATEGIES, models_config
    models = models_config(a.models)
    names = a.strategies or [s for s in STRATEGIES if s in models]
    import tempfile
    import torch
    torch.set_num_threads(4)
    def show(name, r):
        print(name, json.dumps({k: v for k, v in r.items() if k != 'pins'}), flush=True)
    with tempfile.TemporaryDirectory() as work:
        result = check(models, a.data, names, a.device, a.n, a.fs, work, log=show)
    if a.out:
        write_json(a.out, dict(data=str(a.data), n=a.n, results=result))


# ------------------------------------------------------------------ one run
def cmd_run(a):
    import torch
    from .analysis import summarize_run
    from .client import Client, server_calibration
    from .simulator import EXTRA_SECONDS, PAPER_SHA256, SAMPLE_RATE
    from .strategies import MODEL_CALIBRATION, build, models_config
    from ..data import file_sha256
    model = a.model or (models_config(a.models).get(a.strategy) if a.models else None)
    ic = locate(a)
    a.out = a.out.resolve()
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    sha = file_sha256(ic.binary)
    references = ic.references/'manifest.json'
    manifest = dict(strategy=a.strategy, model=model, seed=a.seed, channels=a.channels, duration=a.duration, follow=a.follow,
                    target_permille=a.target_permille, device=a.device, port=a.port,
                    icarust=dict(binary=str(ic.binary), source=str(ic.source), sha256=sha, paper_binary=sha == PAPER_SHA256),
                    references=dict(path=str(references), sha256=file_sha256(references)) if references.exists() else None,
                    started_wall=time.time())
    torch.set_num_threads(4)
    if a.device.startswith('cuda'):
        torch.cuda.set_device(a.device)
    server, sim_out = ic.start(a.out, a.seed, a.duration + a.follow + EXTRA_SECONDS, a.target_permille, a.port, a.channels)
    manifest['server_pid'] = server.pid
    try:
        certificate = ic.tls/'server.crt'
        calibration = server_calibration(a.port + 1, certificate)
        strategy = build(a.strategy, model, a.device, MODEL_CALIBRATION, a.out)
        fs = getattr(strategy, 'fs', None)
        if fs is not None and abs(fs - SAMPLE_RATE) > 1e-6:
            raise ValueError(f'the simulator samples at {SAMPLE_RATE} Hz but the model was trained for {fs:g} Hz')
        manifest.update(calibration=calibration, model_input_calibration=MODEL_CALIBRATION,
                        strategy_pins=getattr(strategy, 'pins', None))
        client = Client(a.out, strategy, a.channels, a.duration, a.port + 1, a.follow, calibration, MODEL_CALIBRATION, certificate)
        manifest.update(client.run(), stats=client.stats, policy_stats=getattr(strategy, 'stats', None), status='complete')
    except BaseException as exc:
        manifest['status'] = f'failed: {type(exc).__name__}: {exc}'
        raise
    finally:
        manifest.update(ic.stop(server, sim_out, a.out))
        manifest['finished_wall'] = time.time()
        write_json(a.out/'manifest.json', manifest)
    s = summarize_run(a.out)
    print(json.dumps({k: s[k] for k in ('strategy', 'seed', 'target_permille', 'reads', 'target_reads',
                                        'yield_target_bp_per_channel_hour', 'accuracy')}), flush=True)


def cmd_analyze(a):
    from .analysis import summarize_run
    for run in a.runs:
        s = summarize_run(run)
        print(run, json.dumps({k: s[k] for k in ('strategy', 'seed', 'target_permille', 'reads', 'yield_target_bp_per_channel_hour')}))


# ------------------------------------------------------------------ matrix
def cells(strategies, seeds, permille, lanes):
    """Every (seed, target fraction, strategy); order shuffled within each seed block, lanes assigned round-robin."""
    out = []
    for seed in seeds:
        block = [dict(seed=seed, permille=p, strategy=s) for p in permille for s in strategies]
        random.Random(seed).shuffle(block); out += block
    for i, c in enumerate(out):
        c.update(lane=i % lanes, name=f"{c['permille']/10:g}pct_{c['seed']}_{c['strategy']}")
    return out


def parse_lane(text):
    device, _, port = text.rpartition(':')
    if not device or not port.isdigit() or not 1024 <= int(port) < 65535:
        raise argparse.ArgumentTypeError(f'lane {text!r}: expected DEVICE:PORT with PORT in 1024-65534, e.g. cuda:0:25100')
    return device, int(port)


def cmd_matrix(a):
    from .strategies import STRATEGIES, models_config
    models = models_config(a.models)
    strategies = a.strategies or ['No-AS'] + [s for s in STRATEGIES if s in models]
    missing = [s for s in strategies if s != 'No-AS' and s not in models]
    if missing:
        raise SystemExit(f'no model in {a.models} for {missing}')
    if 'No-AS' not in strategies:
        log('note: without No-AS runs enrichment cannot be computed for these runs')
    locate(a)
    a.out = a.out.resolve()
    ports = [p for _, p in a.lanes]
    if len(set(ports)) != len(ports) or any(abs(p - q) < 2 for i, p in enumerate(ports) for q in ports[i + 1:]):
        raise SystemExit('each lane needs its own pair of ports (PORT and PORT + 1)')
    live = a.out/'live'; live.mkdir(parents=True, exist_ok=True)
    plan = cells(strategies, a.seeds, a.target_permille, len(a.lanes))
    settings = dict(channels=a.channels, duration=a.duration, follow=a.follow, max_latency_p50=a.max_latency_p50)
    if (a.out/'plan.json').exists():
        old = json.loads((a.out/'plan.json').read_text())
        if old['settings'] != settings or old['models'] != models:
            raise SystemExit(f'{a.out} was started with other settings or models; use a new --out')
    else:
        write_json(a.out/'plan.json', dict(written=time.strftime('%Y-%m-%d %H:%M:%S'), settings=settings, models=models,
                                           lanes=a.lanes, cells=plan))
    base = [sys.executable, '-m', 'poreloom.icarust', 'run', '--icarust', str(a.icarust), '--channels', str(a.channels),
            '--duration', str(a.duration), '--follow', str(a.follow)]
    for flag, value in (('--icarust-src', a.icarust_src), ('--icarust-bin', a.icarust_bin),
                        ('--ld-library-path', a.ld_library_path), ('--tmp', a.tmp)):
        if value is not None: base += [flag, str(value)]

    def accepted(out):
        if not (out/'summary.json').exists(): return False
        if a.max_latency_p50 is None: return True
        p50 = (json.loads((out/'summary.json').read_text()).get('command_latency_s') or {}).get('p50')
        return p50 is None or p50 <= a.max_latency_p50

    def lane(i):
        device, port = a.lanes[i]
        time.sleep(a.stagger*i)
        for c in plan:
            if c['lane'] != i: continue
            out = live/c['name']
            for attempt in range(a.attempts):
                if accepted(out): break
                if out.exists():
                    tag = 'slow' if (out/'summary.json').exists() else 'failed'
                    out.rename(out.with_name(f'{out.name}.{tag}_{int(time.time())}'))
                cmd = base + ['--strategy', c['strategy'], '--out', str(out), '--seed', str(c['seed']),
                              '--target-permille', str(c['permille']), '--port', str(port), '--device', device]
                if c['strategy'] != 'No-AS': cmd += ['--model', models[c['strategy']]]
                log('start', c['name'], device, 'attempt', attempt + 1)
                with open(live/f"{c['name']}.log", 'a') as f:
                    code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
                log('done', c['name'], 'exit', code, 'accepted' if accepted(out) else 'not accepted')
                time.sleep(a.pause)
            if not accepted(out):
                log('giving up', c['name'], f'after {a.attempts} attempts')
    threads = [threading.Thread(target=lane, args=(i,)) for i in range(len(a.lanes))]
    for t in threads: t.start()
    for t in threads: t.join()
    done = [c['name'] for c in plan if accepted(live/c['name'])]
    log(f'{len(done)}/{len(plan)} runs accepted')
    if done:
        aggregate_dir(a.out, strategies, done)


# ------------------------------------------------------------------ aggregate
RETIRED = re.compile(r'\.(failed|slow)_\d+$')     # runs replaced by a later attempt


def aggregate_dir(out, strategies=None, names=None):
    """names: run directories to include (default: every run directory except replaced attempts)."""
    from .analysis import aggregate, markdown
    from .strategies import STRATEGIES, label
    out = Path(out)
    root = out/'live' if (out/'live').is_dir() else out
    summaries = [json.loads(p.read_text()) for p in sorted(root.glob('*/summary.json'))
                 if (names is None and not RETIRED.search(p.parent.name)) or (names is not None and p.parent.name in names)]
    if not summaries:
        raise SystemExit(f'no run summaries under {root}')
    order = [s for s in strategies or STRATEGIES if any(r['strategy'] == s for r in summaries)]
    rows, table = aggregate(summaries, order)
    write_json(out/'aggregate.json', dict(runs=rows, table=[dict(target_permille=p, strategy=s, **q) for (p, s), q in table.items()]))
    text = markdown(table, label)
    (out/'aggregate.md').write_text(text)
    print(text)


def cmd_aggregate(a):
    aggregate_dir(a.out, a.strategies)


# ------------------------------------------------------------------ main
def main(argv=None):
    from .strategies import STRATEGIES
    ap = argparse.ArgumentParser(prog='python -m poreloom.icarust', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)

    p = sub.add_parser('setup', help='TLS certificate and VBZ plugin next to a built Icarust')
    p.add_argument('--icarust', type=Path, required=True, metavar='DIR')
    p.add_argument('--icarust-src', type=Path, help='patched source tree (default DIR/src_tree)')
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser('references', help='host.fa and target.fa from the public genomes')
    p.add_argument('--fasta', type=Path, nargs='+', required=True,
                   help='GRCh38.p14 genomic FASTA (.fna.gz) and the eight bacterial FASTAs listed in reference_blocks.json')
    p.add_argument('--out', type=Path, required=True, help='new directory, normally DIR/references')
    p.add_argument('--no-check', action='store_true', help='do not require the SHA-256 of the paper references')
    p.set_defaults(func=cmd_references)

    p = sub.add_parser('check', help='live strategies against their offline replay on test reads')
    p.add_argument('--models', type=Path, required=True, help='JSON {strategy: model path}')
    p.add_argument('--data', type=Path, required=True, help='dataset root with test/ (the models\' training dataset)')
    p.add_argument('--strategies', nargs='+', choices=STRATEGIES, help='default: every strategy in --models')
    p.add_argument('--n', type=int, default=3000, help='test reads (default %(default)s)')
    p.add_argument('--fs', type=float, default=4000.)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', type=Path, help='write the result as JSON')
    p.set_defaults(func=cmd_check)

    p = sub.add_parser('run', help='one closed-loop run')
    p.add_argument('--strategy', choices=STRATEGIES, required=True)
    p.add_argument('--model', help='trained model of the strategy (PoreLoom: poreloom_sh_cp.pt or its run directory; '
                                   'baselines: run directory or best.pt)')
    p.add_argument('--models', type=Path, help='JSON {strategy: model path}, used when --model is not given')
    p.add_argument('--out', type=Path, required=True, help='new run directory')
    p.add_argument('--seed', type=int, required=True, help='simulator seed')
    p.add_argument('--target-permille', type=int, required=True, help='target molecule weight out of 1000 (10 = 1%%)')
    p.add_argument('--port', type=int, required=True, help='manager port; the position uses PORT + 1')
    p.add_argument('--device', default='cuda:0')
    run_args(p); icarust_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser('matrix', help='seeds x target fractions x strategies on parallel lanes')
    p.add_argument('--models', type=Path, required=True, help='JSON {strategy: model path}')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lanes', type=parse_lane, nargs='+', required=True, metavar='DEVICE:PORT',
                   help='one simulator + client at a time per lane, e.g. cuda:0:25100 cuda:1:25200')
    p.add_argument('--strategies', nargs='+', choices=STRATEGIES, help='default: No-AS and every strategy in --models')
    p.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    p.add_argument('--target-permille', type=int, nargs='+', default=list(PERMILLE))
    p.add_argument('--attempts', type=int, default=3, help='attempts per run (default %(default)s)')
    p.add_argument('--max-latency-p50', type=float,
                   help='accept a run only if its median command latency (s) is at most this; otherwise rerun it')
    p.add_argument('--stagger', type=float, default=60., help='seconds between lane starts (default %(default)s)')
    p.add_argument('--pause', type=float, default=15., help='seconds between runs on a lane (default %(default)s)')
    run_args(p); icarust_args(p)
    p.set_defaults(func=cmd_matrix)

    p = sub.add_parser('aggregate', help='enrichment and run metrics over a matrix output')
    p.add_argument('out', type=Path, help='matrix --out directory (or a directory of run directories)')
    p.add_argument('--strategies', nargs='+', choices=STRATEGIES, help='row order (default: all present)')
    p.set_defaults(func=cmd_aggregate)

    p = sub.add_parser('analyze', help='recompute summary.json of finished runs')
    p.add_argument('runs', type=Path, nargs='+')
    p.set_defaults(func=cmd_analyze)

    a = ap.parse_args(argv)
    a.func(a)
