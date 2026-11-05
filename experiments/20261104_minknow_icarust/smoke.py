#!/usr/bin/env python3
"""Smoke test of the MinKNOW adapter (python -m poreloom.minknow) against Icarust (4 November 2026).

Starts the simulator exactly as the closed-loop matrix did (experiments/20261011_simloop/closed_loop.py: R10 4 kHz
profile with the TEST-role references, 512 channels, break_read_ms 400), then runs the adapter as a separate
process the way it would run next to MinKNOW: manager service -> flow cell position -> per-channel calibration ->
get_live_reads, TLS trusted through MINKNOW_TRUSTED_CA. Afterwards the simulator's audit log (truth) is joined
with the adapter's logs by analyze_live.py. A short run checks the deployment path; it is not a paper result.

    smoke.py --out OUT --bundle RUN/poreloom_sh_cp.pt --icarust-root ROOT --ld-library-path LD \\
             [--icarust-src SRC --icarust-bin BIN] [--device cuda:0] [--port 26900] [--seed 20331101] \\
             [--duration 300] [--target-permille 10]

ROOT holds the simloop simulator support: tls/ (ca.crt, server.crt, server.key), vbz/ and references/ (the layout
under $PORELOOM_RESULTS/poreloom_simloop_20261011/icarust). SRC is upstream Icarust with
experiments/20261011_simloop/icarust.patch applied and BIN its release build; they default to the paper build in
ROOT/icarust_writerfix. A rebuild has a different hash than the paper binary (build paths are compiled in);
smoke.json records the hash and whether it is the paper binary.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SIMLOOP = REPO/'experiments/20261011_simloop'
_spec = importlib.util.spec_from_file_location('simloop_closed_loop', SIMLOOP/'closed_loop.py')
cl = importlib.util.module_from_spec(_spec); sys.modules[_spec.name] = cl; _spec.loader.exec_module(cl)


def wait_for_port(port, ca, timeout=600):
    """Icarust serves the manager at once but the position only after converting its references."""
    import grpc
    cred = grpc.ssl_channel_credentials(root_certificates=Path(ca).read_bytes())
    ch = grpc.secure_channel(f'localhost:{port}', cred)
    grpc.channel_ready_future(ch).result(timeout=timeout)
    ch.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--encoder', type=Path)
    ap.add_argument('--icarust-root', type=Path, required=True)
    ap.add_argument('--icarust-src', type=Path, help='patched Icarust source tree (default ROOT/icarust_writerfix/src_tree)')
    ap.add_argument('--icarust-bin', type=Path, help='its release binary (default ROOT/icarust_writerfix/target/release/icarust)')
    ap.add_argument('--ld-library-path', required=True, help='Icarust native libraries (HDF5, VBZ, zstd)')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--port', type=int, default=26900)
    ap.add_argument('--seed', type=int, default=20331101)
    ap.add_argument('--channels', type=int, default=512)
    ap.add_argument('--duration', type=int, default=300, help='cohort window T, seconds')
    ap.add_argument('--follow', type=int, default=20, help='seconds of streaming after T')
    ap.add_argument('--target-permille', type=int, default=10)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    root = a.icarust_root.resolve()
    src = (a.icarust_src or root/'icarust_writerfix/src_tree').resolve()
    binary = (a.icarust_bin or root/'icarust_writerfix/target/release/icarust').resolve()
    tls = root/'tls'
    for path in (src/'static', src/'src/e7_audit.rs', binary):
        if not path.exists():
            raise FileNotFoundError(f'{path} (build Icarust with experiments/20261011_simloop/icarust.patch)')
    sha = cl.pin(binary)['sha256']
    cl.REFS = root/'references'
    run = a.out/'run'
    run.mkdir(parents=True)
    sim_out = cl.SHM/f'minknow_smoke__{a.out.name}'
    if sim_out.exists():
        shutil.rmtree(sim_out)
    (a.out/'profile.toml').write_text(cl.profile_text(sim_out, a.seed, a.duration + a.follow + 120, a.target_permille))
    (a.out/'config.ini').write_text(f'[TLS]\ncert-dir = {tls}\n[PORTS]\nmanager = {a.port}\nposition = {a.port + 1}\n'
                                    f'[SEQUENCER]\nchannels = {a.channels}\n')
    env = dict(os.environ, LD_LIBRARY_PATH=a.ld_library_path, HDF5_PLUGIN_PATH=str(root/'vbz'))
    server = subprocess.Popen([str(binary), '-s', str(a.out/'profile.toml'), '-c', str(a.out/'config.ini'), '-v', '-p'],
                              cwd=src, env=env, stdout=open(a.out/'server.log', 'wb'),
                              stderr=subprocess.STDOUT, start_new_session=True)
    smoke = dict(seed=a.seed, channels=a.channels, cohort_duration=a.duration, follow=a.follow,
                 target_permille=a.target_permille,
                 icarust=dict(binary=str(binary), source=str(src), sha256=sha, paper_build=sha == cl.ICARUST_SHA),
                 references=cl.pin(cl.REFS/'manifest.json'), driver=cl.pin(__file__), server_pid=server.pid)
    code = None
    try:
        wait_for_port(a.port, tls/'ca.crt'); wait_for_port(a.port + 1, tls/'ca.crt')
        cmd = [sys.executable, '-m', 'poreloom.minknow', '--host', 'localhost', '--port', str(a.port),
               '--bundle', str(a.bundle), '--out', str(a.out/'client'), '--device', a.device,
               '--duration', str(a.duration + a.follow), '--stall-seconds', '10']
        if a.encoder: cmd += ['--encoder', str(a.encoder)]
        if a.dry_run: cmd += ['--dry-run']
        client_env = dict(os.environ, MINKNOW_TRUSTED_CA=str(tls/'ca.crt'),
                          PYTHONPATH=os.pathsep.join([str(REPO/'src'), os.environ.get('PYTHONPATH', '')]))
        smoke['client_cmd'] = cmd
        with open(a.out/'client.log', 'wb') as log:
            code = subprocess.call(cmd, env=client_env, stdout=log, stderr=subprocess.STDOUT)
        smoke['client_exit'] = code
    finally:
        try:
            os.killpg(server.pid, signal.SIGINT); server.wait(timeout=30)
        except Exception:
            os.killpg(server.pid, signal.SIGKILL)
        smoke['server_exit'] = server.returncode
        for audit in sim_out.glob('**/e7_engineering_audit.jsonl'):
            dest = run/'sim'/audit.relative_to(sim_out)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(audit, dest)
        shutil.rmtree(sim_out, ignore_errors=True)
        cl.write(a.out/'smoke.json', smoke)
    if code != 0:
        raise SystemExit(f'adapter exited with {code}; see {a.out/"client.log"}')
    # analyze_live.py layout: manifest (cohort T = --duration), decisions, acks and the simulator audit in one directory
    client = json.loads((a.out/'client'/'manifest.json').read_text())
    cl.write(run/'manifest.json', dict(client, strategy='PoreLoom (poreloom.minknow)', seed=a.seed, channels=a.channels,
                                       duration=a.duration, stats=client.get('stats')))
    for name in ('decisions.jsonl', 'acks.jsonl'):
        shutil.copy2(a.out/'client'/name, run/name)
    subprocess.check_call([sys.executable, str(SIMLOOP/'analyze_live.py'), str(run)])


if __name__ == '__main__':
    main()
