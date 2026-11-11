"""The Icarust simulator of the closed-loop runs: file layout, run profile, start and stop.

An Icarust directory (``--icarust``) holds the patched simulator and its run-time files:

    src_tree/                     upstream Icarust ef05586 with icarust.patch applied (the simulator's working directory)
    target/release/icarust        its release build
    tls/server.crt, server.key    certificate for localhost / 127.0.0.1 (``python -m poreloom.icarust setup``)
    vbz/                          the HDF5 VBZ plugin from src_tree/vbz_plugin/ (``setup``)
    references/host.fa, target.fa reference FASTAs (``python -m poreloom.icarust references``)

The layout of the paper runs (``icarust_writerfix/src_tree`` and ``icarust_writerfix/target``) is also recognized.
Every run simulates an R10 flow cell at 4 kHz and 400 bases/s with two samples, host and target, whose weights set
the target molecule fraction; the run's seed fixes the simulated reads. Icarust writes its output (and the truth
audit log) to tmpfs; only the audit log is kept.
"""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import os
import shutil
import signal
import subprocess

UPSTREAM = 'https://github.com/LooseLab/Icarust'
UPSTREAM_COMMIT = 'ef055864183ca2f149b748d29c34fe4ec1e74cac'
PATCH = Path(__file__).with_name('icarust.patch')
PAPER_SHA256 = '901d029ecb2f1d3c108eaad0fa77f36e7fa0b07b04a1c9de37b0e02014d71152'   # binary of the paper runs
SAMPLE_RATE = 4000
EXTRA_SECONDS = 120             # simulated experiment length beyond the client's streaming time


def profile_text(sim_out, seed, seconds, target_permille, references):
    """Icarust run profile; target_permille = target sample weight out of 1000 (10 = 1%)."""
    references = Path(references)
    return f'''output_path = "{sim_out}"
random_seed = {seed}
target_yield = 1e15
working_pore_percent = 100
pore_type = "R10"
nucleotide_type = "DNA"
[parameters]
sample_name = "poreloom"
experiment_name = "simloop"
flowcell_name = "SIM"
device_id = "SIM"
position = "MINION"
sample_rate = {SAMPLE_RATE}
sequencing_speed = 400
break_read_ms = 400
experiment_duration_set = {seconds}
[[sample]]
name = "host"
input_genome = "{references/'host.fa'}"
mean_read_length = 8000
weight = {1000 - target_permille}
[[sample]]
name = "target"
input_genome = "{references/'target.fa'}"
mean_read_length = 8000
weight = {target_permille}
'''


def config_text(tls, port, channels):
    """Icarust config: manager service on `port`, the sequencing position on `port + 1`."""
    return f'[TLS]\ncert-dir = {tls}\n[PORTS]\nmanager = {port}\nposition = {port + 1}\n[SEQUENCER]\nchannels = {channels}\n'


@dataclass
class Icarust:
    root: Path
    source: Path
    binary: Path
    ld_library_path: str = ''
    tmp: Path = field(default_factory=lambda: Path('/dev/shm/poreloom_icarust'))

    @classmethod
    def locate(cls, root, source=None, binary=None, ld_library_path=None, tmp=None):
        root = Path(root).resolve()
        build = root if (root/'src_tree').exists() or not (root/'icarust_writerfix').exists() else root/'icarust_writerfix'
        return cls(root, Path(source or build/'src_tree').resolve(), Path(binary or build/'target/release/icarust').resolve(),
                   os.environ.get('LD_LIBRARY_PATH', '') if ld_library_path is None else ld_library_path,
                   Path(tmp) if tmp else Path('/dev/shm/poreloom_icarust'))

    @property
    def tls(self):
        return self.root/'tls'

    @property
    def vbz(self):
        return self.root/'vbz'

    @property
    def references(self):
        return self.root/'references'

    def check(self):
        need = {self.source/'src/e7_audit.rs': 'patched Icarust source (git apply icarust.patch)',
                self.source/'static': 'Icarust source tree', self.binary: 'Icarust release build',
                self.tls/'server.crt': 'python -m poreloom.icarust setup', self.tls/'server.key': 'python -m poreloom.icarust setup',
                self.references/'host.fa': 'python -m poreloom.icarust references',
                self.references/'target.fa': 'python -m poreloom.icarust references'}
        missing = [f'{p} ({hint})' for p, hint in need.items() if not p.exists()]
        if not any(self.vbz.glob('libvbz_hdf_plugin*')):
            missing.append(f'{self.vbz}/libvbz_hdf_plugin* (python -m poreloom.icarust setup)')
        if missing:
            raise FileNotFoundError('Icarust directory is incomplete:\n  ' + '\n  '.join(missing))
        if certificate_expired(self.tls/'server.crt'):
            raise ValueError(f'{self.tls}/server.crt has expired; python -m poreloom.icarust setup replaces it')
        return self

    def start(self, out, seed, seconds, target_permille, port, channels):
        """Write profile.toml and config.ini to `out` and start Icarust; returns (process, simulator output dir)."""
        out = Path(out).resolve()                                # Icarust runs in its source tree
        tag = hashlib.sha256(str(out).encode()).hexdigest()[:12]   # run names repeat across matrices
        sim_out = self.tmp/f'{out.name}__{tag}'
        if sim_out.exists():
            shutil.rmtree(sim_out)
        sim_out.parent.mkdir(parents=True, exist_ok=True)
        (out/'profile.toml').write_text(profile_text(sim_out, seed, seconds, target_permille, self.references))
        (out/'config.ini').write_text(config_text(self.tls, port, channels))
        env = dict(os.environ, HDF5_PLUGIN_PATH=str(self.vbz))
        if self.ld_library_path:
            env['LD_LIBRARY_PATH'] = self.ld_library_path
        process = subprocess.Popen([str(self.binary), '-s', str(out/'profile.toml'), '-c', str(out/'config.ini'), '-v', '-p'],
                                   cwd=self.source, env=env, stdout=open(out/'server.log', 'wb'), stderr=subprocess.STDOUT,
                                   start_new_session=True)
        return process, sim_out

    @staticmethod
    def stop(process, sim_out, out):
        """Stop Icarust (Ctrl-C drains its writer), keep the audit log under out/sim/, delete the rest."""
        try:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=30)
        except Exception:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        kept = []
        for audit in Path(sim_out).glob('**/e7_engineering_audit.jsonl'):
            dest = Path(out)/'sim'/audit.relative_to(sim_out)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(audit, dest); kept.append(str(dest))
        discarded = sum(f.stat().st_size for f in Path(sim_out).glob('**/*.pod5'))
        shutil.rmtree(sim_out, ignore_errors=True)
        return dict(server_exit=process.returncode, simulator_output=dict(tmpfs=str(sim_out), kept=kept,
                                                                          raw_bytes_discarded=discarded))


def certificate_expired(path):
    """True when the certificate's validity has ended, None when it cannot be read."""
    import ssl
    import time
    try:
        not_after = ssl._ssl._test_decode_cert(str(path))['notAfter']
    except Exception:
        return None
    return ssl.cert_time_to_seconds(not_after) < time.time()


def make_tls(directory, days=3650):
    """Self-signed certificate for localhost / 127.0.0.1 (server.crt, server.key; ca.crt is the same certificate),
    kept when one exists and is still valid. Returns 'created', 'replaced' (expired) or 'kept'."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = 'created'
    if (directory/'server.crt').exists():
        if not certificate_expired(directory/'server.crt'):
            return 'kept'
        state = 'replaced'
    (directory/'ext.cnf').write_text('[req]\ndistinguished_name=dn\nx509_extensions=server\nprompt=no\n[dn]\nCN=localhost\n'
                                     '[server]\nbasicConstraints=critical,CA:TRUE\n'
                                     'keyUsage=critical,digitalSignature,keyEncipherment,keyCertSign\n'
                                     'extendedKeyUsage=serverAuth\nsubjectAltName=DNS:localhost,IP:127.0.0.1\n')
    done = subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', str(days), '-config', 'ext.cnf',
                           '-keyout', 'server.key', '-out', 'server.crt'], cwd=directory, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f'openssl failed: {done.stderr.strip()}')
    shutil.copy2(directory/'server.crt', directory/'ca.crt')
    return state


def setup(root, source=None):
    """Create root/tls and root/vbz for a built Icarust under root; returns what was done."""
    import platform
    ic = Icarust.locate(root, source)
    done = dict(tls=make_tls(ic.tls))
    name = 'libvbz_hdf_plugin.dylib' if platform.system() == 'Darwin' else f'libvbz_hdf_plugin_{platform.machine()}.so'
    plugin = ic.source/'vbz_plugin'/name
    if not plugin.is_file():
        raise FileNotFoundError(f'{plugin} not found; is {ic.source} the Icarust source tree?')
    ic.vbz.mkdir(parents=True, exist_ok=True)
    if not (ic.vbz/name).exists():
        shutil.copy2(plugin, ic.vbz/name)
    done['vbz'] = str(ic.vbz/name)
    return done
