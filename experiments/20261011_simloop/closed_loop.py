#!/usr/bin/env python3
"""P0/P3: one live Icarust run controlled by one frozen strategy.

The client subscribes to every channel, keeps a per-(channel, read_number) ADC buffer
indexed by sample coordinates, and sends decisions as soon as the frozen strategy makes
one. Inference runs in its own thread and batches every read whose next window has
arrived, so network receipt never waits for the GPU.

Strategies:
  No-AS                      accept every read (stop_further_data after the capture prefix)
  ReadCurrent / NanoDeep /   native single-window argmax (same archived fits as offline)
  CAFES / SquiggleNet
  <baseline>-2W-AND / -gate  two consecutive native windows (twowin.py; same fits)
  GRU-threshold              prefix CNN+GRU with the frozen DEV rule
  PoreLoom-A / PoreLoom-B    frozen action head (accept / reject / defer, K=4)
  GANBase@4500 / @7500       special baseline: Bonito basecall of the causal prefix [0,end) in pA, then the
                             public human GANBase classifier on the last 200 bp; reject if p(human) > 0.9
                             (frozen setup of experiments/20261011_ganbase)

Truth comes only from the simulator's audit log (sample name, length, end reason) and is
joined after the run by analyze_live.py; the client never reads it.
"""
from pathlib import Path
import argparse
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SE, OUT, REGISTRATION, PREFIX, PARENT, BASELINES, setup_imports, pin, write, read

# Reviewed e7 startupfix source (a74c513e...) rebuilt with one change: writer flush batch 4000 -> 1000 reads
# (64 -> 16 MB), see icarust/icarust_writerfix/writer_batch.patch. The original batch left ~96 reads of
# retention headroom and aborted runs whose reads finish faster than the writer drains them.
ICARUST_SRC = OUT / 'icarust/icarust_writerfix/src_tree'
ICARUST_BIN = OUT / 'icarust/icarust_writerfix/target/release/icarust'
ICARUST_SHA = '901d029ecb2f1d3c108eaad0fa77f36e7fa0b07b04a1c9de37b0e02014d71152'
LD_PATH = f"{SE}/external/.icarust-native/lib:/path/to/conda/envs/poreloom/lib"
TLS = OUT / 'icarust/tls'
SHM = Path('/dev/shm/poreloom_simloop')
VBZ = OUT / 'icarust/vbz'           # HDF5 VBZ filter for Icarust's FAST5 writer
REFS = OUT / 'icarust/references'
# Training-domain ADC contract (Squigulator dna-r10-min): digitisation 8192, range 1536.598389 pA,
# offset = mean of model_DEV per-read offsets (13.63; per-read std 16.3). Icarust packets are mapped
# ADC -> pA (Icarust calibration) -> training ADC, identically for every strategy.
TRAIN_CAL = dict(offset=13.627454106396664, range=1536.598389, digitisation=8192.0)
CAPTURE = 7500                      # samples kept per read for offline cross-simulator replay
START, WINDOW, K = 1500, 1500, 4
NATIVE = {v['id']: k for k, v in BASELINES.items()}
POLICY_FAMILY = {'PoreLoom-A': 'A', 'PoreLoom-B': 'B_diagnostic', 'GRU-threshold': 'prior_gru'}
TWO_WINDOW = [f"{b['id']}-2W-{r}" for b in BASELINES.values() for r in ('AND', 'gate')]
GANBASE = {f'GANBase@{end}': end for end in (4500, 7500)}
GANBASE_CFG = Path(__file__).resolve().parents[1] / '20261011_ganbase/config.json'
STRATEGIES = ['No-AS', *NATIVE, *POLICY_FAMILY, *TWO_WINDOW, *GANBASE]


def profile_text(sim_out, seed, duration, target_weight):
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
sample_rate = 4000
sequencing_speed = 400
break_read_ms = 400
experiment_duration_set = {duration}
[[sample]]
name = "host"
input_genome = "{REFS / 'host.fa'}"
mean_read_length = 8000
weight = {1000 - target_weight}
[[sample]]
name = "target"
input_genome = "{REFS / 'target.fa'}"
mean_read_length = 8000
weight = {target_weight}
'''


# ---------------------------------------------------------------- strategies
class Strategy:
    """decide(jobs) -> list of (key, action) where action in accept/reject/defer."""
    max_samples = CAPTURE

    def needed(self, state):
        """Sample count the next decision needs, or None when no decision is pending."""
        raise NotImplementedError


class NoAS(Strategy):
    name = 'No-AS'

    def needed(self, s):
        return CAPTURE if s['k'] == 0 else None

    def decide(self, jobs):
        for s in jobs:
            s['k'] = 1
        return [(s, 'accept', {}) for s in jobs]


class Native(Strategy):
    def __init__(self, name, device, calibration):
        setup_imports()
        from native_window_retest import load_models, HOST_INDEX
        from score_defer_sim_native_development import prepare_windows
        self.name, self.tool, self.device = name, NATIVE[name], device
        job = next(j for j in read(REGISTRATION)['jobs'] if j['id'] == 'sim_baseline')
        self.model = load_models({'models': {self.tool: job['models'][self.tool]}}, device)[self.tool]
        self.host, self.prepare = HOST_INDEX[self.tool], prepare_windows
        self.lo, self.hi = BASELINES[self.tool]['window']
        self.cal = calibration

    def needed(self, s):
        return self.hi if s['k'] == 0 else None

    def logits(self, jobs, lo, hi):
        import torch
        raw = np.stack([s['buf'][lo:hi] for s in jobs])
        if self.tool == 'squigglenet':
            raw = ((raw.astype(np.float64) + self.cal['offset']) * self.cal['range'] / self.cal['digitisation']).astype(np.float32)
        x, ok = self.prepare(raw, self.tool)
        logits = np.full((len(jobs), 2), np.nan, np.float32)
        if ok.any():
            v = torch.from_numpy(x[ok]).to(self.device)
            if self.tool == 'nanodeep':
                v = v[:, None, :]
            with torch.inference_mode():
                logits[ok] = self.model(v).float().cpu().numpy()
        return logits

    def decide(self, jobs):
        logits = self.logits(jobs, self.lo, self.hi)
        out = []
        for i, s in enumerate(jobs):
            s['k'] = 1
            if not np.isfinite(logits[i]).all():
                out.append((s, 'accept', dict(reason='input_fallback')))
            else:
                out.append((s, 'reject' if int(np.argmax(logits[i])) == self.host else 'accept',
                            dict(margin=float(logits[i, self.host] - logits[i, 1 - self.host]))))
        return out


class Native2(Native):
    """Two consecutive native windows (twowin.py): W1 target -> accept; W1 host margin >= tau -> reject;
    otherwise wait for W2 and reject only if W2 is also host. tau = inf is the AND rule."""
    def __init__(self, name, device, calibration):
        from twowin import VARIANTS, selected_tau
        base, rule = VARIANTS[name]
        super().__init__(base, device, calibration)
        self.name, self.tau = name, selected_tau(name)
        self.lo2, self.hi2 = self.hi, self.hi + (self.hi - self.lo)

    def needed(self, s):
        return self.hi if s['k'] == 0 else self.hi2 if s['k'] == 1 else None

    def decide(self, jobs):
        out = []
        groups = {k: [s for s in jobs if s['k'] == k] for k in (0, 1)}     # fixed before any state is advanced
        for k, lo, hi in ((0, self.lo, self.hi), (1, self.lo2, self.hi2)):
            group = groups[k]
            if not group:
                continue
            logits = self.logits(group, lo, hi)
            for i, s in enumerate(group):
                s['k'] = k + 1
                if not np.isfinite(logits[i]).all():
                    out.append((s, 'accept', dict(reason='input_fallback')))
                    continue
                m = float(logits[i, self.host] - logits[i, 1 - self.host])
                if k == 1:
                    action = 'reject' if m >= 0 else 'accept'
                else:
                    action = 'accept' if m < 0 else 'reject' if m >= self.tau else 'defer'
                out.append((s, action, dict(margin=m)))
        return out


class Policy(Strategy):
    def __init__(self, name, device):
        setup_imports()
        import torch
        from run_poreloom_hostfloor import load_checkpoint
        from signal_explore.poreloom.model import build_model
        from signal_explore.poreloom.control_evaluation import RuleController
        from signal_explore.poreloom.windowing import normalize_windows
        self.torch, self.normalize, self.name, self.device = torch, normalize_windows, name, device
        cand = read(PARENT / 'frozen_selection.json')['selected'][POLICY_FAMILY[name]]
        if cand['kind'] == 'rule':
            cp = torch.load(PREFIX, map_location='cpu', weights_only=False)
            self.model = build_model(max_windows=4).to(device)
            self.model.load_state_dict(cp['model_state']); self.model.freeze_encoder(); self.model.eval()
            self.rule = RuleController(cand['rule'], cand['thresholds'], 4)
            self.controller = None
        else:
            assert pin(cand['path']) == cand['checkpoint']
            _, _, _, self.model, self.controller = load_checkpoint(cand['path'], device)
            self.rule = None
        self.candidate = cand

    def needed(self, s):
        return START + (s['k'] + 1) * WINDOW if s['k'] < K else None

    def decide(self, jobs):
        torch = self.torch
        windows = np.stack([s['buf'][START + s['k'] * WINDOW: START + (s['k'] + 1) * WINDOW] for s in jobs])
        x, valid = self.normalize(windows)
        out = []
        live = [i for i in range(len(jobs)) if valid[i]]
        for i in range(len(jobs)):
            if not valid[i]:
                jobs[i]['k'] = K
                out.append((jobs[i], 'accept', dict(reason='input_fallback')))
        if not live:
            return out
        with torch.inference_mode():
            e = self.model.encode(torch.from_numpy(x[live][:, None]).to(self.device))
            step = [jobs[i]['k'] + 1 for i in live]     # fixed before any state is advanced
            for k in sorted(set(step)):
                idx = [j for j, kk in enumerate(step) if kk == k]
                h = torch.stack([jobs[live[j]]['h'] if jobs[live[j]]['h'] is not None else torch.zeros(128, device=self.device) for j in idx])
                state, nh = self.model.step_state(e[idx], h, k)
                p = self.model.head(state).softmax(-1)[:, 1]
                if self.rule is not None:
                    scores = self.rule.score(p[:, None], k)
                    act = scores.argmin(-1)
                else:
                    from signal_explore.poreloom.controllers import mask_scores
                    scores = mask_scores(self.controller.score(state, k), self.controller.score_kind, k, K)
                    act = scores.argmin(-1) if self.controller.score_kind == 'cost' else scores.argmax(-1)
                act = act.cpu().tolist(); p = p.cpu().tolist()
                for r, (j, a) in enumerate(zip(idx, act)):
                    s = jobs[live[j]]
                    s['h'], s['k'] = nh[r], k
                    out.append((s, ('accept', 'reject', 'defer')[a], dict(p_target=p[r])))
        return out


class Ganbase(Strategy):
    """One fixed decision at `end` raw samples; reads too short to classify (< 250 called bases) are accepted."""
    def __init__(self, name, device, calibration):
        sys.path.insert(0, str(GANBASE_CFG.parents[2] / 'src'))
        from poreloom.baselines.ganbase import BonitoCaller, GanbaseClassifier
        cfg = read(GANBASE_CFG)
        self.name, self.end, self.cal, self.threshold = name, GANBASE[name], calibration, cfg['human_threshold']
        model = Path(cfg['output']) / 'resources' / cfg['basecaller_name']
        self.caller = BonitoCaller(model, device, batch_size=64)
        self.classifier = GanbaseClassifier(cfg['upstream'], device)
        self.pins = dict(config=pin(GANBASE_CFG), basecaller_weights=pin(model / 'weights_1.tar'),
                         classifier=pin(Path(cfg['upstream']) / 'model/Human_model.pt'))

    def needed(self, s):
        return self.end if s['k'] == 0 else None

    def decide(self, jobs):
        c = self.cal
        pa = [((s['buf'][:self.end].astype(np.float64) + c['offset']) * c['range'] / c['digitisation']).astype(np.float32) for s in jobs]
        seqs = self.caller.basecall(pa)
        prob = self.classifier.score(seqs)
        out = []
        for s, seq, p in zip(jobs, seqs, prob):
            s['k'] = 1
            if not np.isfinite(p):
                out.append((s, 'accept', dict(reason='input_fallback', called_bases=len(seq))))
            else:
                out.append((s, 'reject' if p > self.threshold else 'accept', dict(p_human=float(p), called_bases=len(seq))))
        return out


def build_strategy(name, device, calibration):
    if name == 'No-AS':
        return NoAS()
    if name in GANBASE:
        return Ganbase(name, device, calibration)
    if name in NATIVE:
        return Native(name, device, calibration)
    if name in TWO_WINDOW:
        return Native2(name, device, calibration)
    return Policy(name, device)


# ---------------------------------------------------------------- live client
class Client:
    def __init__(self, out, strategy, channels, duration, ports, follow, calibration):
        src, dst = calibration, TRAIN_CAL
        # adc_train = (adc + off_src) * scale_src / scale_dst - off_dst
        self.gain = (src['range'] / src['digitisation']) / (dst['range'] / dst['digitisation'])
        self.shift = src['offset'] * self.gain - dst['offset']
        self.out, self.strategy, self.channels = out, strategy, channels
        self.duration, self.follow, self.ports = duration, follow, ports
        self.lock = threading.Lock()
        self.reads = {}                 # channel -> state dict for the current read
        self.ready = queue.Queue()
        self.requests = queue.Queue()
        self.stop = threading.Event()
        self.n_action = 0
        self.t0 = None
        self.failure = None
        self.last_packet = None
        self.logs = dict(decisions=open(out / 'decisions.jsonl', 'w'), reads=open(out / 'client_reads.jsonl', 'w'),
                         acks=open(out / 'acks.jsonl', 'w'))
        self.capture = open(out / 'capture.bin', 'wb') if isinstance(strategy, NoAS) else None
        self.capture_index = []
        self.stats = dict(packets=0, samples=0, gaps=0, overlaps=0, stale_actions=0, batches=0, max_queue=0)

    def now(self):
        return time.monotonic() - self.t0

    def log(self, kind, rec):
        self.logs[kind].write(json.dumps(rec) + '\n')

    def close_read(self, s, reason):
        rec = {k: v for k, v in s.items() if k not in ('buf', 'h')}
        rec['closed'] = reason
        self.log('reads', rec)
        if self.capture is not None and s['received'] >= 1:
            n = min(s['received'], CAPTURE)
            self.capture_index.append(dict(channel=s['channel'], read_id=s['read_id'], read_number=s['read_number'],
                                           offset=self.capture.tell() // 2, samples=n))
            self.capture.write(s['buf'][:n].astype('<i2').tobytes())

    def on_packet(self, ch, rd, at):
        self.last_packet = at
        raw = np.rint(np.frombuffer(rd.raw_data, '<i2') * self.gain + self.shift).astype(np.int16)
        self.stats['packets'] += 1; self.stats['samples'] += len(raw)
        with self.lock:
            s = self.reads.get(ch)
            if s is None or s['read_number'] != rd.number:
                if s is not None and s['read_number'] > rd.number:
                    return                       # stale packet of an older read
                if s is not None:
                    self.close_read(s, 'replaced')
                s = dict(channel=ch, read_id=rd.id, read_number=rd.number, start_sample=rd.start_sample,
                         first_seen=at, received=0, k=0, h=None, decided=None, pending=False,
                         buf=np.zeros(CAPTURE if self.capture is not None else 13500, np.int16))
                self.reads[ch] = s
            offset = rd.chunk_start_sample - rd.start_sample
            if offset > s['received']:
                self.stats['gaps'] += 1
                s['decided'] = s['decided'] or 'gap'
                return
            if offset < s['received']:
                self.stats['overlaps'] += 1
                raw = raw[s['received'] - offset:]
            take = max(0, min(len(raw), len(s['buf']) - s['received']))
            s['buf'][s['received']:s['received'] + take] = raw[:take]
            s['received'] += len(raw)
            s['last_packet'] = at
            if s['decided'] is None and not s['pending']:
                need = self.strategy.needed(s)
                if need is not None and s['received'] >= need:
                    s['pending'] = True; s['ready_at'] = at
                    self.ready.put(s)

    def infer_loop(self):
        try:
            self._infer_loop()
        except BaseException as exc:
            self.failure = f'{type(exc).__name__}: {exc}'
            self.stop.set()
            raise

    def _infer_loop(self):
        while not self.stop.is_set():
            try:
                first = self.ready.get(timeout=.02)
            except queue.Empty:
                continue
            jobs = [first]
            while True:
                try:
                    jobs.append(self.ready.get_nowait())
                except queue.Empty:
                    break
            self.stats['max_queue'] = max(self.stats['max_queue'], len(jobs)); self.stats['batches'] += 1
            with self.lock:
                jobs = [s for s in jobs if self.reads.get(s['channel']) is s]
            if not jobs:
                continue
            t_start = self.now()
            results = self.strategy.decide(jobs)
            t_end = self.now()
            with self.lock:
                for s, action, info in results:
                    current = self.reads.get(s['channel']) is s
                    rec = dict(channel=s['channel'], read_id=s['read_id'], read_number=s['read_number'], k=s['k'],
                               action=action, ready_at=s['ready_at'], infer_start=t_start, infer_end=t_end,
                               batch=len(jobs), received=s['received'], current=current, **info)
                    s['pending'] = False
                    if action == 'defer':
                        need = self.strategy.needed(s)
                        if need is not None and s['received'] >= need:
                            s['pending'] = True; s['ready_at'] = t_end; self.ready.put(s)
                    else:
                        s['decided'] = action; s['decided_at'] = t_end
                        if current:
                            self.n_action += 1
                            aid = f'a{self.n_action}'
                            rec['action_id'] = aid; rec['sent'] = self.now()
                            self.send(aid, s, action)
                        else:
                            self.stats['stale_actions'] += 1
                    self.log('decisions', rec)

    def send(self, aid, s, action):
        from minknow_api import data_pb2
        a = data_pb2.GetLiveReadsRequest.Action(action_id=aid, channel=s['channel'], number=s['read_number'])
        if action == 'reject':
            a.unblock.duration = .1
        else:
            a.stop_further_data.SetInParent()
        self.requests.put(data_pb2.GetLiveReadsRequest(actions=data_pb2.GetLiveReadsRequest.Actions(actions=[a])))

    def request_stream(self):
        from minknow_api import data_pb2
        yield data_pb2.GetLiveReadsRequest(setup=data_pb2.GetLiveReadsRequest.StreamSetup(
            first_channel=1, last_channel=self.channels, raw_data_type=3, sample_minimum_chunk_size=300))
        while not self.stop.is_set():
            try:
                r = self.requests.get(timeout=.05)
            except queue.Empty:
                continue
            yield r

    def run(self):
        import grpc
        from minknow_api import data_pb2_grpc, device_pb2, device_pb2_grpc, acquisition_pb2, acquisition_pb2_grpc
        cred = grpc.ssl_channel_credentials(root_certificates=(TLS / 'server.crt').read_bytes())
        opts = [('grpc.max_receive_message_length', 256 << 20)]
        position = grpc.secure_channel(f'127.0.0.1:{self.ports[1]}', cred, options=opts)
        grpc.channel_ready_future(position).result(timeout=600)
        device = device_pb2_grpc.DeviceServiceStub(position)
        info = dict(run_id=acquisition_pb2_grpc.AcquisitionServiceStub(position).get_current_acquisition_run(
                        acquisition_pb2.GetCurrentAcquisitionRunRequest(), timeout=10).run_id,
                    sample_rate=device.get_sample_rate(device_pb2.GetSampleRateRequest(), timeout=10).sample_rate)
        worker = threading.Thread(target=self.infer_loop, daemon=True)
        worker.start()
        self.t0 = time.monotonic()
        info['client_t0_wall'] = time.time()
        rpc = data_pb2_grpc.DataServiceStub(position).get_live_reads(self.request_stream())
        end = self.duration + self.follow

        def watchdog():
            while not self.stop.wait(.5):
                if self.now() > end:
                    self.stop.set(); rpc.cancel()
                elif self.last_packet is not None and self.now() - self.last_packet > 10:
                    self.failure = 'simulator stream stalled >10 s'; self.stop.set(); rpc.cancel()
            if self.failure:                   # inference thread died: end the stream instead of hanging on it
                rpc.cancel()
        threading.Thread(target=watchdog, daemon=True).start()
        try:
            for response in rpc:
                at = self.now()
                for ack in response.action_responses:
                    self.log('acks', dict(action_id=ack.action_id, response=int(ack.response), at=at))
                for ch, rd in response.channels.items():
                    self.on_packet(int(ch), rd, at)
        except grpc.RpcError as exc:
            if not self.stop.is_set() or self.failure:
                raise RuntimeError(self.failure or 'stream ended early') from exc
        self.stop.set()
        worker.join(timeout=30)
        if self.failure:
            raise RuntimeError('inference thread failed: ' + self.failure)
        with self.lock:
            for s in self.reads.values():
                self.close_read(s, 'run_end')
        for f in self.logs.values():
            f.close()
        if self.capture is not None:
            self.capture.close()
            write(self.out / 'capture_index.json', self.capture_index)
        position.close()
        return info


def calibration_from_server(port):
    import grpc
    from minknow_api import device_pb2, device_pb2_grpc
    cred = grpc.ssl_channel_credentials(root_certificates=(TLS / 'server.crt').read_bytes())
    ch = grpc.secure_channel(f'127.0.0.1:{port}', cred)
    grpc.channel_ready_future(ch).result(timeout=600)
    c = device_pb2_grpc.DeviceServiceStub(ch).get_calibration(device_pb2.GetCalibrationRequest(first_channel=1, last_channel=1), timeout=10)
    ch.close()
    return dict(offset=float(c.offsets[0]), range=float(c.pa_ranges[0]), digitisation=float(c.digitisation))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--strategy', choices=STRATEGIES, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--channels', type=int, default=512)
    ap.add_argument('--duration', type=int, default=1200, help='fixed yield window T, seconds of client stream')
    ap.add_argument('--follow', type=int, default=20, help='seconds of follow-up after T for decisions in flight')
    ap.add_argument('--target-permille', type=int, default=10, help='target sample weight out of 1000 (10 = 1%%)')
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--device', default='cuda:3')
    args = ap.parse_args()
    out = args.out
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    assert pin(ICARUST_BIN)['sha256'] == ICARUST_SHA
    # Icarust aborts when its FAST5 writer lags its 4096-read retention budget; write to tmpfs and keep
    # only the audit log (the truth source) afterwards. FAST5 signal is not used by the evaluation.
    sim_out = SHM / f'{out.parent.name}__{out.name}'  # run names repeat across concurrent matrices
    if sim_out.exists():
        shutil.rmtree(sim_out)
    (out / 'profile.toml').write_text(profile_text(sim_out, args.seed, args.duration + args.follow + 120, args.target_permille))
    (out / 'config.ini').write_text(f'[TLS]\ncert-dir = {TLS}\n[PORTS]\nmanager = {args.port}\nposition = {args.port + 1}\n'
                                    f'[SEQUENCER]\nchannels = {args.channels}\n')
    import torch
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    env = dict(os.environ, LD_LIBRARY_PATH=LD_PATH, HDF5_PLUGIN_PATH=str(VBZ))
    server = subprocess.Popen([str(ICARUST_BIN), '-s', str(out / 'profile.toml'), '-c', str(out / 'config.ini'), '-v', '-p'],
                              cwd=ICARUST_SRC, env=env, stdout=open(out / 'server.log', 'wb'), stderr=subprocess.STDOUT,
                              start_new_session=True)
    manifest = dict(strategy=args.strategy, seed=args.seed, channels=args.channels, duration=args.duration, follow=args.follow,
                    target_permille=args.target_permille, device=args.device, icarust=dict(binary=str(ICARUST_BIN), sha256=ICARUST_SHA),
                    references=pin(REFS / 'manifest.json'), client=pin(__file__), started_wall=time.time(), server_pid=server.pid)
    try:
        cal = calibration_from_server(args.port + 1)
        strategy = build_strategy(args.strategy, args.device, TRAIN_CAL)
        manifest.update(calibration=cal, model_input_calibration=TRAIN_CAL, strategy_pins=getattr(strategy, 'pins', None))
        client = Client(out, strategy, args.channels, args.duration, (args.port, args.port + 1), args.follow, cal)
        manifest.update(client.run())
        manifest['stats'] = client.stats
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest['status'] = f'failed: {type(exc).__name__}: {exc}'
        raise
    finally:
        try:
            os.killpg(server.pid, signal.SIGINT)
            server.wait(timeout=30)
        except Exception:
            os.killpg(server.pid, signal.SIGKILL)
        manifest['server_exit'] = server.returncode
        kept = []
        for audit in sim_out.glob('**/e7_engineering_audit.jsonl'):
            dest = out / 'sim' / audit.relative_to(sim_out)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(audit, dest); kept.append(str(dest))
        manifest['simulator_output'] = dict(tmpfs=str(sim_out), kept=kept,
                                            raw_bytes_discarded=sum(f.stat().st_size for f in sim_out.glob('**/*.pod5')))
        shutil.rmtree(sim_out, ignore_errors=True)
        manifest['finished_wall'] = time.time()
        write(out / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
