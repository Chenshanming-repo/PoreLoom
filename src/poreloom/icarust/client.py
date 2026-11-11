"""Read-until client of one Icarust run (the client of the paper's closed-loop runs).

One ``get_live_reads`` stream covers every channel. The receiving thread maps each chunk from the simulator's ADC to
the model's ADC grid and appends it to the buffer of the channel's current read (keyed by read number, indexed by
sample coordinates: newer reads replace older ones, stale chunks are dropped, overlaps trimmed, a gap makes the read
undecidable). A separate inference thread batches every read whose next decision point has arrived, so receipt
never waits for the GPU. Each terminal decision is sent at once in its own request: reject -> ``unblock`` for
0.1 s, accept -> ``stop_further_data``. Logs: decisions.jsonl, acks.jsonl, client_reads.jsonl.
"""
from pathlib import Path
import json
import queue
import threading
import time

import numpy as np

BUFFER_SAMPLES = 13500          # samples kept per read; every strategy decides within the first 7500
UNBLOCK_SECONDS = .1
STALL_SECONDS = 10.


def adc_map(source, model):
    """(gain, shift) with adc_model = adc_source * gain + shift, so both give the same picoamperes."""
    gain = (source['range']/source['digitisation'])/(model['range']/model['digitisation'])
    return gain, source['offset']*gain - model['offset']


class Client:
    """strategy: ``needed(state)`` -> samples the next decision needs or None, ``decide(jobs)`` ->
    [(state, 'accept' | 'reject' | 'defer', info)], optional ``release(state)``. calibration: the simulator's ADC
    calibration; model_calibration: the ADC grid the strategy reads."""

    def __init__(self, out, strategy, channels, duration, port, follow, calibration, model_calibration, certificate):
        self.gain, self.shift = adc_map(calibration, model_calibration)
        self.out, self.strategy, self.channels = Path(out), strategy, channels
        self.duration, self.follow, self.port, self.certificate = duration, follow, port, Path(certificate)
        self.lock = threading.Lock()
        self.reads = {}                 # channel -> state of its current read
        self.ready = queue.Queue()
        self.requests = queue.Queue()
        self.stop = threading.Event()
        self.n_action = 0
        self.t0 = None
        self.failure = None
        self.last_packet = None
        self.logs = dict(decisions=open(self.out/'decisions.jsonl', 'w'), reads=open(self.out/'client_reads.jsonl', 'w'),
                         acks=open(self.out/'acks.jsonl', 'w'))
        self.stats = dict(packets=0, samples=0, gaps=0, overlaps=0, stale_actions=0, batches=0, max_queue=0)

    def now(self):
        return time.monotonic() - self.t0

    def log(self, kind, rec):
        self.logs[kind].write(json.dumps(rec) + '\n')

    def close_read(self, s, reason):
        release = getattr(self.strategy, 'release', None)
        if release is not None:
            release(s)
        rec = {k: v for k, v in s.items() if k not in ('buf', 'h')}
        rec['closed'] = reason
        self.log('reads', rec)

    def on_packet(self, ch, rd, at):
        self.last_packet = at
        raw = np.rint(np.frombuffer(rd.raw_data, '<i2')*self.gain + self.shift).astype(np.int16)
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
                         buf=np.zeros(BUFFER_SAMPLES, np.int16))
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
            a.unblock.duration = UNBLOCK_SECONDS
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
        """Stream for duration + follow seconds; returns run information."""
        import grpc
        from minknow_api import data_pb2_grpc, device_pb2, device_pb2_grpc, acquisition_pb2, acquisition_pb2_grpc
        cred = grpc.ssl_channel_credentials(root_certificates=self.certificate.read_bytes())
        position = grpc.secure_channel(f'127.0.0.1:{self.port}', cred, options=[('grpc.max_receive_message_length', 256 << 20)])
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
                elif self.last_packet is not None and self.now() - self.last_packet > STALL_SECONDS:
                    self.failure = f'simulator stream stalled > {STALL_SECONDS:g} s'; self.stop.set(); rpc.cancel()
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
        position.close()
        return info


def server_calibration(port, certificate, timeout=600):
    """ADC calibration of channel 1 (Icarust uses one calibration for every channel). Waits until the position
    service is up: Icarust serves it only after converting the references to signal."""
    import grpc
    from minknow_api import device_pb2, device_pb2_grpc
    cred = grpc.ssl_channel_credentials(root_certificates=Path(certificate).read_bytes())
    ch = grpc.secure_channel(f'127.0.0.1:{port}', cred)
    grpc.channel_ready_future(ch).result(timeout=timeout)
    c = device_pb2_grpc.DeviceServiceStub(ch).get_calibration(device_pb2.GetCalibrationRequest(first_channel=1, last_channel=1),
                                                              timeout=10)
    ch.close()
    return dict(offset=float(c.offsets[0]), range=float(c.pa_ranges[0]), digitisation=float(c.digitisation))
