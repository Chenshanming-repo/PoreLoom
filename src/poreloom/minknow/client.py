"""Live read-until client: one bidirectional ``DataService.get_live_reads`` stream per run.

The receiving thread maps each chunk to the model ADC grid and updates the read book; a separate inference thread
batches every read whose next decision point has arrived, so network receipt never waits for the GPU (the design
of the Icarust client, experiments/20261011_simloop/closed_loop.py). Terminal decisions of one inference step are
sent together in one request: reject -> ``unblock`` for ``unblock_duration`` seconds, accept ->
``stop_further_data``. Run logs: decisions.jsonl, acks.jsonl, reads.jsonl.
"""
from pathlib import Path
import json
import queue
import threading
import time

import numpy as np

from .reads import ReadBook
from .signal import to_model_adc


class LiveClient:
    """connection: a minknow_api position Connection (or anything with ``.data.get_live_reads(requests)``).
    policy: ``needed(state)``, ``decide(jobs) -> [(state, 'accept'|'reject'|'defer', info)]``, optional
    ``release(state)`` and ``max_samples`` (prefix length to buffer). gain/shift: per channel, from
    ``signal.adc_mapping``, indexed from first_channel. duration: seconds of streaming, None = until MinKNOW ends
    the stream. accepted: classification ids of reads eligible for decisions (None = all)."""

    def __init__(self, connection, policy, out, gain, shift, first_channel=1, last_channel=512, duration=None,
                 unblock_duration=.1, dry_run=False, accepted=None, stall_seconds=60., chunk_size=300):
        self.connection, self.policy, self.out = connection, policy, Path(out)
        self.gain, self.shift = np.asarray(gain, np.float64), np.asarray(shift, np.float64)
        self.first_channel, self.last_channel = int(first_channel), int(last_channel)
        if len(self.gain) != self.last_channel - self.first_channel + 1 or len(self.shift) != len(self.gain):
            raise ValueError('gain/shift must have one entry per channel in [first_channel, last_channel]')
        self.duration, self.unblock_duration, self.dry_run = duration, float(unblock_duration), bool(dry_run)
        self.stall_seconds, self.chunk_size = float(stall_seconds), int(chunk_size)
        self.book = ReadBook(policy.needed, policy.max_samples, accepted, on_close=self._closed)
        self.lock = threading.Lock()
        self.ready, self.requests = queue.Queue(), queue.Queue()
        self.stop = threading.Event()
        self.t0 = self.last_packet = self.failure = self.reason = None
        self.n_action = 0
        self.stats = dict(actions_sent=0, stale_actions=0, decision_steps=0, max_queue=0)
        self.out.mkdir(parents=True, exist_ok=True)
        self.logs = {k: open(self.out/f'{k}.jsonl', 'w') for k in ('decisions', 'acks', 'reads')}

    def now(self):
        return time.monotonic() - self.t0

    def log(self, kind, rec):
        self.logs[kind].write(json.dumps(rec) + '\n')

    def _closed(self, s, reason):
        release = getattr(self.policy, 'release', None)
        if release is not None:
            release(s)
        rec = {k: v for k, v in s.items() if k not in ('buf', 'h')}
        rec['closed'] = reason
        self.log('reads', rec)

    # ------------------------------------------------------------ stream side
    def on_response(self, response, at):
        for ack in response.action_responses:
            self.log('acks', dict(action_id=ack.action_id, response=int(ack.response), at=at))
        for ch, rd in response.channels.items():
            ch = int(ch); i = ch - self.first_channel
            raw = to_model_adc(np.frombuffer(rd.raw_data, '<i2'), self.gain[i], self.shift[i])
            self.last_packet = at
            with self.lock:
                s = self.book.add(ch, rd.id, rd.number, rd.start_sample, rd.chunk_start_sample, raw,
                                  tuple(rd.chunk_classifications), at)
                if s is not None:
                    self.ready.put(s)

    def request_stream(self):
        from minknow_api import data_pb2
        R = data_pb2.GetLiveReadsRequest
        yield R(setup=R.StreamSetup(first_channel=self.first_channel, last_channel=self.last_channel,
                                    raw_data_type=R.UNCALIBRATED, sample_minimum_chunk_size=self.chunk_size))
        while not self.stop.is_set():
            try:
                r = self.requests.get(timeout=.05)
            except queue.Empty:
                continue
            yield r

    def send(self, actions):
        """actions: [(action_id, state, 'accept'|'reject')] -> one request."""
        from minknow_api import data_pb2
        R = data_pb2.GetLiveReadsRequest
        out = []
        for aid, s, action in actions:
            a = R.Action(action_id=aid, channel=s['channel'], number=s['read_number'])
            if action == 'reject':
                a.unblock.duration = self.unblock_duration
            else:
                a.stop_further_data.SetInParent()
            out.append(a)
        self.requests.put(R(actions=R.Actions(actions=out)))

    # ------------------------------------------------------------ inference side
    def infer_loop(self):
        try:
            self._infer_loop()
        except BaseException as exc:
            self.failure = f'{type(exc).__name__}: {exc}'
            self.stop.set()

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
            self.stats['max_queue'] = max(self.stats['max_queue'], len(jobs))
            with self.lock:
                jobs = [s for s in jobs if self.book.current(s['channel']) is s]
            if not jobs:
                continue
            t_start = self.now()
            results = self.policy.decide(jobs)
            t_end = self.now()
            self.stats['decision_steps'] += 1
            with self.lock:
                batch = []
                for s, action, info in results:
                    current = self.book.current(s['channel']) is s
                    rec = dict(channel=s['channel'], read_id=s['read_id'], read_number=s['read_number'], k=s['k'],
                               action=action, ready_at=s['ready_at'], infer_start=t_start, infer_end=t_end,
                               batch=len(jobs), received=s['received'], current=current, **info)
                    s['pending'] = False
                    if action == 'defer':
                        if self.book.ready(s, t_end):
                            self.ready.put(s)
                    else:
                        s['decided'] = action; s['decided_at'] = t_end
                        if not current:
                            self.stats['stale_actions'] += 1
                        elif not self.dry_run:
                            self.n_action += 1
                            rec['action_id'] = f'a{self.n_action}'; rec['sent'] = self.now()
                            batch.append((rec['action_id'], s, action))
                    self.log('decisions', rec)
                if batch:
                    self.send(batch); self.stats['actions_sent'] += len(batch)

    # ------------------------------------------------------------ run
    def run(self):
        import grpc
        self.t0 = time.monotonic()
        info = dict(client_t0_wall=time.time(), first_channel=self.first_channel, last_channel=self.last_channel,
                    dry_run=self.dry_run, unblock_duration=self.unblock_duration, duration=self.duration)
        worker = threading.Thread(target=self.infer_loop, daemon=True)
        worker.start()
        rpc = self.connection.data.get_live_reads(self.request_stream())

        def watchdog():
            while not self.stop.wait(.2):
                if self.duration is not None and self.now() > self.duration:
                    self.reason = 'duration_reached'; break
                if self.last_packet is not None and self.now() - self.last_packet > self.stall_seconds:
                    self.failure = f'no live-read data for {self.stall_seconds:g} s'; break
            self.stop.set(); rpc.cancel()
        threading.Thread(target=watchdog, daemon=True).start()
        try:
            for response in rpc:
                self.on_response(response, self.now())
                if self.stop.is_set():
                    break
            self.reason = self.reason or 'stream_ended'
        except grpc.RpcError as exc:
            if not self.stop.is_set() or self.failure:
                self.failure = self.failure or f'live-read stream failed: {exc}'
        except KeyboardInterrupt:
            self.reason = 'interrupted'
        finally:
            self.stop.set()
            try:
                rpc.cancel()
            except Exception:
                pass
            worker.join(timeout=30)
            with self.lock:
                for s in list(self.book.reads.values()):
                    self._closed(s, 'run_end')
            for f in self.logs.values():
                f.close()
        if self.failure:
            raise RuntimeError(self.failure)
        info.update(status=self.reason, finished_wall=time.time(), stats=dict(self.book.stats, **self.stats))
        return info
