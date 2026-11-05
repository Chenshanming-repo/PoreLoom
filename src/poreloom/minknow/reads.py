"""Per-channel bookkeeping of the read currently in each pore, fed by live-read chunks.

Same semantics as the Icarust client (experiments/20261011_simloop/closed_loop.py ``Client.on_packet``): a chunk
of a newer read replaces the channel's read, chunks of older reads are dropped, overlaps are trimmed, a gap marks
the read as undecidable, and a read is queued once the policy's next decision point has arrived.
"""
import numpy as np


class ReadBook:
    """needed(state) -> sample count the next decision needs, or None; on_close(state, reason) on replacement.
    accepted: classification ids whose reads are eligible (checked on a read's first chunk); None = all reads."""

    def __init__(self, needed, buffer_samples, accepted=None, on_close=None):
        self.needed, self.buffer_samples = needed, int(buffer_samples)
        self.accepted = None if accepted is None else set(accepted)
        self.on_close = on_close
        self.reads = {}
        self.stats = dict(packets=0, samples=0, gaps=0, overlaps=0, stale_packets=0, skipped_class=0)

    def current(self, channel):
        return self.reads.get(channel)

    def add(self, channel, read_id, number, start_sample, chunk_start_sample, raw, classifications, at):
        """Append one chunk (int16, model grid). Returns the read state when it has just become ready."""
        st = self.stats
        st['packets'] += 1; st['samples'] += len(raw)
        s = self.reads.get(channel)
        if s is None or s['read_number'] != number:
            if s is not None and s['read_number'] > number:
                st['stale_packets'] += 1
                return None
            if s is not None and self.on_close is not None:
                self.on_close(s, 'replaced')
            s = dict(channel=channel, read_id=read_id, read_number=number, start_sample=start_sample, first_seen=at,
                     received=0, k=0, decided=None, pending=False, buf=np.zeros(self.buffer_samples, np.int16))
            if self.accepted is not None and not self.accepted.intersection(classifications):
                s['decided'] = 'skipped_class'; st['skipped_class'] += 1
            self.reads[channel] = s
        offset = chunk_start_sample - s['start_sample']
        if offset > s['received']:
            st['gaps'] += 1
            s['decided'] = s['decided'] or 'gap'
            return None
        if offset < s['received']:
            st['overlaps'] += 1
            raw = raw[s['received'] - offset:]
        take = max(0, min(len(raw), self.buffer_samples - s['received']))
        s['buf'][s['received']:s['received'] + take] = raw[:take]
        s['received'] += len(raw)
        s['last_packet'] = at
        return s if self.ready(s, at) else None

    def ready(self, s, at):
        """Queue s when its next decision point has arrived; True when it was queued."""
        if s['decided'] is not None or s['pending']:
            return False
        need = self.needed(s)
        if need is None or s['received'] < need:
            return False
        s['pending'] = True; s['ready_at'] = at
        return True
