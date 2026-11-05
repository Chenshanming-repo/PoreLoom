#!/usr/bin/env python3
"""Join one live run's client log with the simulator audit (truth) and compute run metrics.

Cohort: reads that start inside the fixed window [t_s, t_s + T] of the simulator clock,
t_s = first read start. Bases are counted only inside the window (10 samples/base after
the 913-sample prefix). Unknown fates are reported as right-censored, never as "kept".
"""
from pathlib import Path
import argparse
import collections
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import write

RATE, SPB = 4000, 10


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def analyze(run):
    run = Path(run)
    m = json.loads((run / 'manifest.json').read_text())
    audit = next(run.glob('sim/**/e7_engineering_audit.jsonl'))
    ev = load_jsonl(audit)
    started = {(e['channel'], e['read_number']): e for e in ev if e['event'] == 'read_started'}
    final = {(e['channel'], e['read_number']): e for e in ev if e['event'] == 'read_finalized'}
    unblock = {(e['channel'], e['read_number']): e for e in ev if e['event'] == 'unblock_accepted'}
    decisions = collections.defaultdict(list)
    for d in load_jsonl(run / 'decisions.jsonl'):
        decisions[(d['channel'], d['read_number'])].append(d)
    acks = {a['action_id']: a for a in load_jsonl(run / 'acks.jsonl')}
    t_s = min(e['monotonic_seconds'] for e in started.values())
    T = m['duration']; t_e = t_s + T
    rows = []
    for key, st in started.items():
        if not t_s <= st['monotonic_seconds'] < t_e:
            continue
        fin = final.get(key)
        ds = decisions.get(key, [])
        term = next((d for d in ds if d['action'] != 'defer' and d.get('current')), None)
        t0 = st['monotonic_seconds']
        if fin is not None:
            end_t, produced, reason = fin['monotonic_seconds'], fin['produced_samples'], fin['reason']
        else:
            end_t, produced, reason = None, None, 'censored'
        upto = t_e if end_t is None else min(end_t, t_e)
        produced_in = min(int((upto - t0) * RATE), st['total_samples'] if produced is None else produced)
        bases_in = max(0, produced_in - st['prefix_samples']) / SPB
        ub = unblock.get(key)
        rows.append(dict(
            channel=key[0], read_number=key[1], read_id=st['read_id'], target=st['sample'] == 'target',
            natural_samples=st['total_samples'], start=t0 - t_s, end=None if end_t is None else end_t - t_s,
            reason=reason, produced=produced, bases_in_window=bases_in,
            command=term['action'] if term else None, command_k=term.get('k') if term else None,
            command_latency=(term['sent'] - term['ready_at']) if term and 'sent' in term else None,
            ack=acks.get(term.get('action_id'), {}).get('response') if term else None,
            effective_reject=reason == 'unblock', unblock_samples=ub['produced_samples'] if ub else None,
            pore_seconds=None if end_t is None else end_t - t0, windows_seen=max([d['k'] for d in ds], default=0)))
    return m, rows


def summarize(m, rows):
    host = [r for r in rows if not r['target']]
    tgt = [r for r in rows if r['target']]
    hours = m['duration'] / 3600
    known = lambda rs: [r for r in rs if r['reason'] != 'censored']

    def rate(rs, f):
        k = sum(f(r) for r in known(rs)); u = sum(r['reason'] == 'censored' for r in rs); n = len(rs)
        return dict(value=k / max(1, len(known(rs))), lower=k / max(1, n), upper=(k + u) / max(1, n), events=k, censored=u, n=n)

    keep_t = [r for r in tgt if not r['effective_reject']]
    out = dict(
        strategy=m['strategy'], seed=m['seed'], channels=m['channels'], duration_s=m['duration'],
        reads=len(rows), host_reads=len(host), target_reads=len(tgt),
        realized_target_read_fraction=len(tgt) / max(1, len(rows)),
        host_effective_reject=rate(host, lambda r: r['effective_reject']),
        target_command_reject=rate(tgt, lambda r: r['command'] == 'reject'),
        target_effective_reject=rate(tgt, lambda r: r['effective_reject']),
        target_bases_retained=sum(r['bases_in_window'] for r in keep_t),
        target_bases_rejected_prefix=sum(r['bases_in_window'] for r in tgt if r['effective_reject']),
        host_bases=sum(r['bases_in_window'] for r in host),
        total_bases=sum(r['bases_in_window'] for r in rows),
        complete_target_molecules=sum(1 for r in keep_t if r['reason'] == 'natural_end' and r['end'] is not None and r['end'] <= m['duration']),
        host_mean_pore_seconds=float(np.mean([r['pore_seconds'] for r in host if r['pore_seconds'] is not None])) if host else None,
        host_unblock_samples_p50=float(np.median([r['unblock_samples'] for r in host if r['unblock_samples'] is not None])) if any(r['unblock_samples'] for r in host) else None,
        command_latency_s=dict(zip(('p50', 'p95', 'p99'), np.percentile([r['command_latency'] for r in rows if r['command_latency'] is not None], [50, 95, 99]).tolist()))
        if any(r['command_latency'] is not None for r in rows) else None,
        failed_acks=sum(1 for r in rows if r['ack'] not in (None, 0)),
        client_stats=m.get('stats'), status=m.get('status'))
    out['yield_target_bp_per_channel_hour'] = out['target_bases_retained'] / (m['channels'] * hours)
    out['target_fraction_of_retained_bases'] = out['target_bases_retained'] / max(1, out['target_bases_retained'] + sum(
        r['bases_in_window'] for r in host if not r['effective_reject']))
    out['target_fraction_of_all_bases'] = sum(r['bases_in_window'] for r in tgt) / max(1, out['total_bases'])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs', nargs='+', type=Path)
    args = ap.parse_args()
    for run in args.runs:
        m, rows = analyze(run)
        s = summarize(m, rows)
        write(run / 'summary.json', s)
        with (run / 'read_outcomes.jsonl').open('w') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
        print(json.dumps({k: s[k] for k in ('strategy', 'reads', 'target_reads', 'yield_target_bp_per_channel_hour')}),
              'host_eff', round(s['host_effective_reject']['value'], 4), 'tgt_cmd', round(s['target_command_reject']['value'], 4))


if __name__ == '__main__':
    main()
