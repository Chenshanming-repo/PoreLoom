"""Metrics of closed-loop runs.

One run (``summarize_run``): the client's decisions and acknowledgements are joined with the simulator's audit log,
the only source of truth (sample, read length, end reason). The cohort is every read that starts inside the fixed
window [t_s, t_s + T] of the simulator clock, t_s = first read start, T = the run's duration. Bases are counted only
inside the window (10 samples per base after the read's prefix). A read without a recorded end is right-censored,
never counted as kept. Writes summary.json and read_outcomes.jsonl into the run directory.

Several runs (``aggregate``): target-base enrichment = a run's target yield divided by the No-AS yield of the same
seed and target fraction; every metric is reported as mean +- sample SD over seeds.
"""
from pathlib import Path
import collections
import json

import numpy as np

RATE, SAMPLES_PER_BASE = 4000, 10


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def analyze(run):
    run = Path(run)
    m = json.loads((run/'manifest.json').read_text())
    audit = next(run.glob('sim/**/e7_engineering_audit.jsonl'), None)
    if audit is None:
        raise FileNotFoundError(f'{run}/sim/**/e7_engineering_audit.jsonl (simulator truth log) is missing')
    ev = load_jsonl(audit)
    started = {(e['channel'], e['read_number']): e for e in ev if e['event'] == 'read_started'}
    final = {(e['channel'], e['read_number']): e for e in ev if e['event'] == 'read_finalized'}
    unblock = {(e['channel'], e['read_number']): e for e in ev if e['event'] == 'unblock_accepted'}
    decisions = collections.defaultdict(list)
    for d in load_jsonl(run/'decisions.jsonl'):
        decisions[(d['channel'], d['read_number'])].append(d)
    acks = {a['action_id']: a for a in load_jsonl(run/'acks.jsonl')}
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
        produced_in = min(int((upto - t0)*RATE), st['total_samples'] if produced is None else produced)
        bases_in = max(0, produced_in - st['prefix_samples'])/SAMPLES_PER_BASE
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
    hours = m['duration']/3600
    known = lambda rs: [r for r in rs if r['reason'] != 'censored']  # noqa: E731

    def rate(rs, f):
        k = sum(f(r) for r in known(rs)); u = sum(r['reason'] == 'censored' for r in rs); n = len(rs)
        return dict(value=k/max(1, len(known(rs))), lower=k/max(1, n), upper=(k + u)/max(1, n), events=k, censored=u, n=n)

    keep_t = [r for r in tgt if not r['effective_reject']]
    latency = [r['command_latency'] for r in rows if r['command_latency'] is not None]
    unblocked = [r['unblock_samples'] for r in host if r['unblock_samples'] is not None]
    out = dict(
        strategy=m['strategy'], seed=m['seed'], target_permille=m.get('target_permille'), channels=m['channels'],
        duration_s=m['duration'], reads=len(rows), host_reads=len(host), target_reads=len(tgt),
        realized_target_read_fraction=len(tgt)/max(1, len(rows)),
        host_effective_reject=rate(host, lambda r: r['effective_reject']),
        target_command_reject=rate(tgt, lambda r: r['command'] == 'reject'),
        target_effective_reject=rate(tgt, lambda r: r['effective_reject']),
        target_bases_retained=sum(r['bases_in_window'] for r in keep_t),
        target_bases_rejected_prefix=sum(r['bases_in_window'] for r in tgt if r['effective_reject']),
        host_bases=sum(r['bases_in_window'] for r in host),
        total_bases=sum(r['bases_in_window'] for r in rows),
        complete_target_molecules=sum(1 for r in keep_t if r['reason'] == 'natural_end' and r['end'] is not None
                                      and r['end'] <= m['duration']),
        host_mean_pore_seconds=float(np.mean([r['pore_seconds'] for r in host if r['pore_seconds'] is not None])) if host else None,
        host_unblock_samples_p50=float(np.median(unblocked)) if any(unblocked) else None,
        command_latency_s=dict(zip(('p50', 'p95', 'p99'), np.percentile(latency, [50, 95, 99]).tolist())) if latency else None,
        failed_acks=sum(1 for r in rows if r['ack'] not in (None, 0)),
        client_stats=m.get('stats'), status=m.get('status'))
    out['yield_target_bp_per_channel_hour'] = out['target_bases_retained']/(m['channels']*hours)
    out['target_fraction_of_retained_bases'] = out['target_bases_retained']/max(1, out['target_bases_retained'] + sum(
        r['bases_in_window'] for r in host if not r['effective_reject']))
    out['target_fraction_of_all_bases'] = sum(r['bases_in_window'] for r in tgt)/max(1, out['total_bases'])
    host_eject, target_eject = out['host_effective_reject'], out['target_effective_reject']
    out['accuracy'] = (host_eject['events'] + len(tgt) - target_eject['events'])/max(1, len(rows))
    return out


def summarize_run(run):
    run = Path(run)
    m, rows = analyze(run)
    s = summarize(m, rows)
    (run/'summary.json').write_text(json.dumps(s, indent=2, default=str) + '\n')
    with (run/'read_outcomes.jsonl').open('w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    return s


METRICS = dict(enrichment=('Target enrichment', 1, 2), yield_target_bp_per_channel_hour=('Target bp / (channel h)', 1, 0),
               accuracy=('Accuracy (%)', 100, 2), host_eject=('Hosts ejected (%)', 100, 2),
               target_eject=('Targets ejected (%)', 100, 2), target_command_frr=('Target reject commands (%)', 100, 2),
               host_pore=('Host pore time (s)', 1, 3), latency_p50=('Command latency p50 (s)', 1, 3))


def run_metrics(s):
    return dict(accuracy=s['accuracy'], host_eject=s['host_effective_reject']['value'],
                target_eject=s['target_effective_reject']['value'], target_command_frr=s['target_command_reject']['value'],
                host_pore=s['host_mean_pore_seconds'], latency_p50=(s.get('command_latency_s') or {}).get('p50'),
                yield_target_bp_per_channel_hour=s['yield_target_bp_per_channel_hour'])


def aggregate(summaries, order=None):
    """summaries: run summaries (summary.json contents). Returns (rows, table) with table[(permille, strategy)] =
    {metric: (mean, sd) or None, 'n': runs}; enrichment needs the No-AS run of the same seed and target fraction."""
    rows = [dict(s, **run_metrics(s)) for s in summaries]
    base = {(r['seed'], r['target_permille']): r['yield_target_bp_per_channel_hour'] for r in rows if r['strategy'] == 'No-AS'}
    for r in rows:
        b = base.get((r['seed'], r['target_permille']))
        r['enrichment'] = r['yield_target_bp_per_channel_hour']/b if b else None
    order = order or sorted({r['strategy'] for r in rows})
    table = {}
    for p in sorted({r['target_permille'] for r in rows}):
        for s in order:
            rs = [r for r in rows if r['target_permille'] == p and r['strategy'] == s]
            if not rs:
                continue
            q = dict(n=len(rs), seeds=sorted(r['seed'] for r in rs))
            for k in METRICS:
                v = np.array([r[k] for r in rs if r[k] is not None], float)
                q[k] = (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else None) if len(v) else None
            table[(p, s)] = q
    return rows, table


def markdown(table, label=lambda s: (s, '')):
    """Enrichment table (strategies x target fractions) followed by one table of every metric per target fraction."""
    def cell(q, k):
        if q is None or q.get(k) is None: return '-'
        _, scale, digits = METRICS[k]
        mean, sd = q[k]
        text = f'{scale*mean:.{digits}f}' + ('' if sd is None else f' ± {scale*sd:.{digits}f}')
        return text if q['n'] > 1 else text + ' (n=1)'
    fractions = sorted({p for p, _ in table})
    strategies = list(dict.fromkeys(s for _, s in table))
    lines = ['## Target-base enrichment over nonselective sequencing (mean ± SD over seeds)', '',
             '| Method | Observation | ' + ' | '.join(f'{p/10:g}%' for p in fractions) + ' |',
             '|---|---|' + '---:|'*len(fractions)]
    for s in strategies:
        if s == 'No-AS': continue
        lines.append(f'| {label(s)[0]} | {label(s)[1]} | ' + ' | '.join(cell(table.get((p, s)), 'enrichment') for p in fractions) + ' |')
    keys = [k for k in METRICS if k != 'enrichment']
    lines += ['', '## Run metrics (mean ± SD over seeds)', '',
              '| Targets | Method | Observation | Runs | ' + ' | '.join(METRICS[k][0] for k in keys) + ' |',
              '|---|---|---|---:|' + '---:|'*len(keys)]
    for p in fractions:
        for s in strategies:
            q = table.get((p, s))
            if q is None: continue
            lines.append(f'| {p/10:g}% | {label(s)[0]} | {label(s)[1]} | {q["n"]} | ' + ' | '.join(cell(q, k) for k in keys) + ' |')
    return '\n'.join(lines) + '\n'
