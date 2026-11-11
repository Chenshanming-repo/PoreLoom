"""poreloom.icarust: closed-loop simulation support (no simulator needed).

Run profile and plan of the paper runs, reference rebuilding, the native 0.5 rule and window contracts, the client's
chunk bookkeeping, run metrics, and live strategies against their offline replay on the bundled example reads."""
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from poreloom.icarust import analysis, references, simulator, strategies
from poreloom.icarust.cli import cells, main

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- simulator files
def test_profile_matches_the_paper_runs():
    text = simulator.profile_text('/dev/shm/x', 20331101, 1340, 10, '/refs')
    assert 'random_seed = 20331101\n' in text and 'experiment_duration_set = 1340\n' in text
    assert 'pore_type = "R10"' in text and 'sample_rate = 4000\n' in text and 'sequencing_speed = 400\n' in text
    assert 'break_read_ms = 400\n' in text and text.count('mean_read_length = 8000') == 2
    host, target = text.split('[[sample]]')[1:]
    assert 'name = "host"' in host and 'input_genome = "/refs/host.fa"' in host and 'weight = 990' in host
    assert 'name = "target"' in target and 'input_genome = "/refs/target.fa"' in target and 'weight = 10' in target


def test_config_uses_the_next_port_for_the_position():
    assert simulator.config_text('/tls', 25100, 512) == ('[TLS]\ncert-dir = /tls\n[PORTS]\nmanager = 25100\nposition = 25101\n'
                                                         '[SEQUENCER]\nchannels = 512\n')


@pytest.mark.parametrize('layout', ['', 'icarust_writerfix/'])
def test_locate_finds_both_layouts_and_lists_missing_files(tmp_path, layout):
    (tmp_path/layout/'src_tree').mkdir(parents=True)
    ic = simulator.Icarust.locate(tmp_path)
    assert ic.source == (tmp_path/layout/'src_tree').resolve()
    assert ic.binary == (tmp_path/layout/'target/release/icarust').resolve()
    with pytest.raises(FileNotFoundError, match='python -m poreloom.icarust references'):
        ic.check()


def test_setup_certificate_is_created_once_and_checked_for_expiry(tmp_path):
    import shutil
    if shutil.which('openssl') is None:
        pytest.skip('openssl is not installed')
    assert simulator.make_tls(tmp_path/'tls', days=2) == 'created'
    assert simulator.certificate_expired(tmp_path/'tls'/'server.crt') is False
    assert (tmp_path/'tls'/'ca.crt').read_bytes() == (tmp_path/'tls'/'server.crt').read_bytes()
    assert simulator.make_tls(tmp_path/'tls') == 'kept'
    assert simulator.certificate_expired(tmp_path/'missing.crt') is None


# ---------------------------------------------------------------- matrix plan
def test_plan_order_is_the_paper_matrix():
    plan = cells(['No-AS', 'PoreLoom', 'readcurrent', 'nanodeep', 'squigglenet'], (20331101, 20331102, 20331103), (10, 100, 300), 3)
    assert [c['name'] for c in plan[:6]] == ['30pct_20331101_No-AS', '10pct_20331101_readcurrent', '10pct_20331101_No-AS',
                                            '1pct_20331101_No-AS', '1pct_20331101_PoreLoom', '1pct_20331101_readcurrent']
    assert len(plan) == 45 and len({c['name'] for c in plan}) == 45 and [c['lane'] for c in plan[:4]] == [0, 1, 2, 0]


def test_cli_help_and_lane_parsing(capsys):
    with pytest.raises(SystemExit) as e:
        main(['--help'])
    assert e.value.code == 0 and 'matrix' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(['matrix', '--models', 'm.json', '--out', 'o', '--icarust', 'i', '--lanes', 'cuda:0'])


# ---------------------------------------------------------------- references
def test_shipped_blocks_are_the_paper_references():
    spec = references.load_blocks()
    assert len(spec['samples']['host']) == 200 and len(spec['samples']['target']) == 13
    blocks = [tuple(b) for rows in spec['samples'].values() for b in rows]
    assert len(set(blocks)) == 213 and all(end - start == 100_000 for _, _, start, end in blocks)
    assert {s['id'] for s in spec['sources']} == {b[0] for b in blocks}


def test_references_are_cut_uppercased_and_rewrapped(tmp_path):
    rng = np.random.default_rng(0)
    chrom = ''.join(rng.choice(list('ACGTacgt'), 1000)); plasmid = ''.join(rng.choice(list('ACGT'), 300))
    with gzip.open(tmp_path/'human.fna.gz', 'wt') as f:
        f.write('>chrX other words\n' + '\n'.join(chrom[i:i + 60] for i in range(0, 1000, 60)) + '\n>chrY\nACGT\n')
    (tmp_path/'bug.fa').write_text('>ENA|P1|P1.1 a plasmid\n' + '\n'.join(plasmid[i:i + 70] for i in range(0, 300, 70)) + '\n')
    spec = dict(line_width=80, samples=dict(host=[['H', 'chrX', 590, 790], ['H', 'chrX', 0, 120]],
                                            target=[['Bug one', 'ENA|P1|P1.1', 10, 300]]))
    manifest = references.build([tmp_path/'human.fna.gz', tmp_path/'bug.fa'], tmp_path/'out', blocks=spec)
    host = (tmp_path/'out'/'host.fa').read_text().split('\n')
    assert host[0] == '>H|chrX|590-790' and len(host[1]) == 80
    assert ''.join(host[1:4]) == chrom[590:790].upper() and host[4] == '>H|chrX|0-120'
    assert (tmp_path/'out'/'target.fa').read_text().startswith('>Bug_one|ENA|P1|P1.1|10-300\n' + plasmid[10:90] + '\n')
    assert manifest['samples']['host']['records'] == 2 and manifest['samples']['target']['bases'] == 290
    with pytest.raises(ValueError, match='not found'):
        references.build([tmp_path/'bug.fa'], tmp_path/'out2', blocks=spec)
    with pytest.raises(ValueError, match='outside'):
        references.build([tmp_path/'bug.fa'], tmp_path/'out3', blocks=dict(samples=dict(target=[['B', 'ENA|P1|P1.1', 250, 400]])))


# ---------------------------------------------------------------- baselines
def test_native_rule_is_torch_argmax_with_ties_to_class_zero():
    torch = pytest.importorskip('torch')
    logits = torch.tensor(np.random.default_rng(1).normal(size=(200, 2)).round(1), dtype=torch.float32)
    logits[:20, 1] = logits[:20, 0]                                         # exact ties
    for host in (0, 1):
        margin = (logits[:, host] - logits[:, 1 - host]).numpy()
        assert np.array_equal(strategies.native_reject(margin, host), (logits.argmax(1) == host).numpy())


def test_window_contracts_switch_and_restore():
    from poreloom.baselines import preprocessing as pp
    strategies.set_window(6000)
    assert pp.WINDOW == 6000 and all((c['raw_start'], c['raw_end']) == (1500, 7500) for c in pp.CONTRACTS.values())
    strategies.set_window(3000)
    assert pp.WINDOW == 3000 and (pp.CONTRACTS['nanodeep']['raw_start'], pp.CONTRACTS['nanodeep']['raw_end']) == (1000, 4000)
    assert (pp.CONTRACTS['squigglenet']['raw_start'], pp.CONTRACTS['squigglenet']['raw_end']) == (1500, 4500)
    assert strategies.parse('nanodeep_w6000') == ('nanodeep', 6000)
    with pytest.raises(ValueError):
        strategies.parse('cafes')


def test_models_config_resolves_relative_paths(tmp_path):
    (tmp_path/'m.json').write_text(json.dumps({'PoreLoom': 'runs/p/poreloom_sh_cp.pt', 'nanodeep': '/abs/nd', 'note': 'x'}))
    assert strategies.models_config(tmp_path/'m.json') == {'PoreLoom': str((tmp_path/'runs/p/poreloom_sh_cp.pt').resolve()),
                                                           'nanodeep': '/abs/nd'}
    (tmp_path/'bad.json').write_text(json.dumps({'PoreLoom-B': 'x'}))
    with pytest.raises(ValueError, match='unknown'):
        strategies.models_config(tmp_path/'bad.json')


# ---------------------------------------------------------------- client bookkeeping
class Every1000:
    def needed(self, s):
        return 1000*(s['k'] + 1) if s['k'] < 2 else None


def packet(number, start, chunk_start, samples, read_id='r'):
    return SimpleNamespace(raw_data=np.asarray(samples, '<i2').tobytes(), number=number, id=read_id,
                           start_sample=start, chunk_start_sample=chunk_start)


def test_client_buffers_chunks_by_sample_position(tmp_path):
    from poreloom.icarust.client import Client
    cal = dict(offset=0., range=1., digitisation=1.)
    c = Client(tmp_path, Every1000(), 4, 1, 25101, 1, cal, cal, tmp_path/'none.crt')
    c.on_packet(1, packet(5, 100, 100, np.arange(600)), 0.)
    c.on_packet(1, packet(5, 100, 500, np.arange(400, 1100)), .1)        # repeats samples 400-599
    s = c.reads[1]
    assert s['received'] == 1100 and np.array_equal(s['buf'][:1100], np.arange(1100)) and c.stats['overlaps'] == 1
    assert c.ready.get_nowait() is s and s['pending']
    c.on_packet(1, packet(4, 0, 0, np.ones(10)), .2)                       # stale packet of an older read
    assert c.reads[1] is s
    c.on_packet(2, packet(1, 0, 50, np.ones(10)), .3)                      # first chunk after a gap
    assert c.reads[2]['decided'] == 'gap' and c.stats['gaps'] == 1
    c.on_packet(1, packet(6, 2000, 2000, np.ones(10), 'next'), .4)         # a new read replaces the old one
    assert c.reads[1]['read_id'] == 'next'
    for f in c.logs.values(): f.close()
    assert json.loads((tmp_path/'client_reads.jsonl').read_text())['closed'] == 'replaced'


# ---------------------------------------------------------------- metrics
def fake_run(path, strategy, seed, permille, target_reject=0, host_reject=2):
    """Two hosts and two targets start in the window; one more read starts after it; one host never ends."""
    path.mkdir(parents=True)
    (path/'manifest.json').write_text(json.dumps(dict(strategy=strategy, seed=seed, target_permille=permille, channels=2,
                                                      duration=100, status='complete')))
    ev, dec = [], []
    reads = [(1, 1, 'host', 10.), (2, 1, 'host', 12.), (1, 2, 'target', 20.), (2, 2, 'target', 21.), (1, 3, 'host', 130.)]
    for i, (ch, num, sample, t) in enumerate(reads):
        ev.append(dict(event='read_started', channel=ch, read_number=num, read_id=f'r{i}', sample=sample,
                       monotonic_seconds=t, total_samples=40000, prefix_samples=1000))
        eject = (sample == 'host' and i < host_reject) or (sample == 'target' and i - 2 < target_reject)
        if eject:
            ev.append(dict(event='unblock_accepted', channel=ch, read_number=num, produced_samples=4000))
            ev.append(dict(event='read_finalized', channel=ch, read_number=num, monotonic_seconds=t + 1, produced_samples=4000,
                           reason='unblock'))
            dec.append(dict(channel=ch, read_number=num, action='reject', current=True, k=1, ready_at=t + .5, sent=t + .6,
                            action_id=f'a{i}'))
        elif i != 1:
            ev.append(dict(event='read_finalized', channel=ch, read_number=num, monotonic_seconds=t + 10, produced_samples=40000,
                           reason='natural_end'))
    (path/'sim').mkdir()
    (path/'sim'/'e7_engineering_audit.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in ev))
    (path/'decisions.jsonl').write_text(''.join(json.dumps(d) + '\n' for d in dec))
    (path/'acks.jsonl').write_text(''.join(json.dumps(dict(action_id=d['action_id'], response=0, at=0)) + '\n' for d in dec))
    return path


def test_run_summary_counts_ejections_bases_and_censored_reads(tmp_path):
    s = analysis.summarize_run(fake_run(tmp_path/'r', 'PoreLoom', 1, 10, host_reject=1))
    assert (s['reads'], s['host_reads'], s['target_reads']) == (4, 2, 2)
    assert s['host_effective_reject'] == dict(value=1., lower=.5, upper=1., events=1, censored=1, n=2)
    assert s['target_bases_retained'] == 2*(40000 - 1000)/10 and s['target_effective_reject']['events'] == 0
    assert s['accuracy'] == .75 and s['command_latency_s']['p50'] == pytest.approx(.1)
    assert s['yield_target_bp_per_channel_hour'] == pytest.approx(7800/(2*100/3600))
    assert len((tmp_path/'r'/'read_outcomes.jsonl').read_text().splitlines()) == 4


def test_enrichment_is_relative_to_no_as_of_the_same_seed_and_fraction(tmp_path):
    runs = [analysis.summarize_run(fake_run(tmp_path/f'{s}_{seed}', s, seed, 10, target_reject=t))
            for seed in (1, 2) for s, t in (('No-AS', 0), ('PoreLoom', 1))]
    rows, table = analysis.aggregate(runs, ['No-AS', 'PoreLoom'])
    assert table[(10, 'PoreLoom')]['enrichment'] == (pytest.approx(.5), pytest.approx(0.))
    assert table[(10, 'No-AS')]['n'] == 2 and '| PoreLoom | Budget 6000 | 0.50 ± 0.00 |' in analysis.markdown(table, strategies.label)


# ---------------------------------------------------------------- live == offline on the example reads
@pytest.mark.skipif(not (REPO/'example'/'models'/'poreloom'/'head.pt').exists(), reason='example/ is not part of this checkout')
def test_live_strategies_match_their_offline_replay_on_the_example_reads(tmp_path):
    pytest.importorskip('torch')
    from poreloom.icarust.check import check
    from poreloom.paths import THIRD_PARTY
    upstream = dict(readcurrent='ReadCurrent/models/ReadCurrent.py', nanodeep='NanoDeep/read_deep/model/nanodeep.py',
                    squigglenet='SquiggleNet/model.py')          # baseline architectures (example/run_example.py fetches them)
    models = {name: str(REPO/'example'/'models'/name) for name, rel in upstream.items() if (THIRD_PARTY/rel).exists()}
    models['PoreLoom'] = str(REPO/'example'/'models'/'poreloom')
    torch = pytest.importorskip('torch')
    threads = torch.get_num_threads(); torch.set_num_threads(min(4, threads))
    try:
        result = check(models, REPO/'example'/'data', list(models), 'cpu', n=60, workdir=tmp_path)
    finally:
        torch.set_num_threads(threads)
    for name, r in result.items():
        assert r['compared'] > 30 and r['agree'] == r['compared'], (name, r)
