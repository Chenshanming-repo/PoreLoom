"""Run a trained PoreLoom bundle as the read-until client of one MinKNOW sequencing position (template).

Start a sequencing run in MinKNOW first (MinKNOW's own adaptive sampling off), then:

    python -m poreloom.minknow --bundle RUN/poreloom_sh_cp.pt --out LOG_DIR [--position MN12345] \\
        [--host localhost] [--port 9502] [--token TOKEN] [--device cuda:0] [--duration 3600] [--dry-run]

The client connects through the manager service (TLS; set MINKNOW_TRUSTED_CA to MinKNOW's rpc-certs/ca.crt when
the CA is not found automatically, and pass a developer API token with --token when MinKNOW requires one),
checks that the device sample rate equals the bundle's, reads every channel's calibration, and streams until
--duration seconds have passed, MinKNOW ends the acquisition, or Ctrl-C. --dry-run computes and logs every
decision without sending it. LOG_DIR receives manifest.json, decisions.jsonl, acks.jsonl and reads.jsonl.

This adapter has been exercised against the Icarust simulator only, not on a sequencing device.
"""
import argparse
import json
from pathlib import Path
import sys
import time

# int16 grid of the buffered samples: the R10 4 kHz simulation training contract used for the Icarust runs
# (experiments/20261014_final/icarust_models.json). The policy sees picoamperes; this only fixes the grid.
ADC_CONTRACT = (13.464453946637898, 1536.598389, 8192.)


def pick_position(positions, name):
    names = [p.name for p in positions]
    if name is None:
        if len(positions) == 1:
            return positions[0]
        raise ValueError(f'choose a position with --position: {", ".join(names) or "none found"}')
    for p in positions:
        if p.name == name:
            return p
    raise ValueError(f'position {name} not found; available: {", ".join(names) or "none"}')


def classification_ids(table, names):
    """table: {id: name} from AnalysisConfigurationService.get_read_classifications; ['all'] -> None."""
    if list(names) == ['all']:
        return None
    by_name = {v: k for k, v in table.items()}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(f'unknown read classifications {missing}; available: {", ".join(sorted(by_name))}')
    return {by_name[n] for n in names}


def device_calibration(connection, first_channel, last_channel):
    c = connection.device.get_calibration(first_channel=first_channel, last_channel=last_channel)
    n = last_channel - first_channel + 1
    if not c.has_calibration:
        raise ValueError('the device reports no calibration; calibrate the flow cell in MinKNOW first')
    if len(c.offsets) != n or len(c.pa_ranges) != n:
        raise ValueError(f'calibration covers {len(c.offsets)} channels, expected {n}')
    return dict(offsets=[float(x) for x in c.offsets], pa_ranges=[float(x) for x in c.pa_ranges],
                digitisation=float(c.digitisation))


def bank_capacity(channels):
    """Encoder state slots: one read per channel plus slots released lazily at the next step."""
    return max(2048, 2*int(channels))


def connect(host, port, position, token=None):
    from minknow_api.manager import Manager
    kwargs = {} if port is None else dict(port=port)
    if token:
        kwargs['developer_api_token'] = token        # minknow_api >= 5
    manager = Manager(host=host, **kwargs)
    pos = pick_position(list(manager.flow_cell_positions()), position)
    return manager, pos, pos.connect()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog='python -m poreloom.minknow', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bundle', type=Path, required=True, help='poreloom_sh_cp.pt from python -m poreloom.method')
    ap.add_argument('--encoder', type=Path, help='StreamR0 checkpoint when the path stored in the bundle is not valid here')
    ap.add_argument('--out', type=Path, required=True, help='new log directory')
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, help='manager port (minknow_api default when omitted)')
    ap.add_argument('--position', help='flow cell position name; required when the host has several')
    ap.add_argument('--token', help='MinKNOW developer API token')
    ap.add_argument('--device', default='cuda:0', help='torch device for the policy')
    ap.add_argument('--first-channel', type=int, default=1)
    ap.add_argument('--last-channel', type=int, help="default: the flow cell's channel count")
    ap.add_argument('--duration', type=float, help='seconds to stream (default: until the acquisition ends)')
    ap.add_argument('--unblock-duration', type=float, default=.1, help='seconds of reversed voltage per reject')
    ap.add_argument('--accept-classes', nargs='+', default=['strand', 'adapter'],
                    help="first-chunk read classifications eligible for decisions, or 'all'")
    ap.add_argument('--adc-contract', type=float, nargs=3, default=ADC_CONTRACT, metavar=('OFFSET', 'RANGE', 'DIGITISATION'))
    ap.add_argument('--capacity', type=int, help='encoder state slots (default: 2 x channels, at least 2048)')
    ap.add_argument('--chunk-size', type=int, default=300, help='minimum samples per live-read chunk')
    ap.add_argument('--stall-seconds', type=float, default=60., help='abort when no data arrives for this long')
    ap.add_argument('--dry-run', action='store_true', help='decide and log, but send no actions')
    return ap.parse_args(argv)


def main(argv=None):
    import torch
    from .client import LiveClient
    from .policy import PoreLoomLivePolicy
    from .signal import adc_mapping
    a = parse_args(argv)
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    manifest = dict(strategy='PoreLoom', template='poreloom.minknow (validated on Icarust only)', argv=sys.argv[1:],
                    started_wall=time.time(), dry_run=a.dry_run, duration=a.duration, unblock_duration=a.unblock_duration)

    def save():
        (a.out/'manifest.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n')
    try:
        manager, pos, conn = connect(a.host, a.port, a.position, a.token)
        manifest.update(host=a.host, port=a.port, position=pos.name, minknow=getattr(manager, 'core_version', None))
        import grpc
        try:
            run = conn.acquisition.get_current_acquisition_run()
        except grpc.RpcError:
            run = None
        if run is None or not run.run_id:
            raise RuntimeError(f'no acquisition is running on {pos.name}; start sequencing in MinKNOW first')
        last = a.last_channel or int(conn.device.get_flow_cell_info().channel_count)
        channels = last - a.first_channel + 1
        sample_rate = float(conn.device.get_sample_rate().sample_rate)
        classes = {int(k): v for k, v in conn.analysis_configuration.get_read_classifications().read_classifications.items()}
        accepted = classification_ids(classes, a.accept_classes)
        cal = device_calibration(conn, a.first_channel, last)
        contract = dict(offset=a.adc_contract[0], range=a.adc_contract[1], digitisation=a.adc_contract[2])
        gain, shift = adc_mapping(cal, contract)
        if a.device.startswith('cuda'):
            torch.cuda.set_device(a.device)
        policy = PoreLoomLivePolicy(a.bundle, a.device, contract, capacity=a.capacity or bank_capacity(channels),
                                    encoder=a.encoder)
        if abs(sample_rate - policy.fs) > 1e-6:
            raise ValueError(f'device samples at {sample_rate:g} Hz but the bundle was trained for {policy.fs:g} Hz')
        manifest.update(run_id=run.run_id, sample_rate=sample_rate, first_channel=a.first_channel, last_channel=last,
                        channels=channels, accepted_classes=None if accepted is None else sorted(accepted),
                        adc_contract=contract, calibration=dict(digitisation=cal['digitisation'],
                        offsets_range=[min(cal['offsets']), max(cal['offsets'])],
                        pa_ranges_range=[min(cal['pa_ranges']), max(cal['pa_ranges'])]),
                        policy=policy.pins, status='running')
        save()
        client = LiveClient(conn, policy, a.out, gain, shift, a.first_channel, last, duration=a.duration,
                            unblock_duration=a.unblock_duration, dry_run=a.dry_run, accepted=accepted,
                            stall_seconds=a.stall_seconds, chunk_size=a.chunk_size)
        manifest.update(client.run())
        manifest['policy_stats'] = policy.stats
    except BaseException as exc:
        manifest['status'] = f'failed: {type(exc).__name__}: {exc}'
        raise
    finally:
        manifest['finished_wall'] = time.time()
        save()
    print(json.dumps(dict(status=manifest['status'], out=str(a.out), **{k: manifest['stats'][k] for k in
                                                                    ('packets', 'actions_sent', 'decision_steps')})))
