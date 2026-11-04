"""Recompute fixed-window ACC on a shared raw-length subset of archived TEST reads.

Only the reporting population changes. Saved decisions, DEV-B selections and
training runs are retained; reads ending before command completion still count.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

from result_archive import ROOT, ARCHIVE
FINAL = ARCHIVE / 'results/final_20261014'
ENDPOINT = 7500
METHODS = ['poreloom', 'readcurrent', 'nanodeep_upstream', 'squigglenet']


def family(dataset):
    return ('Simulated' if dataset.startswith('sim') else 'R9' if dataset == 'r9'
            else 'R10 mixed' if dataset.startswith('r10') else 'R10 sputum')


def prepare():
    sources, coverage, results, metric_rows, checks = set(), [], [], [], []
    report_path = FINAL / 'win6000/summary_zh.json'
    sources.add(str(report_path.relative_to(ARCHIVE)))
    report = json.loads(report_path.read_text())
    units = sorted({(r['dataset'], r['rep']) for r in report})
    metadata = {}
    for dataset in sorted({ds for ds, rep in units}):
        for split in ['train', 'test']:
            path = ARCHIVE / 'provenance/legacy_story/cache' / dataset / split / 'metadata.npz'
            sources.add(str(path.relative_to(ARCHIVE)))
            with np.load(path, allow_pickle=False) as z:
                order = np.argsort(z['read_id'])
                ids, target, length = (z[key][order] for key in ['read_id', 'is_target', 'raw_length'])
            target = target.astype(bool)
            assert len(np.unique(ids)) == len(ids)
            eligible = length >= ENDPOINT
            if split == 'test':
                metadata[dataset] = (ids, target, eligible)
            row = dict(dataset=dataset, family=family(dataset), split=split,
                       raw_endpoint=ENDPOINT, sampling_rate_hz=5000 if dataset.startswith('sputum') else 4000,
                       n_reads=len(ids), n_eligible=int(eligible.sum()))
            for label, mask in [('host', ~target), ('target', target)]:
                row.update({f'n_{label}': int(mask.sum()),
                            f'n_{label}_eligible': int((mask & eligible).sum()),
                            f'n_{label}_short': int((mask & ~eligible).sum()),
                            f'{label}_coverage': float(eligible[mask].mean()),
                            f'{label}_median_samples': float(np.median(length[mask]))})
            row['length_only_acc_ceiling'] = 1 - row['n_host_short'] / len(ids)
            coverage.append(row)

    for dataset, rep in units:
        ids, target, eligible = metadata[dataset]
        unit = f'{dataset}/r{rep}'
        selected_ids = ids[eligible]
        subset_hash = hashlib.sha256('\n'.join(selected_ids).encode()).hexdigest()
        for method in METHODS:
            for variant in (['streaming'] if method == 'poreloom' else ['w3000', 'w6000']):
                folder = 'win6000' if variant == 'w6000' else 'runs/main'
                filename = ('outcomes_0.4_nominal.npz' if method == 'poreloom'
                            else 'outcomes_max_acc_0.4_measured.npz')
                path = FINAL / folder / dataset / f'{method}_r{rep}' / 'test' / filename
                sources.add(str(path.relative_to(ARCHIVE)))
                with np.load(path, allow_pickle=False) as z:
                    order = np.argsort(z['read_id'])
                    assert np.array_equal(ids, z['read_id'][order]), (unit, method, variant)
                    assert np.array_equal(target, z['is_target'][order]), (unit, method, variant)
                    rejected, command = (z[key][order].astype(bool) for key in ['command_reject', 'command'])
                    pore = z['pore_seconds'][order]
                correct = rejected != target
                short_host = ~eligible & ~target
                if variant == 'w6000':
                    assert not command[~eligible].any(), (unit, method)
                row = dict(unit=unit, dataset=dataset, family=family(dataset), replicate=rep,
                           method=method, variant=variant, raw_endpoint=ENDPOINT,
                           n_all=len(ids), n_eligible=int(eligible.sum()),
                           n_host_eligible=int((eligible & ~target).sum()),
                           n_target_eligible=int((eligible & target).sum()),
                           n_correct_all=int(correct.sum()), n_correct_eligible=int(correct[eligible].sum()),
                           acc_all=float(correct.mean()), acc_eligible=float(correct[eligible].mean()),
                           n_command_all=int(command.sum()), n_command_eligible=int(command[eligible].sum()),
                           short_host_rejection=float(rejected[short_host].mean()) if short_host.any() else np.nan,
                           subset_sha256=subset_hash)
                results.append(row)
                for scope, mask in [('all', np.ones(len(ids), dtype=bool)),
                                    ('ge7500', eligible), ('lt7500', ~eligible)]:
                    host_mask, target_mask = mask & ~target, mask & target
                    metric_rows.append(dict(unit=unit, dataset=dataset, family=family(dataset), replicate=rep,
                                            method=method, variant=variant, scope=scope,
                                            n_reads=int(mask.sum()), n_host=int(host_mask.sum()), n_target=int(target_mask.sum()),
                                            n_correct=int(correct[mask].sum()), n_command=int(command[mask].sum()),
                                            acc=float(correct[mask].mean()),
                                            host_reject=float(rejected[host_mask].mean()),
                                            frr=float(rejected[target_mask].mean()),
                                            host_pore=float(pore[host_mask].mean()), subset_sha256=subset_hash if scope == 'ge7500' else ''))
                if variant == 'w6000':
                    config_path = FINAL / folder / dataset / f'{method}_r{rep}' / 'fit/config.json'
                    sources.add(str(config_path.relative_to(ARCHIVE)))
                    config = json.loads(config_path.read_text())
                    training = next(r for r in coverage if r['dataset'] == dataset and r['split'] == 'train')
                    assert config['eligible_train'] == training['n_eligible']
                    checks.append(dict(unit=unit, method=method, test_ids_and_labels_match=True,
                                       no_short_read_commands=True, eligible_train=config['eligible_train'],
                                       subset_sha256=subset_hash))
    result, coverage, metric_rows = pd.DataFrame(results), pd.DataFrame(coverage), pd.DataFrame(metric_rows)
    means = result.groupby(['family', 'method', 'variant'], sort=False)[['acc_all', 'acc_eligible']].mean().reset_index()
    long = metric_rows[metric_rows.scope == 'ge7500'].copy()
    matched_main = long[(long.method == 'poreloom') | (long.variant == 'w6000')]
    matched_windows = long[long.method != 'poreloom']
    metric_means = metric_rows.groupby(['family', 'method', 'variant', 'scope'], sort=False)[['acc', 'host_reject', 'frr', 'host_pore']].mean().reset_index()
    for name, data in [('length_acc', result), ('length_acc_means', means), ('length_eligibility', coverage),
                       ('length_metrics', metric_rows), ('length_metrics_means', metric_means),
                       ('main_length_matched', matched_main), ('window_length_matched', matched_windows)]:
        data.to_csv(ROOT / 'figure_data' / (name + '.csv'), index=False, float_format='%.15g')
    audit = dict(raw_endpoint=ENDPOINT, outcome='command-level ACC, host rejection, target FRR, and host pore occupancy',
                 population='Same TEST read IDs with raw_length >= 7500 for all seven model/window configurations within each unit; no class rebalancing.',
                 decisions='Frozen archived decisions and DEV-B-selected operating points; no TEST tuning or retraining.',
                 short_read_policy='Main operational comparisons include all TEST reads; length-restricted results are an additional control.',
                 remaining_timing_effect='Length-eligible reads without a completed command remain in the conditional denominator.',
                 rows=len(result), units=len(units), checks=checks,
                 sources=[dict(path=p, sha256=hashlib.sha256((ARCHIVE / p).read_bytes()).hexdigest()) for p in sorted(sources)])
    (ROOT / 'review/length_eligibility_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return sources


if __name__ == '__main__':
    prepare()
