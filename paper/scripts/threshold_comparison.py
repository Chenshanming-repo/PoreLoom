"""Reproduce the final supplement's threshold comparison from completed runs.

Both decisions reuse the original scores, arrival phases, model weights and
measured processing delays. No model fitting or inference is performed.
"""
import hashlib
import json
import math
import pandas as pd
from result_archive import ROOT, ARCHIVE


def prepare():
    main = pd.read_csv(ROOT / 'figure_data/length_metrics.csv')
    records, sources = [], []
    for r in main.to_dict('records'):
        if r['scope'] != 'all' or r['method'] == 'poreloom':
            continue
        dataset, rep = r['unit'].split('/')
        folder = 'win6000' if r['variant'] == 'w6000' else 'runs/main'
        run = ARCHIVE / 'results/final_20261014' / folder / dataset / f"{r['method']}_{rep}"
        objects = []
        for path in [run / 'selection.json', run / 'test/metrics.json']:
            sources.append({'source': str(path.relative_to(ARCHIVE)),
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            objects.append(json.loads(path.read_text()))
        selection, metrics = objects
        pair = {x['rule']: x for x in metrics['rows'] if x.get('rule') in {'native', 'max_acc'}
                and x['interval'] == .4 and x['service'] == 'measured' and x['group'] == 'pooled'}
        assert set(pair) == {'native', 'max_acc'}
        native, selected = pair['native'], pair['max_acc']
        assert all(native[k] == selected[k] for k in ['n', 'n_host', 'n_target'])
        assert selection['rules']['native']['threshold'] is None
        assert selection['rules']['max_acc']['dev_metrics']['acc'] >= selection['rules']['native']['dev_metrics']['acc']
        row = {k: r[k] for k in ['family', 'unit', 'method']}
        row.update(window=int(r['variant'][1:]),
                   threshold=1 / (1 + math.exp(-selection['rules']['max_acc']['threshold'])),
                   n_host=selected['n_host'], n_target=selected['n_target'],
                   service_ms=selection['service_seconds'] * 1000)
        for key, metric in [('acc', 'acc'), ('host_reject', 'host_command_reject_rate'),
                            ('frr', 'target_command_frr'), ('host_pore', 'host_mean_pore_seconds')]:
            assert math.isclose(selected[metric], float(r[key]), rel_tol=1e-12, abs_tol=1e-12)
            row[f'{key}_default'], row[f'{key}_selected'] = native[metric], selected[metric]
        row['dacc'] = row['acc_selected'] - row['acc_default']
        row['dpore'] = row['host_pore_selected'] - row['host_pore_default']
        records.append(row)
    assert len(records) == 72
    data = pd.DataFrame(records)
    means = data.groupby(['family', 'method', 'window'], sort=False).mean(numeric_only=True).reset_index()
    data.to_csv(ROOT / 'figure_data/threshold_comparison.csv', index=False, float_format='%.15g')
    means.to_csv(ROOT / 'figure_data/threshold_comparison_means.csv', index=False, float_format='%.15g')
    audit = dict(pairs=len(data), unit_accuracy_improved=int((data.dacc > 0).sum()),
                 unit_accuracy_equal=int((data.dacc == 0).sum()), unit_accuracy_reduced=int((data.dacc < 0).sum()),
                 mean_accuracy_improved=int((means.dacc > 0).sum()), sources=sources,
                 definition='DEV-B-selected minus fixed 0.5; same TEST scores and measured delays.')
    assert [audit[k] for k in ['unit_accuracy_improved', 'unit_accuracy_equal',
                               'unit_accuracy_reduced', 'mean_accuracy_improved']] == [50, 1, 21, 21]
    (ROOT / 'provenance/threshold_comparison_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    return data


if __name__ == '__main__':
    prepare()
