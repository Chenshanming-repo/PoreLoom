"""Select and audit two actual read decisions; no inferred evidence scores.

Continue callbacks are reconstructed from the registered phase and saved terminal
step. These are action timelines, not recovered logit trajectories.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from signal_position_data import SERVICE, INTERVAL, PHASE_SEED, SKIP, ENDPOINT

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['poreloom', 'readcurrent', 'nanodeep_upstream', 'squigglenet']


def prepare():
    base = ROOT/'results/final_20261014/runs/main/sim_g1'
    paths = {m: base/f'{m}_r0/test'/('outcomes_0.4_nominal.npz' if m == 'poreloom'
             else 'outcomes_max_acc_0.4_measured.npz') for m in METHODS}
    arrays = {}
    for m, path in paths.items():
        with np.load(path, allow_pickle=False) as z:
            arrays[m] = {k: z[k] for k in z.files}
    pl = arrays['poreloom']
    fs = float(json.loads((base/'poreloom_r0/run.json').read_text())['fs'])
    candidates = pl['command'] & (pl['command_reject'] != pl['is_target'])
    for m, z in arrays.items():
        assert np.array_equal(z['read_id'], pl['read_id'])
        assert np.array_equal(z['is_target'], pl['is_target'])
        if m != 'poreloom':
            candidates &= z['command'] & (z['command_reject'] == z['is_target'])
            candidates &= z['decision_seconds'] < pl['decision_seconds']
    # Require at least 1.6 s before the final decision, so multiple looks are visible.
    candidates &= pl['decision_seconds'] > 1.6
    phases = np.random.default_rng(PHASE_SEED).random(len(candidates))*INTERVAL
    rows, cases = [], []
    for target, case in [(False, 'Host'), (True, 'Target')]:
        indices = np.flatnonzero(candidates & (pl['is_target'] == target))
        indices = sorted(indices, key=lambda i: (pl['decision_seconds'][i], pl['read_id'][i]))
        i = indices[len(indices)//2]
        rid = str(pl['read_id'][i])
        terminal = int(pl['steps'][i])
        arrival = phases[i] + np.arange(terminal+1)*INTERVAL
        observed = np.minimum(np.floor(arrival*fs).astype(int), ENDPOINT)-SKIP
        looks = np.flatnonzero(observed//48*48 >= 432)
        assert len(looks) >= 3 and looks[-1] == terminal
        assert abs(arrival[-1]+SERVICE-pl['decision_seconds'][i]) < 1e-12
        for k in looks:
            action = ('Reject' if pl['command_reject'][i] else 'Accept') if k == terminal else 'Continue'
            rows.append(dict(case=case, read_id=rid, method='poreloom', action=action,
                             time_seconds=float(arrival[k]+SERVICE), callback_step=int(k),
                             usable_samples=int(observed[k]), correct=bool(k == terminal)))
        for m in METHODS[1:]:
            z = arrays[m]
            rows.append(dict(case=case, read_id=rid, method=m,
                             action='Reject' if z['command_reject'][i] else 'Accept',
                             time_seconds=float(z['decision_seconds'][i]), callback_step=None,
                             usable_samples=3000, correct=False))
        cases.append(dict(case=case, read_id=rid, row_index=int(i), eligible_reads=len(indices),
                          poreloom_seconds=float(pl['pore_seconds'][i]),
                          baseline_pore_seconds={m: float(arrays[m]['pore_seconds'][i]) for m in METHODS[1:]},
                          timing_reconstruction_error_seconds=float(abs(arrival[-1]+SERVICE-pl['decision_seconds'][i]))))
    pd.DataFrame(rows).to_csv(ROOT/'figure_data/read_cases.csv', index=False, float_format='%.15g')
    sources = set(str(p.relative_to(ROOT)) for p in [*paths.values(), base/'poreloom_r0/run.json'])
    audit = dict(unit='sim_g1/r0', selection='Within each truth class, take the upper median by PoreLoom decision time (read ID breaks ties) among reads where all three 3000-sample baselines issue the wrong command earlier and PoreLoom issues the correct command after 1.6 s.',
                 interpretation='Post hoc illustrative cases, not prevalence estimates. Nonterminal eligible callbacks imply Continue; logit trajectories are not available and are not plotted.',
                 sources=[dict(path=p, sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()) for p in sorted(sources)],
                 interval_seconds=INTERVAL, phase_seed=PHASE_SEED, service_seconds=SERVICE, cases=cases)
    (ROOT/'review/read_case_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    return sources


if __name__ == '__main__':
    prepare()
