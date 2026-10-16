"""Exact stopping training: minimize target commands subject to a host floor."""
from pathlib import Path
import copy
import json
import time
import numpy as np
import torch
from torch import nn
from .controllers import PolicyController
from .model import prefix_loss
from .risk_training import exact_expected_components
from ._policy_training import branch_cost_table, sample_balanced_ids
from .control_evaluation import prefix_scores, evaluate_policy
from .hostfloor import hostfloor_key


def conditional_hostfloor_loss(components, is_target, host_floor, dual):
    """Class means include environment-only natural endings (zero reject mass)."""
    target = is_target.bool()
    if not target.any() or target.all():
        raise ValueError('class conditional loss requires both classes')
    risk = components['command_reject'][target].mean()
    host_rate = components['effective_reject'][~target].mean()
    return risk + dual * (host_floor - host_rate), risk, host_rate



def train_hostfloor(model, train_data, dev_data, config, output, device='cpu'):
    """Initialize from softstop, freeze CNN, and select only on hard DEV replay."""
    cfg = copy.deepcopy(config)
    cfg.update(variant='hostfloor', objective='host_constrained_target_commands',
               target_frr_limit=None, deployment_mode='argmax', auxiliary_weight=.1)
    setting = cfg.setdefault('training', {}).setdefault('hostfloor', {})
    defaults = dict(epochs=20, patience=5, batch_size=128, head_only_epochs=2,
                    head_lr=1e-4, gru_lr=3e-5)
    for key, value in defaults.items():
        setting.setdefault(key, value)
    constraint = cfg['host_constraint']
    train_floor, dev_floor = float(constraint['train_floor']), float(constraint['dev_floor'])
    if not all(0 <= floor <= 1 for floor in (train_floor, dev_floor)):
        raise ValueError('host floors must lie in [0,1]')
    dual = float(constraint.setdefault('dual_initial', 1.))
    dual_lr = float(constraint.setdefault('dual_lr', 10.))
    if dual < 0 or dual_lr < 0:
        raise ValueError('host dual and dual_lr must be nonnegative')
    if setting['batch_size'] < 2 or setting['epochs'] < 0 or setting['patience'] < 1:
        raise ValueError('invalid hostfloor training settings')
    for data in (train_data, dev_data):
        if data.features is None or data.valid.shape[1] != cfg['max_windows']:
            raise ValueError('hostfloor requires fixed-budget cached features')
        if not data.is_target.any() or data.is_target.all():
            raise ValueError('hostfloor training requires both classes in TRAIN and DEV')
    if getattr(dev_data, 'split', None) == 'test':
        raise ValueError('TEST cannot select hostfloor checkpoints')
    if cfg.get('backbone_train_mode', 'frozen_encoder') != 'frozen_encoder':
        raise ValueError('hostfloor requires frozen_encoder')
    checkpoint = cfg.get('controller_checkpoint')
    if not checkpoint or not Path(checkpoint).is_file():
        raise ValueError('hostfloor requires an existing softstop controller_checkpoint')
    cp = torch.load(checkpoint, map_location=device, weights_only=False)
    if cp.get('variant', cp.get('config', {}).get('variant')) != 'softstop':
        raise ValueError('hostfloor requires a softstop initialization checkpoint')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('hostfloor output must be empty; historical results are read-only')
    (output/'checkpoints').mkdir(parents=True)
    seed = cfg.setdefault('seed', 20261010)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to(device)
    model.load_state_dict(cp['model_state'])
    model.freeze_encoder()
    frozen = {name: param.detach().clone() for name, param in model.state_dict().items()
              if name.startswith(('encoder.', 'projection.'))}
    controller = PolicyController(max_windows=cfg['max_windows']).to(device)
    controller.load_state_dict(cp['controller_state'])
    recurrent = list(model.gru.parameters()) + list(model.head.parameters())
    params = list(controller.parameters()) + recurrent
    optimizer = torch.optim.AdamW([
        dict(params=list(controller.parameters()), lr=setting['head_lr']),
        dict(params=recurrent, lr=setting['gru_lr'])], weight_decay=1e-4)
    _, train_available, _, train_branches, train_ends = branch_cost_table(train_data, cfg, return_details=True)
    _, dev_available, _, dev_branches, dev_ends = branch_cost_table(dev_data, cfg, return_details=True)
    eligible = np.arange(len(train_data.valid))
    history = []
    best_key = None
    stale = 0
    started = time.time()
    (output/'config.json').write_text(json.dumps(cfg, indent=2))

    def tensors(values, ids):
        return {key: torch.as_tensor(value[ids], device=device) for key, value in values.items()}

    def components(states, ids, available, branches, ends):
        return exact_expected_components(controller(states), tensors(branches, ids),
            torch.as_tensor(available[ids], device=device), tensors(ends, ids))

    def cohort_expectation(data, available, branches, ends, states=None):
        # Aggregate numerators and denominators separately, not minibatch means.
        model.eval(); controller.eval()
        target_sum = host_sum = 0.
        target_count = host_count = 0
        mass_error = 0.
        with torch.no_grad():
            for start in range(0, len(data.valid), 256):
                ids = np.arange(start, min(start+256, len(data.valid)))
                target = torch.as_tensor(data.is_target[ids], device=device)
                if states is None:
                    e = torch.as_tensor(np.asarray(data.features[ids]).copy(), dtype=torch.float32, device=device)
                    valid = torch.as_tensor(data.valid[ids], device=device)
                    batch_states = model.forward_states(e, valid)
                else:
                    batch_states = states[ids].to(device)
                comp = components(batch_states, ids, available, branches, ends)
                target_sum += float(comp['command_reject'][target].sum())
                host_sum += float(comp['effective_reject'][~target].sum())
                target_count += int(target.sum()); host_count += int((~target).sum())
                mass_error = max(mass_error, float((comp['probability_mass'] - 1).abs().max()))
        if mass_error > 1e-5:
            raise RuntimeError('exact policy probability mass is not conserved')
        return target_sum / target_count, host_sum / host_count, mass_error

    def validation():
        model.eval(); controller.eval()
        states, probabilities = prefix_scores(model, dev_data, device)
        metrics = evaluate_policy(model, controller, dev_data, cfg, device=device,
            mode='argmax', states=states, probabilities=probabilities)
        risk, host, error = cohort_expectation(dev_data, dev_available, dev_branches, dev_ends, states)
        metrics.update(soft_target_command_frr=risk, soft_host_effective_reject_rate=host,
            soft_probability_mass_max_error=error, host_floor=dev_floor,
            host_constraint_met=metrics['host_effective_reject_rate'] >= dev_floor)
        return metrics

    def persist(epoch, metrics, row):
        nonlocal best_key, stale
        for name, original in frozen.items():
            if not torch.equal(model.state_dict()[name], original):
                raise RuntimeError('frozen CNN/projection weights changed')
        key = hostfloor_key(metrics, dev_floor)
        improved = best_key is None or key < best_key
        if improved:
            best_key = key; stale = 0
        else:
            stale += 1
        history.append(row)
        state = dict(model_state=model.state_dict(), controller_state=controller.state_dict(),
            config=cfg, variant='hostfloor', epoch=epoch, dev_metrics=metrics, dual=dual,
            host_constraint_met=metrics['host_constraint_met'],
            selection_status='feasible' if metrics['host_constraint_met'] else 'diagnostic_infeasible',
            selection_key=key, best_key=best_key, stale=stale,
            optimizer_state=optimizer.state_dict(), history=history,
            numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state())
        if str(device).startswith('cuda'):
            state['cuda_rng'] = torch.cuda.get_rng_state(device)
        torch.save(state, output/'checkpoints'/f'epoch_{epoch:03d}.pt')
        torch.save(state, output/'last.pt')
        if improved:
            torch.save(state, output/'selected.pt')
        (output/'history.json').write_text(json.dumps(history, indent=2))

    metrics = validation()
    persist(0, metrics, dict(epoch=0, stage='initialization', dual=dual, dev=metrics))
    for epoch in range(1, setting['epochs']+1):
        head_only = epoch <= setting['head_only_epochs']
        for param in recurrent:
            param.requires_grad_(not head_only)
        model.train(not head_only); controller.train()
        sums = dict(loss=0., target_command_risk=0., host_effective_rate=0., prefix_ce=0.)
        count = 0
        used_dual = dual
        for ids, _ in sample_balanced_ids(train_data.is_target, eligible, setting['batch_size'], rng):
            e = torch.as_tensor(np.asarray(train_data.features[ids]).copy(), dtype=torch.float32, device=device)
            valid = torch.as_tensor(train_data.valid[ids], device=device)
            target = torch.as_tensor(train_data.is_target[ids], device=device)
            states = model.forward_states(e, valid)
            comp = components(states, ids, train_available, train_branches, train_ends)
            objective, risk, host = conditional_hostfloor_loss(comp, target, train_floor, used_dual)
            ce = prefix_loss(model.head(states), target.long(), valid) if valid.any() else states.sum()*0
            loss = objective + .1*ce
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.); optimizer.step()
            for key, value in zip(sums, (loss, risk, host, ce)):
                sums[key] += float(value.detach())*len(ids)
            count += len(ids)
        risk, host, error = cohort_expectation(train_data, train_available, train_branches, train_ends)
        dual = max(0., used_dual + dual_lr*(train_floor-host))
        metrics = validation()
        row = dict(epoch=epoch, train={key: value/count for key, value in sums.items()},
            train_target_command_frr=risk, train_host_rate=host,
            train_probability_mass_max_error=error, dual_used=used_dual, dual=dual,
            head_only=head_only, dev=metrics, elapsed_seconds=time.time()-started)
        persist(epoch, metrics, row)
        print(row, flush=True)
        if stale >= setting['patience']:
            break
    return output/'selected.pt'
