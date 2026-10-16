"""Controlled resource ablation with hard DEV protection and fixed representations."""
from pathlib import Path
import copy
import json
import time
import hashlib
import math
import numpy as np
import torch
from torch import nn
from .controllers import PolicyController
from .model import prefix_loss
from .risk_training import exact_expected_components
from ._policy_training import branch_cost_table, sample_balanced_ids
from .control_evaluation import prefix_scores, evaluate_policy
from .hostfloor import hostfloor_key
from .hostfloor_training import conditional_hostfloor_loss


def _positive_scales(*values):
    if any(not math.isfinite(float(v)) or float(v) <= 0 for v in values):
        raise ValueError('resource normalizers must be finite and positive')


def resource_loss(components, is_target, host_floor, dual, alpha, beta,
                  compute_scale, pore_scale):
    """Old conditional objective plus all-read compute and host-only pore means."""
    _positive_scales(compute_scale, pore_scale)
    if any(not math.isfinite(float(v)) or float(v) < 0 for v in (alpha, beta)):
        raise ValueError('resource weights must be finite and nonnegative')
    objective, risk, host = conditional_hostfloor_loss(components, is_target, host_floor, dual)
    compute = components['compute_seconds'].mean() / compute_scale
    pore = components['pore_seconds'][~is_target.bool()].mean() / pore_scale
    # Preserve the original floating-point graph exactly for arm A.
    if alpha != 0:
        objective = objective + alpha * compute
    if beta != 0:
        objective = objective + beta * pore
    return objective, risk, host, compute, pore


def resource_key(metrics, host_floor, target_limit, compute_scale, pore_scale):
    """Feasibility first, then the prespecified resource score and tie breakers."""
    _positive_scales(compute_scale, pore_scale)
    if not math.isfinite(float(target_limit)) or not 0 <= target_limit <= 1:
        raise ValueError('target_limit must lie in [0,1]')
    host, risk, compute, pore = (float(metrics[k]) for k in (
        'host_effective_reject_rate', 'target_command_frr', 'mean_compute_seconds',
        'host_mean_pore_seconds'))
    if not all(math.isfinite(v) for v in (host, risk, compute, pore)):
        raise ValueError('selection metrics must be finite')
    score = compute / compute_scale + pore / pore_scale
    tail = (score, risk, pore, compute)
    if host >= host_floor and risk <= target_limit:
        return (0, *tail)
    return (1, max(0., host_floor-host), max(0., risk-target_limit), *tail)


def _tensor_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def train_resource(model, train_data, dev_data, config, output, device='cpu'):
    """Mirror hostfloor optimization; separate early stopping from hard gated selection."""
    cfg = copy.deepcopy(config)
    cfg.update(variant='resource', objective='hostfloor_plus_resource',
               target_frr_limit=None, deployment_mode='argmax', auxiliary_weight=.1)
    resource = cfg['resource_ablation']
    alpha, beta = float(resource['alpha']), float(resource['beta'])
    compute_scale, pore_scale = float(resource['compute_scale']), float(resource['pore_scale'])
    dev_compute_scale, dev_pore_scale = float(resource['dev_compute_scale']), float(resource['dev_pore_scale'])
    _positive_scales(compute_scale, pore_scale, dev_compute_scale, dev_pore_scale)
    if any(not math.isfinite(v) or v < 0 for v in (alpha, beta)):
        raise ValueError('resource weights must be finite and nonnegative')
    target_limit = float(resource['target_limit'])
    if not math.isfinite(target_limit) or not 0 <= target_limit <= 1:
        raise ValueError('target_limit must lie in [0,1]')
    fixed = bool(resource.get('frozen_representation', False))
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
    if cp.get('variant', cp.get('config', {}).get('variant')) != ('hostfloor' if fixed else 'softstop'):
        raise ValueError('resource requires hostfloor for frozen representation or softstop initialization')
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
    if fixed:
        for param in model.parameters():
            param.requires_grad_(False)
    frozen = {name: param.detach().clone() for name, param in model.state_dict().items()
              if fixed or name.startswith(('encoder.', 'projection.'))}
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
    feasible_key = None
    diagnostic_key = None
    stale = 0
    started = time.time()
    (output/'config.json').write_text(json.dumps(cfg, indent=2))

    cached_train = cached_dev = cached_probabilities = None
    if fixed:
        cached_train, _ = prefix_scores(model, train_data, device)
        cached_dev, cached_probabilities = prefix_scores(model, dev_data, device)
        # Inference tensors cannot be saved for policy-head backward.
        cached_train = cached_train.clone(); cached_dev = cached_dev.clone()
        cached_probabilities = cached_probabilities.clone()
    source_files = [Path(__file__), Path(__file__).with_name('hostfloor_training.py'),
                    Path(__file__).with_name('risk_training.py'),
                    Path(__file__).with_name('_policy_training.py'),
                    Path(__file__).with_name('control_evaluation.py')]
    source_files.extend(Path(__file__).with_name(name + '.py') for name in
                        ('model', 'controllers', 'replay', 'costs', 'contracts', 'hostfloor'))
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    static_hashes = dict(config=_json_hash(cfg), timing=_json_hash(cfg.get('timing', {})),
                         sources=source_hashes, initialization=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest())

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

    def validation(epoch):
        model.eval(); controller.eval()
        states, probabilities = (prefix_scores(model, dev_data, device) if not fixed
                                 else (cached_dev, cached_probabilities))
        metrics = evaluate_policy(model, controller, dev_data, cfg, device=device,
            mode='argmax', states=states, probabilities=probabilities,
            output=output/'dev_replay'/f'epoch_{epoch:03d}')
        risk, host, error = cohort_expectation(dev_data, dev_available, dev_branches, dev_ends, states)
        metrics.update(soft_target_command_frr=risk, soft_host_effective_reject_rate=host,
            soft_probability_mass_max_error=error, host_floor=dev_floor,
            host_constraint_met=metrics['host_effective_reject_rate'] >= dev_floor,
            target_constraint_met=metrics['target_command_frr'] <= target_limit,
            target_command_limit=target_limit)
        return metrics

    def persist(epoch, metrics, row):
        nonlocal best_key, stale, feasible_key, diagnostic_key
        for name, original in frozen.items():
            if not torch.equal(model.state_dict()[name], original):
                raise RuntimeError('frozen representation tensors changed')
        stop_key = hostfloor_key(metrics, dev_floor)
        key = resource_key(metrics, dev_floor, target_limit, dev_compute_scale, dev_pore_scale)
        improved = best_key is None or stop_key < best_key
        if improved:
            best_key = stop_key; stale = 0
        else:
            stale += 1
        history.append(row)
        state = dict(model_state=model.state_dict(), controller_state=controller.state_dict(),
            config=cfg, variant='resource', epoch=epoch, dev_metrics=metrics, dual=dual,
            checkpoint_role='epoch',
            host_constraint_met=metrics['host_constraint_met'],
            target_constraint_met=metrics['target_constraint_met'],
            selection_status='feasible' if key[0] == 0 else 'diagnostic_infeasible',
            selection_key=key, stop_key=stop_key, best_stop_key=best_key,
            best_key=key if key[0] == 0 and (feasible_key is None or key < feasible_key) else feasible_key,
            stale=stale, hashes=dict(static_hashes, model=_tensor_hash(model.state_dict()),
                                   controller=_tensor_hash(controller.state_dict())),
            optimizer_state=optimizer.state_dict(), history=history,
            numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state())
        if str(device).startswith('cuda'):
            state['cuda_rng'] = torch.cuda.get_rng_state(device)
        torch.save(state, output/'checkpoints'/f'epoch_{epoch:03d}.pt')
        torch.save(state, output/'last.pt')
        if diagnostic_key is None or key < diagnostic_key:
            diagnostic_key = key
            torch.save(dict(state, checkpoint_role='diagnostic', selection_status='diagnostic_only',
                            candidate_feasible=key[0] == 0), output/'diagnostic.pt')
        if key[0] == 0 and (feasible_key is None or key < feasible_key):
            feasible_key = key
            torch.save(dict(state, checkpoint_role='selected'), output/'selected.pt')
        (output/'history.json').write_text(json.dumps(history, indent=2))

    metrics = validation(0)
    persist(0, metrics, dict(epoch=0, stage='initialization', dual=dual, dev=metrics))
    for epoch in range(1, setting['epochs']+1):
        head_only = epoch <= setting['head_only_epochs']
        for param in recurrent:
            param.requires_grad_(not fixed and not head_only)
        model.train(not fixed and not head_only); controller.train()
        sums = dict(loss=0., target_command_risk=0., host_effective_rate=0., prefix_ce=0.,
                    compute_normalized=0., host_pore_normalized=0.)
        count = 0
        used_dual = dual
        for ids, _ in sample_balanced_ids(train_data.is_target, eligible, setting['batch_size'], rng):
            e = torch.as_tensor(np.asarray(train_data.features[ids]).copy(), dtype=torch.float32, device=device)
            valid = torch.as_tensor(train_data.valid[ids], device=device)
            target = torch.as_tensor(train_data.is_target[ids], device=device)
            states = (cached_train[ids].to(device) if fixed else model.forward_states(e, valid))
            comp = components(states, ids, train_available, train_branches, train_ends)
            objective, risk, host, compute, pore = resource_loss(comp, target, train_floor, used_dual,
                alpha, beta, compute_scale, pore_scale)
            ce = prefix_loss(model.head(states), target.long(), valid) if valid.any() else states.sum()*0
            loss = objective + .1*ce
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.); optimizer.step()
            for key, value in zip(sums, (loss, risk, host, ce, compute, pore)):
                sums[key] += float(value.detach())*len(ids)
            count += len(ids)
        risk, host, error = cohort_expectation(train_data, train_available, train_branches, train_ends, cached_train)
        dual = max(0., used_dual + dual_lr*(train_floor-host))
        metrics = validation(epoch)
        row = dict(epoch=epoch, train={key: value/count for key, value in sums.items()},
            train_target_command_frr=risk, train_host_rate=host,
            train_probability_mass_max_error=error, dual_used=used_dual, dual=dual,
            head_only=head_only, dev=metrics, elapsed_seconds=time.time()-started)
        persist(epoch, metrics, row)
        print(row, flush=True)
        if stale >= setting['patience']:
            break
    return output/'selected.pt' if feasible_key is not None else None
