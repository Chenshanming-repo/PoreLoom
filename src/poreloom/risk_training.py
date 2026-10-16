"""Class conditional risk, causal imitation, and feasibility-first selection."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.nn import functional as F
from .controllers import PolicyController, masked_softmax
from .contracts import Action
from .costs import TimingConfig, CostConfig
from .replay import ReplayEnv


def exact_expected_components(logits, branch_details, available, end_details):
    n, K, actions = logits.shape
    if actions != 3:
        raise ValueError('exact stopping requires three actions')
    survival = logits.new_ones(n)
    result = {key: logits.new_zeros(n) for key in branch_details}
    mass = logits.new_zeros(n)
    for k in range(K):
        alive = available[:, k].to(logits.dtype)
        ended = survival * (1 - alive)
        for key in result:
            result[key] = result[key] + ended * end_details[key]
        mass = mass + ended
        survival = survival * alive
        p = masked_softmax(logits[:, k], k=k + 1, max_windows=K)
        for key in result:
            result[key] = result[key] + survival * (p[:, :2] * branch_details[key][:, k]).sum(-1)
        mass = mass + survival * p[:, :2].sum(-1)
        survival = survival * p[:, 2]
    for key in result:
        result[key] = result[key] + survival * end_details[key]
    result['probability_mass'] = mass + survival
    return result


def resource_parameters(config):
    resource = config.get('resource', {})
    time_ref = resource.get('host_time_ref', 1.)
    if isinstance(time_ref, str):
        time_ref = resource.get('train_host_mean_natural_seconds', config.get('train_host_mean_natural_seconds'))
    if time_ref is None or float(time_ref) <= 0:
        raise ValueError('host_time_ref must be resolved from TRAIN and positive')
    compute_ref = float(resource.get('compute_time_ref', .03))
    if compute_ref <= 0:
        raise ValueError('compute_time_ref must be positive')
    return float(time_ref), compute_ref, float(resource.get('compute_weight', .01))


def risk_event(config):
    event = config.get('risk', {}).get('event', 'command_reject')
    if event not in ('command_reject', 'effective_reject'):
        raise ValueError('unknown risk event: ' + event)
    return event


def conditional_resource_risk(components, is_target, config):
    target = is_target.bool()
    if not target.any() or target.all():
        raise ValueError('class conditional loss requires both classes')
    T, C, mu = resource_parameters(config)
    rho = config.get('rho_target', .01)
    compute = (1-rho)*components['compute_seconds'][~target].mean() + rho*components['compute_seconds'][target].mean()
    resource = components['pore_seconds'][~target].mean()/T + mu*compute/C
    return resource, components[risk_event(config)][target].mean()


def annotate_metrics(metrics, config):
    T, C, mu = resource_parameters(config)
    rho = config.get('rho_target', .01)
    hc = metrics.get('host_mean_compute_seconds')
    tc = metrics.get('target_mean_compute_seconds')
    if hc is None or tc is None:
        raise ValueError('resource score requires per-class compute metrics')
    metrics['resource_score'] = metrics['host_mean_pore_seconds']/T + mu*((1-rho)*hc+rho*tc)/C
    limit = config.get('target_frr_limit')
    risk = metrics.get('target_command_frr' if risk_event(config)=='command_reject' else 'target_effective_frr')
    metrics['constraint_enabled'] = limit is not None
    metrics['constraint_met'] = None if limit is None else bool(risk is not None and risk <= limit)
    metrics['risk_event'] = risk_event(config)
    return metrics


def selection_key(metrics, config):
    metrics = annotate_metrics(metrics, config)
    risk = metrics['target_command_frr' if risk_event(config)=='command_reject' else 'target_effective_frr']
    score = metrics['resource_score']
    return (0, score) if not metrics['constraint_enabled'] or metrics['constraint_met'] else (1, float('inf') if risk is None else risk, score)


def update_beta(beta, train_risk, config):
    risk = config.get('risk', {})
    return max(0., min(risk.get('dual_max', 100.), float(beta)+risk.get('dual_lr', 10.)*(float(train_risk)-risk.get('epsilon_train', .0025))))


class RiskCheckpointManager:
    def __init__(self, output, config):
        self.output = Path(output); self.output.mkdir(parents=True, exist_ok=True)
        self.config = config; self.best_key = None; self.feasible_key = None; self.diagnostic_key = None

    @property
    def best_path(self):
        return self.output / ('best_feasible.pt' if self.feasible_key is not None else 'best_diagnostic.pt')

    def consider(self, state, metrics, beta):
        key = selection_key(metrics, self.config)
        payload = dict(state, dev_metrics=dict(metrics), beta=float(beta))
        diagnostic = (key if key[0] else (0, metrics['resource_score']))
        if self.diagnostic_key is None or diagnostic < self.diagnostic_key:
            torch.save(payload, self.output/'best_diagnostic.pt'); self.diagnostic_key = diagnostic
        if metrics['constraint_met'] is True and (self.feasible_key is None or key < self.feasible_key):
            torch.save(payload, self.output/'best_feasible.pt'); self.feasible_key = key
        improved = self.best_key is None or key < self.best_key
        if improved: self.best_key = key
        return improved


def generate_teacher_labels(data, probabilities, config):
    """Probabilities are consumed only at the current live observation."""
    K = config['max_windows']; actions = np.full((len(data.valid), K), -1, dtype=np.int64)
    visits = np.zeros_like(actions, dtype=bool)
    timing = TimingConfig(**config.get('timing', {})); cost = CostConfig(**config.get('cost', {}))
    threshold = config.get('teacher', {}).get('target_threshold', .5)
    for i, record in enumerate(data.records()):
        env = ReplayEnv(record, timing, cost, K); obs = env.reset()
        while obs is not None:
            k = obs.k-1
            action = Action.ACCEPT if probabilities[i, k] >= threshold else (Action.REJECT if obs.k == K else Action.DEFER)
            actions[i, k] = int(action); visits[i, k] = True
            obs, _ = env.step(action)
    return dict(actions=actions, visits=visits)


def teacher_loss(logits, actions, visits):
    K = logits.shape[1]
    legal = torch.ones_like(logits, dtype=torch.bool); legal[:, -1, 2:] = False
    losses = F.cross_entropy(logits.masked_fill(~legal, -float('inf')).reshape(-1, logits.shape[-1]), actions.clamp_min(0).reshape(-1), reduction='none').reshape(actions.shape)
    counts = visits.sum(1); present = counts > 0
    per_read = (losses*visits).sum(1)/counts.clamp_min(1)
    return per_read[present].mean() if present.any() else logits.sum()*0


def train_teacher(model, train_data, dev_data, independent_head, config, output, device='cpu'):
    from .control_evaluation import prefix_scores, evaluate_policy
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    model = model.to(device).eval(); independent_head = independent_head.to(device).eval()
    with torch.no_grad():
        chunks = []
        for start in range(0, len(train_data.valid), 256):
            e = torch.as_tensor(np.asarray(train_data.features[start:start+256, :config['max_windows']]).copy(), dtype=torch.float32, device=device)
            chunks.append(independent_head(e).softmax(-1)[..., 1].cpu().numpy())
    labels = generate_teacher_labels(train_data, np.concatenate(chunks), config)
    states, _ = prefix_scores(model, train_data, device)
    controller = PolicyController(max_windows=config['max_windows']).to(device)
    setting = config.get('teacher', {})
    optimizer = torch.optim.AdamW(controller.parameters(), lr=setting.get('lr', 3e-4))
    history = []
    for epoch in range(setting.get('epochs', 3)):
        total = 0.; count = 0
        controller.train()
        for ids in torch.randperm(len(states)).split(setting.get('batch_size', 128)):
            logits = controller(states[ids].to(device))
            actions = torch.as_tensor(labels['actions'][ids.numpy()], device=device)
            visits = torch.as_tensor(labels['visits'][ids.numpy()], device=device)
            loss = teacher_loss(logits, actions, visits)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += float(loss.detach())*len(ids); count += len(ids)
        row = dict(epoch=epoch+1, teacher_loss=total/count)
        history.append(row); print('teacher', row, flush=True)
    (output/'teacher_history.json').write_text(json.dumps(history, indent=2))
    metrics = evaluate_policy(model, controller, dev_data, config, device=device)
    beta = config.get('risk', {}).get('dual_initial', 5.)
    state = dict(model_state=model.state_dict(), controller_state=controller.state_dict(), config=config, variant='softstop', epoch=0, dev_metrics=metrics, beta=beta)
    torch.save(state, output/'teacher.pt')
    manager = RiskCheckpointManager(output, config); manager.consider(state, metrics, beta)
    print('teacher DEV', metrics, flush=True)
    return output/'teacher.pt'
