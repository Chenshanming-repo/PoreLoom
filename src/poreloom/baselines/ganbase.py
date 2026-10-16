"""Frozen official GANBase classifier and causal, supplied-prefix Bonito adapter."""
import ast
import json
import hashlib
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from ..contracts import Action, Outcome


class GanbaseClassifier:
    """Use checkpoint-compatible notebook classes without executing notebook jobs."""
    def __init__(self, upstream, device):
        upstream = Path(upstream)
        notebook = json.loads((upstream / 'notebook/human_prediction.ipynb').read_text())
        classes = {}
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            for node in ast.parse(''.join(cell['source'])).body:
                if isinstance(node, ast.ClassDef) and node.name in ('PositionalEncoding', 'Discriminator'):
                    if node.name in classes:
                        raise ValueError(f'duplicate notebook class: {node.name}')
                    classes[node.name] = node
        if set(classes) != {'PositionalEncoding', 'Discriminator'}:
            raise ValueError('official notebook classes missing')
        namespace = dict(torch=torch, nn=nn, math=math, Variable=Variable)
        module = ast.Module(body=[classes['PositionalEncoding'], classes['Discriminator']], type_ignores=[])
        exec(compile(module, str(upstream / 'notebook/human_prediction.ipynb'), 'exec'), namespace)
        self.device = torch.device(device)
        self.model = namespace['Discriminator']()
        state = torch.load(upstream / 'model/Human_model.pt', map_location='cpu', weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    def score(self, sequences):
        sequences = list(sequences)
        result = np.full(len(sequences), np.nan, dtype=np.float64)
        indices = [i for i, seq in enumerate(sequences)
                   if len(seq) >= 250 and set(seq[-200:]) <= set('ACGT')]
        if indices:
            encoding = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
            inputs = torch.tensor([[encoding[b] for b in sequences[i][-200:]] for i in indices],
                                  dtype=torch.long, device=self.device)
            with torch.inference_mode():
                result[indices] = self.model(inputs).exp()[:, 1].cpu().numpy()
        return result


class _BatchFirstScores(nn.Module):
    """Bridge expanded non-Koi TNC scores to move-only Koi NTC input.

    Each ordinary CRF state has a blank followed by four base moves. Koi
    receives its fixed blank score separately and requires only the moves.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.stride = model.stride
        self.seqdist = model.seqdist

    def forward(self, batch):
        scores = self.model(batch)
        if scores.ndim != 3 or scores.shape[1] != batch.shape[0]:
            raise ValueError('unexpected non-Koi model score layout; expected TNC')
        n_base = self.seqdist.n_base
        n_states = n_base ** self.seqdist.state_len
        if scores.shape[2] != n_states * (n_base + 1):
            raise ValueError('unexpected expanded non-Koi CRF channel count')
        scores = scores.reshape(*scores.shape[:2], n_states, n_base + 1)[..., 1:].flatten(2)
        return scores.transpose(0, 1).contiguous()


class BonitoCaller:
    """Normalize each supplied pA prefix independently and preserve input order."""
    def __init__(self, model_dir, device, batch_size=64, sampling_rate_evidence=None):
        from bonito.util import load_model
        from bonito.reader import normalisation
        from bonito.crf.basecall import basecall
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size must be positive integer')
        self.model = load_model(str(model_dir), device=device, half=True, use_koi=False)
        self.config = self.model.config
        rates = []
        def collect(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in ('sample_rate', 'sampling_rate', 'sample_rate_hz', 'sampling_rate_hz'):
                        rates.append(item)
                    elif isinstance(item, dict):
                        collect(item)
        collect(self.config)
        if not rates:
            evidence_path = Path(sampling_rate_evidence) if sampling_rate_evidence is not None else Path(model_dir).parent / 'sample_rate_evidence.json'
            if evidence_path.exists():
                import toml
                evidence = json.loads(evidence_path.read_text())
                if evidence.get('model_name') != Path(model_dir).name or evidence.get('sample_rate_hz') != 4000:
                    raise ValueError('sampling evidence does not match model name and 4000 Hz')
                source = Path(evidence['source_config_path'])
                if hashlib.sha256(source.read_bytes()).hexdigest() != evidence['source_config_sha256']:
                    raise ValueError('sampling evidence source hash mismatch')
                if evidence.get('evidence_kind') == 'official_registry':
                    header = Path(evidence['registry_models_path'])
                    if hashlib.sha256(header.read_bytes()).hexdigest() != evidence['registry_models_sha256']:
                        raise ValueError('registry header hash mismatch')
                    header_text = header.read_text()
                    supported = re.search(r'namespace simplex\s*\{.*?models\s*=\s*\{(.*?)\};', header_text, re.S)
                    if supported is None or evidence['model_name'] not in re.findall(r'"([^"\n]+)"', supported.group(1)):
                        raise ValueError('model absent from official simplex registry')
                    mapping = re.search(r'sample_rate_by_model\s*=\s*\{(.*?)\};', header_text, re.S)
                    function = re.search(r'uint16_t get_sample_rate_by_model_name\(const std::string& model_name\)\s*\{(.*?)\n\}', source.read_text(), re.S)
                    if mapping is None or function is None:
                        raise ValueError('official sample-rate mapping missing')
                    body = re.sub(r'//[^\n]*', '', function.group(1))
                    expected = 'auto iter = sample_rate_by_model.find(model_name); if (iter != sample_rate_by_model.end()) { return iter->second; } else { return 4000; }'
                    if re.sub(r'\s+', '', body) != re.sub(r'\s+', '', expected):
                        raise ValueError('unrecognized official sample-rate function')
                    mapped = dict((name, int(rate)) for name, rate in re.findall(r'\{\s*"([^"\n]+)"\s*,\s*(\d+)\s*\}', mapping.group(1)))
                    rates.append(mapped.get(evidence['model_name'], 4000))
                else:
                    collect(toml.loads(source.read_text()))
        if not rates or any(float(rate) != 4000 for rate in rates):
            raise ValueError('Bonito requires explicit, consistent 4000 Hz sampling metadata')
        self.batch_size = batch_size
        self._normalisation = normalisation
        self._basecall = basecall

    def basecall(self, prefixes_pa):
        reads = []
        for i, prefix in enumerate(prefixes_pa):
            raw = np.asarray(prefix, dtype=np.float32)
            if raw.ndim != 1 or not len(raw) or not np.isfinite(raw).all():
                raise ValueError('expected nonempty finite one-dimensional pA prefix')
            shift, scale = self._normalisation(raw, scaling_strategy=self.config.get('scaling'),
                                               norm_params=self.config.get('normalisation'))
            if not np.isfinite(shift) or not np.isfinite(scale) or scale <= 0:
                raise ValueError('invalid prefix normalization')
            reads.append(SimpleNamespace(read_id=str(i), signal=((raw - shift) / scale).astype(np.float32)))
        result = [None] * len(reads)
        if not reads:
            return []
        settings = self.config.get('basecaller', {})
        decoder_model = _BatchFirstScores(self.model) if isinstance(self.model, nn.Module) else self.model
        for read, attrs in self._basecall(decoder_model, reads,
                                         chunksize=settings.get('chunksize', 10000),
                                         overlap=settings.get('overlap', 500), batchsize=self.batch_size):
            index = int(read.read_id)
            if not 0 <= index < len(result) or result[index] is not None:
                raise ValueError('basecaller returned duplicate or unknown read')
            result[index] = attrs['sequence']
        if any(sequence is None for sequence in result):
            raise ValueError('basecaller omitted supplied prefix')
        return result


def fixed_outcome(read_id, is_target, raw_length, end, probability, basecall_seconds,
                  classify_seconds, threshold=0.9):
    """Replay one fixed observation with strict natural-end and ejection boundaries.

    total_cost is explicitly unassessed (zero); compute records incurred service.
    """
    for value in (basecall_seconds, classify_seconds):
        if not math.isfinite(value) or value < 0:
            raise ValueError('service seconds must be finite and nonnegative')
    for value in (raw_length, end):
        if not math.isfinite(value) or value < 0 or int(value) != value:
            raise ValueError('sample counts must be nonnegative integers')
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold must be finite in [0,1]')
    if not math.isnan(probability) and (not math.isfinite(probability) or not 0 <= probability <= 1):
        raise ValueError('probability must be NaN or finite in [0,1]')
    natural = raw_length / 4000.
    if raw_length <= end:
        return Outcome(read_id, bool(is_target), 'natural_end', False, False, natural, 0., 0.)
    valid = math.isfinite(probability)
    compute = basecall_seconds + (classify_seconds if valid else 0.)
    finish = end / 4000. + compute
    if finish >= natural:
        return Outcome(read_id, bool(is_target), 'late_decision', False, False, natural, compute, 0.)
    action = Action.REJECT if valid and probability > threshold else Action.ACCEPT
    effective = action == Action.REJECT and finish + .1 < natural
    pore = finish + .1 if effective else natural
    return Outcome(read_id, bool(is_target), 'classified' if valid else 'input_fallback', True,
                   effective, pore, compute, 0., action=action, decision_seconds=finish,
                   effective_seconds=pore if effective else None, fallback=not valid)
