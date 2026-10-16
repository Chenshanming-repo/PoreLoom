import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from poreloom.baselines import ganbase
from poreloom.contracts import Action


def test_classifier_filters_and_returns_human_probability(tmp_path):
    (tmp_path / 'notebook').mkdir()
    (tmp_path / 'model').mkdir()
    source = '''raise RuntimeError("notebook side effects")
class PositionalEncoding(nn.Module):
    pass
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([0., 2.]))
    def forward(self, x):
        assert not self.training
        assert torch.is_inference_mode_enabled()
        assert x.dtype == torch.long and x.shape[1] == 200
        assert (x[0] == torch.arange(200, device=x.device) % 4).all()
        return self.logits.log_softmax(0).expand(len(x), -1)
'''
    (tmp_path / 'notebook/human_prediction.ipynb').write_text(json.dumps({'cells':[{'cell_type':'code','source':source}]}))
    torch.save({'logits':torch.tensor([0.,2.])}, tmp_path / 'model/Human_model.pt')
    classifier = ganbase.GanbaseClassifier(tmp_path, 'cpu')
    scores = classifier.score(['A'*249, 'A'*50+'ACGT'*50, 'A'*249+'N', 'N'*50+'ACGT'*50, 'a'*250])
    assert np.isnan(scores[[0,2,4]]).all()
    np.testing.assert_allclose(scores[[1,3]], torch.tensor(2.).sigmoid().item())
    torch.save({}, tmp_path / 'model/Human_model.pt')
    with pytest.raises(RuntimeError):
        ganbase.GanbaseClassifier(tmp_path, 'cpu')


@pytest.mark.parametrize('length,p,base,cls,command,effective,compute', [
    (4500,.99,.1,.2,False,False,0.),
    (8000,.99,.1,.2,True,True,.3),
    (8000,.9,.1,.2,True,False,.3),
    (5000,.99,.125,0.,False,False,.125),
    (5300,.99,.1,0.,True,False,.1),
    (8000,np.nan,.1,100.,True,False,.1),
    (4900,np.nan,.1,100.,False,False,.1),
])
def test_fixed_replay_boundaries(length,p,base,cls,command,effective,compute):
    out = ganbase.fixed_outcome('r', False, length,4500,p,base,cls)
    assert out.read_id == 'r' and out.command_issued == command
    assert out.effective_reject == effective
    assert out.compute_seconds == pytest.approx(compute)
    assert out.pore_seconds == pytest.approx(1.125+base+cls+.1 if effective else length/4000)
    if command:
        assert out.action == (Action.REJECT if p > .9 else Action.ACCEPT)
    if np.isnan(p) and command:
        assert out.fallback


@pytest.mark.parametrize('kwargs',[{'basecall_seconds':-1},{'classify_seconds':float('inf')},{'probability':1.1},{'probability':float('-inf')},{'threshold':float('nan')}])
def test_invalid_replay_parameters(kwargs):
    args=dict(read_id='r',is_target=True,raw_length=8000,end=4500,probability=.5,basecall_seconds=0.,classify_seconds=0.)
    args.update(kwargs)
    with pytest.raises(ValueError): ganbase.fixed_outcome(**args)


def test_prefix_normalization_and_order(monkeypatch):
    seen=[]
    config={'sample_rate':4000,'normalisation':{'q':.2},'basecaller':{'chunksize':100,'overlap':10}}
    model=SimpleNamespace(config=config)
    def normalization(raw, **kwargs):
        assert kwargs['norm_params']==config['normalisation']
        return raw.mean(), 2.
    def basecall(model, reads, **kwargs):
        reads=list(reads)
        seen.extend(r.signal.copy() for r in reads)
        for read in reversed(reads): yield read, {'sequence':str(read.read_id)}
    monkeypatch.setitem(sys.modules,'bonito.util',SimpleNamespace(load_model=lambda *a,**kw:model))
    monkeypatch.setitem(sys.modules,'bonito.reader',SimpleNamespace(normalisation=normalization))
    monkeypatch.setitem(sys.modules,'bonito.crf.basecall',SimpleNamespace(basecall=basecall))
    caller=ganbase.BonitoCaller('/unused','cpu')
    full=np.array([1.,3.,5.,99999.])
    assert caller.basecall([full[:3],np.array([4.,8.])])==['0','1']
    np.testing.assert_array_equal(seen[0],[-1.,0.,1.])
    assert seen[0].dtype==np.float32
    config['sample_rate']=5000
    with pytest.raises(ValueError): ganbase.BonitoCaller('/unused','cpu')


def test_official_model_parity():
    upstream=Path('/homeb/user/github_repos/GANBase')
    if not (upstream/'model/Human_model.pt').exists(): pytest.skip('official artifacts unavailable')
    classifier=ganbase.GanbaseClassifier(upstream,'cpu')
    namespace=dict(torch=torch, nn=torch.nn, math=__import__('math'), Variable=torch.autograd.Variable)
    # Independent execution of the authoritative two class source cells.
    notebook=json.loads((upstream/'notebook/human_prediction.ipynb').read_text())
    for cell in notebook['cells']:
        source=''.join(cell.get('source',[]))
        if cell['cell_type']=='code' and ('class PositionalEncoding' in source or 'class Discriminator' in source):
            exec(source, namespace)
    reference=namespace['Discriminator']().eval()
    reference.load_state_dict(torch.load(upstream/'model/Human_model.pt',map_location='cpu',weights_only=True),strict=True)
    with torch.inference_mode(): expected=reference(torch.tensor([[0,1,2,3]*50])).exp()[:,1].numpy()
    np.testing.assert_allclose(classifier.score(['A'*50+'ACGT'*50]),expected,rtol=1e-6,atol=0.)
    assert len(classifier.model.state_dict())==100


def test_missing_sampling_metadata_rejected(monkeypatch):
    monkeypatch.setitem(sys.modules,'bonito.util',SimpleNamespace(load_model=lambda *a,**kw:SimpleNamespace(config={})))
    monkeypatch.setitem(sys.modules,'bonito.reader',SimpleNamespace(normalisation=lambda *a,**kw:(0.,1.)))
    monkeypatch.setitem(sys.modules,'bonito.crf.basecall',SimpleNamespace(basecall=lambda *a,**kw:[]))
    with pytest.raises(ValueError, match='sampling metadata'):
        ganbase.BonitoCaller('/unused','cpu')


def test_basecaller_missing_read_is_not_silently_filtered(monkeypatch):
    monkeypatch.setitem(sys.modules,'bonito.util',SimpleNamespace(load_model=lambda *a,**kw:SimpleNamespace(config={'run_info':{'sample_rate':4000}})))
    monkeypatch.setitem(sys.modules,'bonito.reader',SimpleNamespace(normalisation=lambda *a,**kw:(0.,1.)))
    monkeypatch.setitem(sys.modules,'bonito.crf.basecall',SimpleNamespace(basecall=lambda *a,**kw:[]))
    caller=ganbase.BonitoCaller('/unused','cpu')
    assert caller.basecall([])==[]
    with pytest.raises(ValueError, match='omitted'):
        caller.basecall([np.ones(10)])
    with pytest.raises(ValueError, match='finite'):
        caller.basecall([np.array([np.nan])])


def test_verified_sampling_sidecar(monkeypatch,tmp_path):
    import hashlib
    monkeypatch.setitem(sys.modules,'bonito.util',SimpleNamespace(load_model=lambda *a,**kw:SimpleNamespace(config={})))
    monkeypatch.setitem(sys.modules,'bonito.reader',SimpleNamespace(normalisation=lambda *a,**kw:(0.,1.)))
    monkeypatch.setitem(sys.modules,'bonito.crf.basecall',SimpleNamespace(basecall=lambda *a,**kw:[]))
    source=tmp_path/'official.toml'
    source.write_text('[run_info]\nsample_rate = 4000\n')
    evidence=dict(model_name='model',sample_rate_hz=4000,source_config_path=str(source),source_config_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (tmp_path/'sample_rate_evidence.json').write_text(json.dumps(evidence))
    ganbase.BonitoCaller(tmp_path/'model','cpu')
    evidence['model_name']='other'
    (tmp_path/'sample_rate_evidence.json').write_text(json.dumps(evidence))
    with pytest.raises(ValueError): ganbase.BonitoCaller(tmp_path/'model','cpu')


def test_official_registry_sampling_evidence(monkeypatch,tmp_path):
    import hashlib
    monkeypatch.setitem(sys.modules,'bonito.util',SimpleNamespace(load_model=lambda *a,**kw:SimpleNamespace(config={})))
    monkeypatch.setitem(sys.modules,'bonito.reader',SimpleNamespace(normalisation=lambda *a,**kw:(0.,1.)))
    monkeypatch.setitem(sys.modules,'bonito.crf.basecall',SimpleNamespace(basecall=lambda *a,**kw:[]))
    root=Path('/homeb/user/signal_explore/results/ganbase_baseline_20261011/resources')
    cpp=root/'dorado_v0.3.4_models.cpp'; header=root/'dorado_v0.3.4_models.h'
    if not cpp.exists(): pytest.skip('registry artifacts unavailable')
    name='dna_r10.4.1_e8.2_400bps_hac@v4.1.0'
    evidence=dict(evidence_kind='official_registry',model_name=name,sample_rate_hz=4000,source_config_path=str(cpp),source_config_sha256=hashlib.sha256(cpp.read_bytes()).hexdigest(),registry_models_path=str(header),registry_models_sha256=hashlib.sha256(header.read_bytes()).hexdigest())
    path=tmp_path/'evidence.json';path.write_text(json.dumps(evidence))
    ganbase.BonitoCaller(tmp_path/name,'cpu',sampling_rate_evidence=path)
    evidence['model_name']='invented'
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError): ganbase.BonitoCaller(tmp_path/'invented','cpu',sampling_rate_evidence=path)
    evidence['model_name']=name; evidence['sample_rate_hz']=5000
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError): ganbase.BonitoCaller(tmp_path/name,'cpu',sampling_rate_evidence=path)


def test_non_koi_model_scores_are_batch_first_for_decoder():
    class TimeFirstModel(torch.nn.Module):
        stride=5
        seqdist=SimpleNamespace(n_base=4, state_len=1)
        def forward(self,x):
            scores=torch.arange(120.).reshape(3,2,20)
            scores[:,:,::5]=-999.  # Each state's fixed blank is not a move.
            return scores
    wrapped=ganbase._BatchFirstScores(TimeFirstModel())
    scores=wrapped(torch.zeros(2,1,15))
    assert scores.shape==(2,3,16)
    assert scores.is_contiguous()
    expected=torch.tensor([21.,22.,23.,24.,26.,27.,28.,29.,31.,32.,33.,34.,36.,37.,38.,39.])
    torch.testing.assert_close(scores[1,0],expected)
    torch.testing.assert_close(scores[1,2],expected+80)
    assert not (scores==-999).any()
    assert wrapped.stride==5


def test_decoder_bridge_rejects_unrecognized_channels():
    class InvalidModel(torch.nn.Module):
        stride=5
        seqdist=SimpleNamespace(n_base=4, state_len=1)
        def forward(self,x):
            return torch.zeros(3,2,19)
    with pytest.raises(ValueError, match='channel'):
        ganbase._BatchFirstScores(InvalidModel())(torch.zeros(2,1,15))
