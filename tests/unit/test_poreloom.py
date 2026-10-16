import numpy as np
import pytest
import torch
from torch import nn
from poreloom.windowing import normalize_windows, split_windows, SignalBuffer
from poreloom.model import PoreLoom, prefix_loss
from poreloom.policy import decide
from poreloom.simulator_adapter import StreamingAdapter
from poreloom.evaluation import terminal_decisions, correction_metrics


def test_window_bounds_and_short_reads():
    x,v=split_windows(np.arange(4499,dtype=np.int16))
    assert x.shape==(8,1500) and v.tolist()==[True,False,False,False,False,False,False,False]
    np.testing.assert_array_equal(x[0],np.arange(1500,3000))


def test_normalization_is_current_window_only():
    x=np.stack([np.arange(1500),np.ones(1500),np.arange(1500)]).astype(float)
    x[2,3]=np.nan
    z,v=normalize_windows(x)
    assert v.tolist()==[True,False,False]
    assert np.isfinite(z).all() and abs(np.median(z[0]))<1e-6
    with pytest.raises(ValueError):normalize_windows(np.zeros((1,1499)))


def test_normalization_matches_native_recipe():
    from poreloom.baselines.preprocessing import readcurrent_inputs as normalize_batch
    raw=np.random.default_rng(2).integers(-100,100,(3,3000),dtype=np.int16)
    # Repeat a short window to compare identical statistics and interior replacements.
    raw[:,:1500]=raw[:,1500:]
    actual,v=normalize_windows(raw[:,:1500]);expected,ev=normalize_batch(raw)
    np.testing.assert_allclose(actual,expected[:,:1500]);assert np.array_equal(v,ev)


@pytest.mark.parametrize('p,k,expected',[(.99,1,('reject','confident_host')),(.1,1,('accept','confident_target')),(.5,1,('defer','need_next_window')),(.989,8,('accept','budget_accept'))])
def test_policy_boundaries(p,k,expected):assert decide(p,1-p,k)==expected


def test_nonfinite_fallback():assert decide(np.nan,np.nan,1)==('accept','input_fallback')


def test_buffer_deduplicates_and_requires_contiguous_offsets():
    b=SignalBuffer();b.push(0,np.arange(2000));b.push(1500,np.arange(1500,3000))
    w=b.pop();np.testing.assert_array_equal(w,np.arange(1500,3000));assert b.pop() is None
    b.push(3000,np.arange(3000,4500));np.testing.assert_array_equal(b.pop(),np.arange(3000,4500))
    with pytest.raises(ValueError):b.push(5000,np.arange(10))


def tiny_model():return PoreLoom(nn.Sequential(nn.Conv1d(1,512,3,padding=1)))


def test_causal_prefix_and_padding_do_not_update_state():
    torch.manual_seed(2);m=tiny_model().eval();e=torch.randn(2,8,128);v=torch.ones(2,8,dtype=torch.bool);v[1,2:]=False
    z=m.forward_features(e,v)
    h=torch.zeros(2,128)
    for k in range(8):
        one,new=m.step_features(e[:,k],h,k+1)
        h=torch.where(v[:,k,None],new,h)
        torch.testing.assert_close(z[:,k][v[:,k]],one[v[:,k]])
    changed=e.clone();changed[:,3:]+=100
    torch.testing.assert_close(z[:,:3],m.forward_features(changed,v)[:,:3])
    torch.testing.assert_close(z[1,2:],torch.zeros_like(z[1,2:]))


def test_read_normalized_prefix_loss_and_invalid_reads():
    z=torch.tensor([[[4.,0.],[0.,4.]],[[0.,4.],[99.,-99.]]],requires_grad=True)
    v=torch.tensor([[True,True],[True,False]]);y=torch.tensor([0,1])
    ce=nn.functional.cross_entropy(z.reshape(-1,2),y[:,None].expand(-1,2).reshape(-1),reduction='none').reshape(2,2)
    torch.testing.assert_close(prefix_loss(z,y,v),((ce[0].mean())+ce[1,0])/2)
    prefix_loss(z,y,v).backward();assert torch.equal(z.grad[1,1],torch.zeros(2))
    with pytest.raises(ValueError):prefix_loss(z,y,torch.zeros_like(v))


def test_frozen_encoder_remains_eval():
    m=PoreLoom(nn.Sequential(nn.Conv1d(1,512,1),nn.BatchNorm1d(512)))
    m.freeze_encoder();m.train()
    assert not m.encoder.training and not m.projection.training and m.gru.training
    assert not any(p.requires_grad for p in m.encoder.parameters())


def test_terminal_never_uses_later_evidence_and_missing_breaks_prefix():
    p=np.array([[.995,.01,.01],[.5,.01,.01],[.5,.5,.5]])
    v=np.array([[1,1,1],[1,0,1],[1,1,1]],bool)
    d=terminal_decisions(p,v,np.array([4500,7500,10500]),np.array([14000]*3),'B2')
    assert d['action'].tolist()==['reject','accept','accept']
    assert d['window_index'].tolist()==[1,2,3]
    assert d['reason'].tolist()==['confident_host','input_fallback','budget_accept']


def test_conservative_accept_does_not_count_as_correction():
    labels=np.array([0,0,1]);wrong=np.array([1,1,0],bool)
    d={'action':np.array(['accept','accept','reject']), 'reason':np.array(['confident_target','budget_accept','confident_host']), 'decision_time':np.array([1.,2.,1.])}
    r=correction_metrics(labels,wrong,d)
    assert r['target_error_count']==2 and r['target_explicit_corrections']==1 and r['target_fallback_retained']==1


def test_adapter_drops_stale_results_and_releases_state():
    m=tiny_model().eval();sent=[];current={1:'a'}
    a=StreamingAdapter(m,lambda c:current.get(c),lambda c,r,action:sent.append((c,r,action)))
    a.start('a',1);current[1]='b'
    a.feed('a',0,np.arange(3000,dtype=np.int16),1.)
    assert sent==[] and 'a' not in a.states and a.logs[-1]['reason']=='natural_end'


def test_encoder_feature_dimensions():
    m=tiny_model().eval()
    with torch.no_grad():e=m.encode(torch.randn(2,1,1500))
    assert e.shape==(2,128)


def test_missing_first_window_is_not_originally_correct():
    d={'action':np.array(['reject','reject']), 'reason':np.array(['confident_host']*2), 'decision_time':np.array([1.,1.])}
    r=correction_metrics(np.array([0,0]),np.array([False,False]),d,first_correct=np.array([False,True]))
    assert r['new_target_false_rejects']==1


def test_latency_accumulates_only_until_terminal():
    p=np.array([[.5,.995,.01]]);v=np.ones((1,3),bool)
    d=terminal_decisions(p,v,np.array([3000,4500,6000]),np.array([10000]),'P',np.array([[.5,.5,9.]]))
    assert d['window_index'][0]==2 and d['decision_time'][0]==1.75


def test_natural_end_sends_no_action_and_is_separate_from_fallback():
    d=terminal_decisions(np.array([[.5]]),np.array([[False]]),np.array([3000]),np.array([2000]),'P')
    assert d['action'][0]=='none' and d['reason'][0]=='natural_end'
    r=correction_metrics(np.array([0]),np.array([False]),d,first_correct=np.array([False]))
    assert r['fallback_count']==0 and r['natural_end_count']==1


def test_adapter_ignores_chunks_after_invalid_input_terminal():
    sent=[];m=tiny_model().eval();a=StreamingAdapter(m,lambda c:'a',lambda *args:sent.append(args))
    a.start('a',1);a.feed('a',0,np.ones(3000,dtype=np.int16),1.)
    a.feed('a',3000,np.arange(1500,dtype=np.int16),2.)
    assert len(sent)==1 and sent[0]==(1,'a','accept') and len(a.logs)==1 and a.logs[0]['reason']=='input_fallback'


def test_adapter_decision_timestamp_includes_processing(monkeypatch):
    m=tiny_model().eval();a=StreamingAdapter(m,lambda c:'a',lambda *args:None)
    a.start('a',1)
    values=iter([0.,.02]);monkeypatch.setattr('poreloom.simulator_adapter.time.perf_counter',lambda:next(values))
    a.feed('a',0,np.arange(3000,dtype=np.int16),1.)
    assert a.logs[0]['decision_time']==1.02
