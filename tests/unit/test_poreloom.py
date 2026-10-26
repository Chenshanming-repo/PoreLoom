import numpy as np
import pytest
import torch
from torch import nn
from poreloom.windowing import normalize_windows, split_windows, SignalBuffer
from poreloom.model import PoreLoom, prefix_loss


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


def test_encoder_feature_dimensions():
    m=tiny_model().eval()
    with torch.no_grad():e=m.encode(torch.randn(2,1,1500))
    assert e.shape==(2,128)
