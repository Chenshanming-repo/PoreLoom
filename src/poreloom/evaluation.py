"""Paired terminal policy accounting; project labels target0/host1."""
import numpy as np
from .policy import decide


def terminal_decisions(p_host,valid,ends,raw_length,group,service=None):
    p=np.asarray(p_host);v=np.asarray(valid,bool);n,k=p.shape;ends=np.asarray(ends)
    if v.shape!=p.shape or ends.shape!=(k,) or len(raw_length)!=n or group not in ('B0','B1','B2','P'):raise ValueError('invalid terminal inputs')
    service=np.zeros_like(p,dtype=float) if service is None else np.asarray(service,float)
    if service.shape!=p.shape or not np.isfinite(service).all() or (service<0).any():raise ValueError('invalid service times')
    d=dict(action=np.full(n,'accept',dtype='<U6'),reason=np.full(n,'input_fallback',dtype='<U24'),window_index=np.zeros(n,np.int8),signal_end=np.zeros(n,np.int64),decision_time=np.zeros(n),p_host=np.full(n,np.nan))
    for i in range(n):
        completion=0.
        for j in range(1 if group in ('B0','B1') else k):
            arrival=min(int(raw_length[i]),int(ends[j]))
            completion=max(completion,arrival/4000)+service[i,j]
            d['window_index'][i]=j+1;d['signal_end'][i]=arrival;d['decision_time'][i]=completion
            if not v[i,j] or not np.isfinite(p[i,j]):
                d['reason'][i]='natural_end' if raw_length[i]<ends[j] else 'input_fallback'
                if d['reason'][i]=='natural_end':d['action'][i]='none'
                break
            ph=float(p[i,j]);d['p_host'][i]=ph
            if group=='B0':action,reason=('reject' if ph>=.5 else 'accept'),'original_argmax'
            elif group=='B1':action,reason=('reject','confident_host') if ph>=.99 else ('accept','confident_target' if 1-ph>=.9 else 'threshold_accept')
            else:action,reason=decide(ph,1-ph,j+1,k)
            if action!='defer':d['action'][i]=action;d['reason'][i]=reason;break
    return d


def correction_metrics(labels,first_wrong,d,first_correct=None):
    y=np.asarray(labels);wrong=np.asarray(first_wrong,bool);target=wrong&(y==0);host=wrong&(y==1)
    original_correct=~wrong if first_correct is None else np.asarray(first_correct,bool)
    explicit_target=(d['action']=='accept')&(d['reason']=='confident_target');explicit_host=(d['action']=='reject')&(d['reason']=='confident_host')
    fallback=np.isin(d['reason'],['budget_accept','input_fallback','timeout_accept','threshold_accept'])
    result=dict(target_error_count=int(target.sum()),host_error_count=int(host.sum()),target_explicit_corrections=int((target&explicit_target).sum()),host_explicit_corrections=int((host&explicit_host).sum()),target_fallback_retained=int((target&fallback&(d['action']=='accept')).sum()),target_retained=int((target&(d['action']=='accept')).sum()),target_false_rejects=int(((y==0)&(d['action']=='reject')).sum()),host_rejects=int(((y==1)&(d['action']=='reject')).sum()),target_total=int((y==0).sum()),host_total=int((y==1).sum()),fallback_count=int(fallback.sum()),natural_end_count=int((d['reason']=='natural_end').sum()),new_target_false_rejects=int(((y==0)&original_correct&(d['action']=='reject')).sum()),decision_p50_seconds=float(np.median(d['decision_time'])),decision_p95_seconds=float(np.percentile(d['decision_time'],95)))
    for side in ('target','host'):
        count=result[f'{side}_error_count'];result[f'{side}_explicit_correction_rate']=result[f'{side}_explicit_corrections']/count if count else None
    result['target_false_reject_rate']=result['target_false_rejects']/max(result['target_total'],1)
    result['host_reject_rate']=result['host_rejects']/max(result['host_total'],1)
    return result
