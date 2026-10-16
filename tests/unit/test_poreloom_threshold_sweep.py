import numpy as np
from poreloom.threshold_sweep import exact_final_sweep
from poreloom.hostfloor import hard_components, component_metrics


def test_exact_sweep_matches_full_causal_replay_with_short_and_stopped_reads():
    target = np.array([False,False,False,True,True])
    available = np.array([[1,1,1,1],[1,1,1,0],[1,1,1,1],[1,1,1,1],[0,0,0,0]],bool)
    actions = np.full((5,4),2);actions[:,-1]=0;actions[2,0]=1
    probability = np.array([.0007,.8,.1,.0009,1.],np.float32)
    ends = {k:np.zeros(5) for k in ['command_reject','effective_reject','pore_seconds','compute_seconds']}
    ends['pore_seconds'][:] = [10.,1.5,8.,20.,.5]
    ends['compute_seconds'][:] = [.04,.03,.04,.04,0.]
    branches = {k:np.repeat(np.repeat(v[:,None,None],4,axis=1),3,axis=2) for k,v in ends.items()}
    branches['command_reject'][:,:,1]=1.;branches['effective_reject'][:,:,1]=1.
    branches['pore_seconds'][:,:,1]=np.arange(1,5)[None,:]/2
    branches['compute_seconds'][:]=np.arange(1,5)[None,:,None]/100
    result = exact_final_sweep(probability,actions,available,branches,ends,target)
    assert .0007 in result['thresholds'] and .0009 in result['thresholds']
    for i,threshold in enumerate(result['thresholds']):
        deployed=actions.copy();deployed[:,-1]=np.where(probability>=np.float32(threshold),0,1)
        expected=component_metrics(hard_components(deployed,available,branches,ends),target)
        for key,values in result['metrics'].items():
            assert np.isclose(values[i],expected[key],atol=1e-12),(threshold,key)


def test_boundary_uses_accept_at_equal_probability_and_includes_full_denominators():
    available=np.ones((2,4),bool);actions=np.full((2,4),2);actions[:,-1]=0
    ends={k:np.zeros(2) for k in ['command_reject','effective_reject','pore_seconds','compute_seconds']}
    branches={k:np.zeros((2,4,3)) for k in ends}
    branches['command_reject'][:,:,1]=1.;branches['effective_reject'][:,:,1]=1.
    result=exact_final_sweep(np.array([.001,1.],np.float32),actions,available,branches,ends,np.array([False,True]))
    i=np.flatnonzero(result['thresholds']==np.float32(.001))[0]
    assert result['metrics']['host_effective_reject_rate'][i]==0.
    assert result['metrics']['target_command_frr'][-1]==0.
