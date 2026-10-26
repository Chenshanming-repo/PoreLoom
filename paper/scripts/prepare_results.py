#!/usr/bin/env python3
"""Derive final-publication inputs from completed experiments; no model execution.

Set PORELOOM_RESULTS_ARCHIVE when the archive is outside paper/.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from signal_position_data import prepare as prepare_signal_positions
from length_eligibility import prepare as prepare_length_eligibility
from result_archive import ROOT, ARCHIVE
from plot_beta import prepare as prepare_beta
from analyze_matched_rejection import prepare as prepare_matched
from threshold_comparison import prepare as prepare_thresholds
FINAL = ARCHIVE/'results/final_20261014'
BUDGET = ARCHIVE/'results/budget_up_20261016/runs'
OUT = ROOT/'figure_data'
METHODS = ['poreloom','readcurrent','nanodeep_upstream','squigglenet']
NAMES = dict(zip(METHODS,['PoreLoom','ReadCurrent','NanoDeep','SquiggleNet']))
FAMILIES = ['Simulated','R9','R10 mixed','R10 sputum']
INPUTS = set()
def read(p):
    INPUTS.add(str(p.relative_to(ARCHIVE)))
    return json.loads(p.read_text())
def family(ds):
    return 'Simulated' if ds.startswith('sim') else 'R10 mixed' if ds.startswith('r10') else 'R10 sputum' if ds.startswith('sputum') else 'R9'
def write(df, name):
    df.to_csv(OUT/(name+'.csv'),index=False,float_format='%.15g')
    return df
def metrics(d):
    return dict(acc=d['acc'],frr=d.get('target_command_frr',d.get('frr')),host_pore=d.get('host_mean_pore_seconds',d.get('host_pore')),
                host_reject=d.get('host_command_reject',d.get('host_command_reject_rate')))
def tex_escape(s):
    return str(s).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&').replace('→',r'$\to$').replace('–','--')
def publication_unit(unit):
    dataset, repeat = unit.split('/r')
    if dataset == 'r9':
        return f'R9 / {repeat}'
    label = ('Simulated ' + dataset.removeprefix('sim_g') if dataset.startswith('sim_g')
             else 'Mixed ' + str(int(dataset.removeprefix('r10_run'))) if dataset.startswith('r10_run')
             else 'Sputum ' + dataset.removeprefix('sputum_g'))
    return f'{label} / {repeat}'

def table(df, cols, heads, path, caption, label):
    lines=[r'\begin{longtable}{'+'l'*len(cols)+'}',r'\caption{'+caption+r'}\label{'+label+r'}\\',r'\toprule',' & '.join(heads)+r'\\\midrule\endfirsthead',r'\toprule',' & '.join(heads)+r'\\\midrule\endhead',r'\bottomrule\endfoot']
    for _,row in df.iterrows():
        vals=[]
        for c in cols:
            v=row[c]
            vals.append(f'{v:.2f}' if isinstance(v,(float,np.floating)) else tex_escape(v))
        lines.append(' & '.join(vals)+r'\\')
    lines.append(r'\end{longtable}')
    (ROOT/'tables'/path).write_text('\n'.join(lines)+'\n')
def main():
    OUT.mkdir(exist_ok=True)
    (ROOT/'tables').mkdir(exist_ok=True)
    (ROOT/'review').mkdir(exist_ok=True)
    (ROOT/'provenance').mkdir(exist_ok=True)
    w=read(FINAL/'win6000/summary_zh.json'); rows=[]; seen=set(); history=[]
    for r in w:
        u=f"{r['dataset']}/r{r['rep']}"; f=family(r['dataset'])
        base=dict(unit=u,family=f,dataset=r['dataset'],replicate=r['rep'])
        if u not in seen:
            rows.append(dict(**base,method='poreloom',**metrics(r['poreloom']),n_host=r['poreloom']['n_host'],n_target=r['poreloom']['n_target']))
            seen.add(u)
        rows.append(dict(**base,method=r['tool'],**metrics(r['w6000']),n_host=r['w6000']['n_host'],n_target=r['w6000']['n_target']))
        for v in ['w3000','w6000']:
            history.append(dict(**base,method=r['tool'],variant=v,**metrics(r[v])))
    d=write(pd.DataFrame(rows),'main'); assert len(d)==48 and len(seen)==12
    write(pd.DataFrame(history),'window_baselines')
    means=write(d.groupby(['family','method'],sort=False)[['acc','frr','host_pore','host_reject']].mean().reset_index(),'main_means')
    INPUTS.update(prepare_length_eligibility())
    length_main=pd.read_csv(OUT/'main_length_matched.csv')
    write(length_main.groupby(['family','method'],sort=False)[['acc','frr','host_pore','host_reject']].mean().reset_index(),'main_length_matched_means')
    ab=read(FINAL/'ablation/report.json'); rows=[]
    for u,rs in ab.items():
        for r in rs:
            if r['variant'] not in {'main','window_cp','fixed_len','const_thr','calib_thr'}:continue
            q=dict(unit=u,family=family(u),variant=r['variant'],**metrics(r))
            for k in ['beta','dacc','dpore']:
                if k in r:q[k]=r[k]
            rows.append(q)
    a=write(pd.DataFrame(rows),'ablation_all')
    write(a[a.variant.isin(['main','window_cp'])],'encoder')
    write(a[a.variant.isin(['main','fixed_len','const_thr','calib_thr'])],'stopping')
    n=[]
    for u in sorted(seen):
        ds,rep=u.split('/')
        for m in METHODS[1:]:
            baseline=read(FINAL/'runs/main'/ds/f'{m}_{rep}'/'test/metrics.json')
            selected=[v for v in baseline['rows'] if v.get('rule')=='max_acc' and v.get('group')=='pooled' and v['interval']==.4 and v['service']=='measured']
            assert len(selected)==1
            facp=read(FINAL/'native_cp'/u/m/'metrics.json')['locked']
            for state,r in [('Native',selected[0]),('C-p',facp)]:
                n.append(dict(unit=u,family=family(u),method=m,state=state,**metrics(r)))
    nc=write(pd.DataFrame(n),'native_cp'); assert len(nc)==72
    outcome_keys=['acc','host_reject','frr','host_pore']
    paired=nc.pivot(index=['unit','family','method'],columns='state',values=outcome_keys)
    write(pd.DataFrame({k:paired[k]['C-p']-paired[k]['Native'] for k in outcome_keys}).reset_index(),'native_cp_deltas')
    reference=a[a.variant=='main'].assign(method='poreloom',state='Original')
    write(pd.concat([reference,nc[nc.state=='C-p']],ignore_index=True)[['unit','family','method','state']+outcome_keys],'native_cp_comparison')
    rows=[]
    for ds,bs in [('r10_run01',[6000,9000,12000,15000,18000]),('sputum_g1',[6000,9000,12000])]:
        for b in bs:
            for rep in range(3):
                p=BUDGET/ds/f'b{b}_r{rep}'/'test/metrics.json';r=read(p)
                for v in r['rows']:
                    if v.get('group')!='pooled':continue
                    q=dict(dataset=ds,budget=b,replicate=rep,beta=r['beta'],interval=v['interval'],service=v['service'],**metrics(v),n_host=v['n_host'],n_target=v['n_target'])
                    if v['interval']==.4 and v['service']=='nominal':rows.append(q)
    b=write(pd.DataFrame(rows),'budget');assert len(b)==24
    write(b.groupby(['dataset','budget'])[['acc','frr','host_pore']].mean().reset_index(),'budget_means')
    t=read(ARCHIVE/'results/stream_timing_20261016/rtx2080ti/timing.json'); assert '2080 Ti' in t['meta']['gpu']
    tr=write(pd.DataFrame(t['rows']),'timing_raw')
    med=[]
    for keys,g in tr.groupby(['scenario','chunk','batch','method']):
        sc,ch,ba,me=keys
        if sc=='mixed':
            med.append(dict(scenario=sc,chunk=ch,batch=ba,method=me,n=-1,median_ms=1000*g.seconds.median(),min_ms=1000*g.seconds.min(),max_ms=1000*g.seconds.max()))
        else:
            for n,g2 in g.groupby('n'):
                med.append(dict(scenario=sc,chunk=ch,batch=ba,method=me,n=n,median_ms=1000*g2.seconds.median(),min_ms=1000*g2.seconds.min(),max_ms=1000*g2.seconds.max()))
    tm=write(pd.DataFrame(med),'timing_medians')
    ratios=tm.pivot(index=['scenario','chunk','batch','n'],columns='method',values='median_ms').reset_index()
    ratios['speed_ratio']=ratios['scratch']/ratios['incremental'];write(ratios,'timing_ratios')
    l=pd.DataFrame(read(FINAL/'lso/analysis.json'))
    l=l.drop(columns=[c for c in l if '_ci' in c],errors='ignore')
    l=l[l.method.isin(METHODS)].copy()
    l['scope']=l.fold.map(lambda s:'Four-genus subset' if s=='Actinomycetota' else 'Genus')
    write(l,'lso')
    ir=read(FINAL/'icarust/aggregate.json')['rows']; no={(r['seed'],r['permille']):r['yield_target_bp_per_channel_hour'] for r in ir if r['strategy']=='No-AS'}
    iy=[]
    for r in ir:
        q={k:v for k,v in r.items() if not isinstance(v,(dict,list))}
        q['relative_yield']=r['yield_target_bp_per_channel_hour']/no[r['seed'],r['permille']]
        q['frr']=r['target_command_reject']['value'];q['host_reject']=r['host_effective_reject']['value']
        q['target_effective_reject']=r['target_effective_reject']['value']
        q['operational_acc']=(r['host_effective_reject']['events']+r['target_reads']-r['target_effective_reject']['events'])/r['reads']
        iy.append(q)
    iy=write(pd.DataFrame(iy),'icarust');assert len(iy)==45
    # Add the W6000 closed-loop runs without changing the original 45-run input.
    extension=ARCHIVE/'results/icarust_w6000_20261018'
    extended=read(extension/'aggregate.json')['rows']
    by_key={(r['seed'],r['permille'],r['strategy']):r for r in extended}
    assert len(by_key)==len(extended)==72
    for r in ir:
        original=by_key[r['seed'],r['permille'],r['strategy']]
        assert all(original[k]==v for k,v in r.items()), r['name']
    extra=[r for r in extended if r['strategy'].endswith('_w6000')]
    assert len(extra)==27
    combined=[]
    for r in ir+extra:
        q={k:v for k,v in r.items() if not isinstance(v,(dict,list)) and k!='source'}
        q['relative_yield']=r['yield_target_bp_per_channel_hour']/no[r['seed'],r['permille']]
        q['frr']=r['target_command_reject']['value']
        q['host_reject']=r['host_effective_reject']['value']
        q['target_effective_reject']=r['target_effective_reject']['value']
        q['operational_acc']=(r['host_effective_reject']['events']+r['target_reads']-r['target_effective_reject']['events'])/r['reads']
        q['command_latency_p50']=r['command_latency_s']['p50']
        combined.append(q)
    ix=write(pd.DataFrame(combined),'icarust_extended')
    simulator_names={'No-AS':'No-AS','PoreLoom':'PoreLoom','readcurrent':'ReadCurrent 3000',
                     'readcurrent_w6000':'ReadCurrent 6000','nanodeep':'NanoDeep 3000',
                     'nanodeep_w6000':'NanoDeep 6000','squigglenet':'SquiggleNet 3000',
                     'squigglenet_w6000':'SquiggleNet 6000'}
    summary=[]
    for strategy,display in simulator_names.items():
        for per in [10,100,300]:
            group=ix[(ix.strategy==strategy)&(ix.permille==per)]
            assert len(group)==3 and group.status.eq('complete').all()
            summary.append(dict(strategy=strategy,method=display,permille=per,n=3,
                relative_yield=group.relative_yield.mean(),yield_min=group.relative_yield.min(),
                yield_max=group.relative_yield.max(),host_effective_reject=group.host_reject.mean(),
                target_effective_reject=group.target_effective_reject.mean(),operational_acc=group.operational_acc.mean(),
                target_command_frr=group.frr.mean(),host_pore=group.host_mean_pore_seconds.mean(),
                command_latency_p50=group.command_latency_p50.mean()))
    simulator_summary=write(pd.DataFrame(summary),'icarust_summary')
    # Register closed-loop settings and the saved outcomes.
    for path in [extension/'plan.json',extension/'check.json']:
        read(path)
    for method in METHODS[1:]:
        for name in ['selection.json','window.json']:
            read(extension/'models'/f'{method}_r0'/name)
    for path in (extension/'live').glob('*/read_outcomes.jsonl'):
        INPUTS.add(str(path.relative_to(ARCHIVE)))
        read(path.parent/'manifest.json')
    # Supplement tables keep all units, avoiding false biological replication.
    for name,df in [('main',d),('encoder',a[a.variant.isin(['main','window_cp'])]),('stopping',a[a.variant.isin(['main','fixed_len','const_thr','calib_thr'])]),('native_cp',nc)]:
        q=df.copy();q['ACC']=100*q.acc;q['FRR']=100*q.frr;q['Host']=q.host_pore;q['Reject']=100*q.host_reject
        q['unit']=q.unit.map(publication_unit)
        if name=='main':q['method']=q.method.map(NAMES)
        if name=='encoder':q['method']=q['variant'].map({'main':'Streaming prefix','window_cp':'Sliding window'})
        if name=='stopping':q['method']=q['variant'].map({'main':'FACP','fixed_len':'Fixed length','const_thr':'Constant threshold','calib_thr':'Length-calibrated'})
        cols=['unit','method'];heads=['Unit','Method']
        if name=='native_cp':
            q['method']=q.method.map(NAMES)
            q['state']=q.state.map({'Native':'Single window','C-p':'FACP'})
            cols+=['state'];heads+=['Policy']
        table(q,cols+['Reject','ACC','FRR','Host'],heads+[r'Host rej. (\%)',r'ACC (\%)',r'Target FRR (\%)','Host (s)'],name+'.tex',{'main':r'All-read operational comparison, including short reads. Baselines use 6000-sample windows; PoreLoom uses the default 6000-sample budget. The length-restricted analysis is an additional control.','encoder':'Encoding scheme comparison, with separately fitted factorized action-cost policies (FACP).','stopping':'Stopping rules on the same frozen streaming encoder.','native_cp':'Frozen 3000-sample baselines with single-window decisions or repeated sliding-window scoring and FACP.'}[name],'tab:s_'+name)
    q=b.copy();q['unit']=q.dataset.str.replace('r10_run01','Mixed',regex=False).str.replace('sputum_g1','Sputum',regex=False);q['ACC']=q.acc*100;q['FRR']=q.frr*100
    table(q,['unit','budget','replicate','ACC','FRR','host_pore'],['Data','Budget','Rep.',r'ACC (\%)',r'FRR (\%)','Host (s)'],'budget.tex','All maximum-budget training repeats. Budget excludes the skipped 1500 samples.','tab:s_budget')
    q=l[l.method.isin(METHODS)].copy();q['group']=q.group.str.removeprefix('g');q['fold']=q.fold.replace({'Actinomycetota':'Four genera'});q['method']=q.method.map(NAMES)
    for c in ['frr_main','frr_fold','dfrr']:q[c]=100*q[c]
    table(q,['group','fold','method','n_held','frr_main','frr_fold','dfrr'],['Group','Holdout','Method',r'$n_T$',r'Full (\%)',r'Out (\%)',r'$\Delta$ (pp)'],'lso.tex','Held-out target false rejection in the three sputum groups. Full and held-out models are each selected on their available DEV-B.','tab:s_lso')
    # Additional read-length eligibility control; main results use all TEST reads.
    q=pd.read_csv(OUT/'length_eligibility.csv').query("split == 'test'").copy()
    q['unit']=q.dataset.map(lambda x: publication_unit(x+'/r0').split(' / ')[0])
    q['Host']=q.host_coverage*100;q['Target']=q.target_coverage*100
    q['Median']=q.target_median_samples/1000
    table(q,['unit','n_host_eligible','n_target_eligible','Host','Target','Median'],
          ['Data','Long hosts','Long targets',r'Host (\%)',r'Target (\%)',r'Target median (k)'],
          'length_eligibility.tex','TEST coverage at raw length 7500 samples. Percentages use all reads of the corresponding class; target median length is in thousands of raw samples. R9 is one shared TEST set used for all three training repeats.','tab:s_length_eligibility')
    q=length_main.copy();q['unit']=q.unit.map(publication_unit);q['method']=q.method.map(NAMES)
    q['Reject']=100*q.host_reject;q['ACC']=100*q.acc;q['FRR']=100*q.frr
    table(q,['unit','method','Reject','ACC','FRR','host_pore'],
          ['Unit','Method',r'Host rej. (\%)',r'ACC (\%)',r'Target FRR (\%)','Host (s)'],
          'length_main.tex',r'Shared length-subset main comparison. Every method uses the same TEST reads with raw length $\geq7500$ within each unit; all baselines here have 6000-sample windows. Read counts are given in Table~\ref{tab:s_length_eligibility}. No-command reads of sufficient length remain in the denominators.','tab:s_length_main')
    q=pd.read_csv(OUT/'length_metrics.csv').query("dataset == 'r9'").copy()
    q['method']=q.apply(lambda r: NAMES[r['method']]+(' '+r['variant'][1:] if r['variant'].startswith('w') else ''),axis=1)
    q['scope']=q.scope.map({'all':'All','ge7500':'Long','lt7500':'Short'})
    q['Reject']=100*q.host_reject;q['ACC']=100*q.acc;q['FRR']=100*q.frr
    table(q,['replicate','method','scope','Reject','FRR','host_pore','ACC'],
          ['Rep.','Method / window','Reads',r'Host rej. (\%)',r'FRR (\%)','Host (s)',r'ACC (\%)'],
          'r9_length_strata.tex',r'R9 outcomes by read length, with the original DEV-B selections and 0.4-s callbacks. All: 20000 reads (10000 host, 10000 target); Long: raw length $\geq7500$, 15339 reads (8079 host, 7260 target); Short: raw length $<7500$, 4661 reads (1921 host, 2740 target). All repeats use these same reads.','tab:s_r9_length_strata')
    yield_rows=[]
    for strategy,display in simulator_names.items():
        row={'method':display}
        for per in [10,100,300]:
            r=simulator_summary[(simulator_summary.strategy==strategy)&(simulator_summary.permille==per)].iloc[0]
            row[str(per)]=f'{r.relative_yield:.2f} [{r.yield_min:.2f}, {r.yield_max:.2f}]'
        yield_rows.append(row)
    table(pd.DataFrame(yield_rows),['method','10','100','300'],
          ['Strategy / window',r'1\% target',r'10\% target',r'30\% target'],
          'icarust_yield.tex',r'Closed-loop relative target yield, reported as the mean [minimum, maximum] of three seed-matched ratios to nonselective sequencing. Each run uses 512 channels for 1200 s. Baseline labels give fixed window lengths in samples; PoreLoom uses a maximum usable budget of 6000 samples. These are simulator seeds, not training or biological replicates.','tab:s_icarust_yield')
    q=simulator_summary.sort_values('permille',kind='stable').copy()
    q['fraction']=q.permille.map(lambda x:f'{x/10:g}')
    q['HostReject']=q.host_effective_reject.map(lambda x:f'{100*x:.2f}')
    q['FRR']=q.target_effective_reject.map(lambda x:f'{100*x:.2f}')
    q['Host']=q.host_pore.map(lambda x:f'{x:.3f}')
    q['Latency']=q.command_latency_p50.map(lambda x:f'{1000*x:.1f}')
    table(q,['fraction','method','HostReject','FRR','Host','Latency'],
          [r'Target (\%)','Strategy / window',r'Host eff. (\%)',r'FRR (\%)','Host (s)','Latency (ms)'],
          'icarust_operations.tex',r'Closed-loop operational outcomes, averaged across three simulator seeds. Host eff. and FRR use effective host and target ejections; Host is mean host pore occupancy. Latency is the mean of the three within-run median command latencies.','tab:s_icarust_operations')
    INPUTS.update(prepare_signal_positions())
    prepare_beta()
    prepare_matched()
    prepare_thresholds()
    inp=[dict(path=p,sha256=hashlib.sha256((ARCHIVE/p).read_bytes()).hexdigest()) for p in sorted(INPUTS)]
    (ROOT/'provenance/derived_inputs.json').write_text(json.dumps(inp,indent=2)+'\n')
    # Numeric macros make article claims traceable and keep prose aligned with rebuilt data.
    macros={}; claims=[]
    def macro(k,v,fmt,source,rule):
        macros[k]=format(v,fmt);claims.append(dict(macro=k,value=float(v),display=macros[k],source=source,aggregation=rule))
    for fam,short in zip(FAMILIES,['Sim','Rnine','Mix','Sputum']):
        for m in METHODS:
            r=means[(means.family==fam)&(means.method==m)].iloc[0]
            tag=dict(poreloom='PL',readcurrent='RC',nanodeep_upstream='ND',squigglenet='SN')[m]
            for key,suffix,scale,fmt in [('acc','Acc',100,'.2f'),('host_reject','Reject',100,'.2f'),('frr','Frr',100,'.2f'),('host_pore','Pore',1,'.3f')]:macro(short+tag+suffix,r[key]*scale,fmt,'figure_data/main.csv',f'equal mean across 3 {fam} units; all TEST reads')
    pl_rejection=means[means.method=='poreloom'].host_reject
    macro('MainPLMinReject',pl_rejection.min()*100,'.2f','figure_data/main.csv','minimum of four PoreLoom family means; all TEST reads')
    macro('MainPLMaxReject',pl_rejection.max()*100,'.2f','figure_data/main.csv','maximum of four PoreLoom family means; all TEST reads')
    prefix=pd.read_csv(OUT/'fixed_prefix.csv');positions=pd.read_csv(OUT/'decision_position_summary.csv')
    for fam,tag in zip(FAMILIES,['Sim','Rnine','Mix','Sputum']):
        for length,point in [(432,'Base'),(6000,'Max')]:
            r=prefix[(prefix.family==fam)&(prefix.usable_samples==length)]
            macro(tag+'Prefix'+point+'Acc',r.acc.mean()*100,'.2f','figure_data/fixed_prefix.csv','equal mean of 3 unit accuracies at fixed usable length '+str(length))
        r=positions[(positions.family==fam)&(positions.category=='all')]
        for key,suffix in [('fraction_by_3000','DecisionByThree'),('no_decision_fraction','NoDecision')]:
            macro(tag+suffix,r[key].mean()*100,'.2f','figure_data/decision_position_summary.csv','equal mean across 3 units; completed decisions only for position fractions')
    for sc,tag,n in [('mixed','Mixed',-1),('aligned','Aligned',19200)]:
        r=ratios[(ratios.scenario==sc)&(ratios.chunk==1600)&(ratios.batch==512)&(ratios.n==n)].iloc[0]
        for k,label,fmt in [('incremental','Inc','.1f'),('scratch','Full','.1f'),('speed_ratio','Ratio','.1f')]:macro('Timing'+tag+label,r[k],fmt,'figure_data/timing_raw.csv','median across 3 timed repeats; ratio of medians')
    for ds,tag,bs in [('r10_run01','MixBudget',[6000,18000]),('sputum_g1','SputumBudget',[6000,12000])]:
        for budget,point in zip(bs,['Base','Max']):
            rr=b[(b.dataset==ds)&(b.budget==budget)]
            for k,label,scale,fmt in [('acc','Acc',100,'.2f'),('frr','Frr',100,'.2f'),('host_pore','Pore',1,'.3f')]:macro(tag+point+label,rr[k].mean()*scale,fmt,'figure_data/budget.csv','mean across 3 training repeats on one fixed dataset')
    for per,tag in [(10,'One'),(100,'Ten'),(300,'Thirty')]:
        r=iy[(iy.strategy=='PoreLoom')&(iy.permille==per)];macro('Yield'+tag,r.relative_yield.mean(),'.2f','figure_data/icarust.csv','mean of 3 seed-matched ratios to No-AS')
        r=simulator_summary[(simulator_summary.strategy.str.endswith('_w6000'))&(simulator_summary.permille==per)]
        macro('YieldWsixBest'+tag,r.relative_yield.max(),'.2f','figure_data/icarust_summary.csv','maximum W6000 baseline mean across 3 seed-matched ratios to No-AS')
    (ROOT/'numbers.tex').write_text('% Generated by scripts/prepare_results.py\n'+'\n'.join(chr(92)+'newcommand{'+chr(92)+k+'}{'+v+'}' for k,v in macros.items())+'\n')
    (ROOT/'provenance/numerical_claims.json').write_text(json.dumps(claims,indent=2)+'\n')
    print(json.dumps(dict(main=len(d),budget=len(b),native_cp=len(nc),lso=len(l),icarust=len(iy),icarust_extended=len(ix),timing=len(tr),input_files=len(inp)),indent=2))
if __name__=='__main__':main()
