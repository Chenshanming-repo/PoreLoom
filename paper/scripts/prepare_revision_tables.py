#!/usr/bin/env python3
"""Rebuild all current tables from the frozen 2026-11-04 data snapshot."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/revision_20261104'
FAMILIES=['Simulated','R9','R10 mixed','R10 sputum']
METHODS={'poreloom':'PoreLoom','readcurrent':'ReadCurrent','nanodeep_upstream':'NanoDeep','squigglenet':'SquiggleNet'}
METRICS=['acc','host_reject','frr','host_pore']
HEADERS=[r'Accuracy (\%)',r'Hosts rejected (\%)',r'Targets rejected (\%)','Host pore time (s)']
def read(name):return pd.read_csv(DATA/(name+'.csv'))
def row(values):return ' & '.join(values)+r'\\'
def write(name,spec,headers,rows,sep=None,upper=None):
    lines=[r'\begin{tabular}{@{}'+spec+r'@{}}',r'\toprule']+(upper or [])+[row(headers)+r'\midrule',*rows,r'\bottomrule',r'\end{tabular}']
    if sep:lines=[r'\begingroup\setlength{\tabcolsep}{'+sep+'}',*lines,r'\endgroup']
    (ROOT/'tables'/name).write_text('\n'.join(lines)+'\n')
def cell(avg,sd,metric,bold=False):
    scale,digits=(1,3) if metric=='host_pore' else (100,2)
    value=f'{avg*scale:.{digits}f} \\pm {sd*scale:.{digits}f}'
    return '$'+(r'\mathbf{'+value+'}' if bold else value)+'$'
def cells(r,best=None,metrics=METRICS):
    return [cell(r[m+'_mean'],r[m+'_sd'],m,best is not None and np.isclose(r[m+'_mean'],best[m],atol=1e-12,rtol=0)) for m in metrics]
def best_values(frame):return {m:(frame[m+'_mean'].max() if m in ['acc','host_reject'] else frame[m+'_mean'].min()) for m in METRICS}
def build():
    d=read('table1_new');body=[]
    for family in FAMILIES:
        q=d[d.family==family];best=best_values(q)
        for i,r in enumerate(q.to_dict('records')):
            m=r['method'];window=int(r['window'])
            body.append(row([family if i==0 else '',METHODS[m] if m=='poreloom' or window==3000 else '',('Budget ' if m=='poreloom' else 'Window ')+str(window)]+cells(r,best)))
        if family!=FAMILIES[-1]:body.append(r'\addlinespace')
    write('main_overall.tex','lllrrrr',['Dataset','Method','Observation']+HEADERS,body,'3.5pt')
    d=read('table3_fixed_ext_summary');body=[];methods=['facp','d0.01','d0.05','d0.1','d0.2','d0.3']
    decisions=read('table3_fixed_ext_decisions').set_index(['family','method'])
    for family in FAMILIES:
        q=d[(d.family==family)&d.method.isin(methods)].set_index('method');best=best_values(q)
        for i,method in enumerate(methods):
            label='FACP' if method=='facp' else r'Threshold, $\delta='+method[1:]+'$'
            body.append(row([family if i==0 else '',label]+cells(q.loc[method],best)))
        if family!=FAMILIES[-1]:body.append(r'\addlinespace')
    write('supp_stopping.tex','llrrrr',['Dataset','Method']+HEADERS,body,'4pt')
    body=[]
    for family in FAMILIES:
        vals=[]
        for method in methods:
            timing=decisions.loc[(family,method)]
            vals.append(cell(timing.at_budget_mean,timing.at_budget_sd,'at_budget'))
        body.append(row([family]+vals))
    headers=['Dataset','FACP']+[r'$\delta='+m[1:]+'$' for m in methods[1:]]
    upper=[row(['',r'\multicolumn{6}{c}{Forced decision (\%)}']),r'\cmidrule(lr){2-7}']
    write('supp_forced_decision.tex','lrrrrrr',headers,body,'4pt',upper)
    d=read('tableS1_counts');body=[]
    for family in FAMILIES:
        group_rows=d[d.family==family]
        if family=='R9':
            # The three seed repeats share one data group, listed once as Group 1.
            group_rows=group_rows.drop_duplicates(subset=['dataset','train','dev_a','test'])
        for i,r in enumerate(group_rows.itertuples(index=False)):
            body.append(row([family if i==0 else '',str(i+1),str(r.train),str(r.dev_a),str(r.test)]))
        if family!=FAMILIES[-1]:body.append(r'\addlinespace')
    write('supp_partitions_summary.tex','llrrr',['Dataset','Group','Training','Validation','Test'],body)
    d=read('tableS2_summary');body=[]
    for family in FAMILIES:
        q=d[d.family==family].set_index('method')
        for i,m in enumerate(list(METHODS)[1:]):body.append(row([family if i==0 else '',METHODS[m]]+cells(q.loc[m])))
        if family!=FAMILIES[-1]:body.append(r'\addlinespace')
    write('supp_baseline_facp.tex','llrrrr',['Dataset','Method']+HEADERS,body)
    d=read('budget_summary');body=[]
    for ds,label in [('r10_run01','R10 mixed, group 1'),('sputum_g1','R10 sputum, group 1')]:
        for i,r in enumerate(d[d.dataset==ds].sort_values('budget').to_dict('records')):
            body.append(row([label if i==0 else '',str(r['budget'])]+cells(r,metrics=METRICS+['host_forced'])))
        if ds=='r10_run01':body.append(r'\addlinespace')
    hdr=[r'\shortstack{'+h.replace(' (',r'\\(')+'}' for h in HEADERS]
    write('supp_budget_summary.tex','llrrrrr',['Data group','Budget']+hdr+[r'\shortstack{Hosts decided\\at budget (\%)}'],body)
    aggregate=json.loads((DATA/'icarust_aggregate.json').read_text())['table']
    sim_methods=[('No-AS','Nonselective','---'),('PoreLoom_beta1','PoreLoom','Budget 6000')]
    for m,lab in [('readcurrent','ReadCurrent'),('nanodeep','NanoDeep'),('squigglenet','SquiggleNet')]:
        sim_methods.extend([(m+'_native',lab,'Window 3000'),(m+'_w6000_native',lab,'Window 6000')])
    body=[]
    for strategy,lab,obs in sim_methods[1:]:
        vals=[]
        for fraction in [10,100,300]:
            avg,sd=aggregate[f'{fraction}/{strategy}']['enrichment'];value=f'{avg:.2f} $\\pm$ {sd:.2f}'
            vals.append(r'\textbf{'+value+'}' if strategy=='PoreLoom_beta1' else value)
        body.append(row([lab if not 'w6000' in strategy else '',obs]+vals))
    write('closed_loop.tex','llrrr',['Method','Observation',r'1\%',r'10\%',r'30\%'],body,'3.5pt')
    body=[]
    for fraction in [10,100,300]:
        for i,(strategy,lab,obs) in enumerate(sim_methods):
            r=aggregate[f'{fraction}/{strategy}'];vals=[cell(*r[key],metric) for key,metric in zip(['acc','host_eject','target_eject','host_pore'],METRICS)]
            body.append(row([str(fraction//10)+r'\%' if i==0 else '',lab,obs]+vals))
        if fraction!=300:body.append(r'\addlinespace')
    hdr=[r'\shortstack{'+h.replace(' (',r'\\(').replace('rejected','ejected')+'}' for h in HEADERS]
    write('supp_simulation_summary.tex','lllrrrr',['Targets','Method','Observation']+hdr,body,'3.5pt')
    d=read('lso_rows');d=d[d.version.isin(['new','base'])];assert len(d)==48
    folds=['Prevotella','Streptococcus','Fusobacterium','Actinomycetota']
    for all_bacterial in [False,True]:
        body=[]
        for fold in folds:
            label='Four genera' if fold=='Actinomycetota' else r'\textit{'+fold+'}'
            for i,(method,lab) in enumerate(METHODS.items()):
                q=d[(d.fold==fold)&(d.method==method)].sort_values('group');assert len(q)==3
                fields=['bact_fold','d_bact'] if all_bacterial else ['taxon_fold','d_taxon','host_fold','d_host'];vals=[]
                for key in fields:vals += [f'${v*100:+.2f}$' if key.startswith('d_') else f'${v*100:.2f}$' for v in q[key]]
                body.append(row([label if i==0 else '',lab]+vals))
            if fold!=folds[-1]:body.append(r'\addlinespace')
        if all_bacterial:
            titles=[r'Rejection after exclusion (\%)',r'\shortstack{$\Delta$Rejection\\(\%)}'];name='supp_all_bacterial_rejection.tex';spec=r'll*{6}{w{r}{28pt}}';sep=None
        else:
            titles=[r'\shortstack{Held-out taxon\\rejection (\%)}',r'\shortstack{$\Delta$Taxon rejection\\(\%)}',r'Host rejection (\%)',r'\shortstack{$\Delta$Host rejection\\(\%)}'];name='supp_lso_summary.tex';spec='ll'+'r'*12;sep='3pt'
        upper=[row(['','']+[r'\multicolumn{3}{c}{'+t+'}' for t in titles]),''.join(r'\cmidrule(lr){'+f'{3+3*i}-{5+3*i}'+'}' for i in range(len(titles)))]
        write(name,spec,['Held-out taxa','Method']+['G1','G2','G3']*len(titles),body,sep,upper)
    print('Rebuilt 3 main tables and 7 supplementary tables from frozen revision data.')
if __name__=='__main__':build()
