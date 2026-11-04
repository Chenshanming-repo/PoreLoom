#!/usr/bin/env python3
"""Verify current publication assets and frozen data without manuscript or TeX files."""
from pathlib import Path
import hashlib,json,re
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/revision_20261104'
METRICS=['acc','host_reject','frr','host_pore']
report={'revision_date':'2026-11-04','aggregation':'Equal-weight means and sample SD (ddof=1), n=3'}
def read(n):
    d=pd.read_csv(DATA/(n+'.csv'))
    return d[d.rule.isin(['beta1','native'])] if n=='table1_units' else d
def check_stats(units,summary,keys,metrics):
    summary=summary.set_index(keys);checked=0
    for key,q in units.groupby(keys):
        if len(keys)==1:key=key[0] if isinstance(key,tuple) else key
        assert len(q)==3,(keys,key,len(q))
        for m in metrics:
            np.testing.assert_allclose([q[m].mean(),q[m].std(ddof=1)],[summary.loc[key,m+'_mean'],summary.loc[key,m+'_sd']],atol=1e-11,rtol=1e-10)
            checked+=1
    return checked
checks=0
for u,s,k,m in [('table1_units','table1_new',['family','method','window'],METRICS),('table3_fixed_units','table3_fixed_summary',['family','method'],METRICS),('tableS2_units','tableS2_summary',['family','method'],METRICS),('tableS4_units','tableS4_summary',['family'],[p+'_'+m for p in ['s','w','d'] for m in METRICS]),('budget_units','budget_summary',['dataset','budget'],METRICS+['host_forced']),('figS_beta_units','figS_beta_summary',['family','beta'],METRICS),('fig2b_cdf_units','fig2b_cdf_summary',['family','used'],['frac'])]:
    checks+=check_stats(read(u),read(s),k,m)
checks+=check_stats(read('table3_fixed_ext_units'),read('table3_fixed_ext_summary'),['family','method'],METRICS)
report['mean_sd_metric_groups_verified']=checks
for name,meta in json.loads((DATA/'provenance.json').read_text()).items():
    assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==meta['sha256'],name
report['source_snapshot_hashes_verified']=True
# The beta=1 results must equal the main policy in each of the 12 units.
main_units=read('table1_units');main_units=main_units[main_units.method=='poreloom'].copy()
main_units['unit']=main_units.dataset+'/r'+main_units.rep.astype(str)
beta=read('figS_beta_units')
assert len(beta)==504 and beta.groupby('family').beta.nunique().eq(42).all()
report['beta_sweep_unit_rows']=len(beta)
beta=beta[beta.beta==1]
assert len(beta)==12
for r in beta.itertuples(index=False):
    q=main_units[(main_units.family==r.family)&(main_units.unit==r.unit)];assert len(q)==1
    np.testing.assert_allclose(q[METRICS].to_numpy()[0],[getattr(r,m) for m in METRICS],atol=1e-12,rtol=0)
report['beta1_units_equal_main']=len(beta)
# Recompute all closed-loop results from the 3 seed rows, including enrichment.
a=json.loads((DATA/'icarust_aggregate.json').read_text())
for key,summary in a['table'].items():
    fraction,strategy=key.split('/');q=[r for r in a['rows'] if r['permille']==int(fraction) and r['strategy']==strategy]
    assert len(q)==3
    for r in q:
        base=next(b for b in a['rows'] if b['strategy']=='No-AS' and b['seed']==r['seed'] and b['permille']==r['permille'])
        assert np.isclose(r['enrichment'],r['target_bases_retained']/base['target_bases_retained'])
        expected=(r['host_reads']*r['host_eject']+r['target_reads']*(1-r['target_eject']))/r['reads']
        assert np.isclose(expected,r['acc'])
    for m in ['enrichment','acc','host_eject','target_eject','host_pore']:
        vals=[r[m] for r in q];np.testing.assert_allclose([np.mean(vals),np.std(vals,ddof=1)],summary[m],atol=1e-12)
report['closed_loop_seed_rows_verified']=len(a['rows'])
active_rows=[r for r in a['rows'] if r['strategy']!='PoreLoom']
assert len(active_rows)==72
report['closed_loop_active_seed_rows']=len(active_rows)
# Match current table values against the revision document's companion numeric record.
numbers=json.loads((DATA/'paper_numbers.json').read_text())
def pm(line):return re.findall(r'([0-9]+\.[0-9]+)\s*(?:\$)?\\pm(?:\$)?\s*([0-9]+\.[0-9]+)',line)
def expected_pm(vals):return [tuple(v.split(' ± ')) for v in vals]
def table_lines(name,needle=None):
    lines=(ROOT/'tables'/name).read_text().splitlines()
    return [x for x in lines if r'\pm' in x and (needle is None or needle in x)]
for line,values in zip(table_lines('main_overall.tex','PoreLoom'),numbers['table1_poreloom'].values()):assert pm(line)==expected_pm(values)
assert pm(table_lines('closed_loop.tex','PoreLoom')[0])==expected_pm(numbers['table2_poreloom'])
for line,values in zip(table_lines('supp_simulation_summary.tex','PoreLoom'),numbers['tableS5_poreloom'].values()):assert pm(line)==expected_pm(values)
for name,key in [('supp_baseline_facp.tex','tableS2'),('supp_budget_summary.tex','tableS3')]:
    expected=[v for family in numbers[key].values() for v in family.values()]
    actual=table_lines(name);assert len(actual)==len(expected),(name,len(actual),len(expected))
    for line,vals in zip(actual,expected):assert pm(line)==expected_pm(vals),(name,line,vals)
for name,key in [('supp_lso_summary.tex','tableS6_poreloom'),('supp_all_bacterial_rejection.tex','tableS7_poreloom')]:
    actual=[x for x in (ROOT/'tables'/name).read_text().splitlines() if '& PoreLoom &' in x]
    for line,vals in zip(actual,numbers[key].values()):assert re.findall(r'\$([+-]?\d+\.\d+)\$',line)==vals
# Table 3 now uses the extended export, not the superseded companion table.
stopping_methods=['facp','d0.01','d0.05','d0.1','d0.2','d0.3']
stopping_families=['Simulated','R9','R10 mixed','R10 sputum']
stopping=read('table3_fixed_ext_units')
decisions=read('table3_fixed_ext_decisions')
decision_stats=decisions.set_index(['family','method'])
assert decision_stats.index.is_unique and (decisions.n==3).all()
actual=table_lines('supp_stopping.tex')
assert len(actual)==len(stopping_families)*len(stopping_methods)
assert r'\delta=0.4' not in (ROOT/'tables/supp_stopping.tex').read_text()
for i,family in enumerate(stopping_families):
    q=stopping[(stopping.family==family)&stopping.method.isin(stopping_methods)]
    means=q.groupby('method')[METRICS].mean()
    sd=q.groupby('method')[METRICS].std(ddof=1)
    for j,method in enumerate(stopping_methods):
        line=actual[i*len(stopping_methods)+j]
        assert ('FACP' if method=='facp' else r'\delta='+method[1:]+'$') in line
        expected=[]
        for col,m in enumerate(METRICS):
            scale,digits=(1,3) if m=='host_pore' else (100,2)
            expected.append((f'{means.loc[method,m]*scale:.{digits}f}',f'{sd.loc[method,m]*scale:.{digits}f}'))
            best=means[m].max() if m in ['acc','host_reject'] else means[m].min()
            assert (r'\mathbf' in line.split('&')[col+2])==np.isclose(means.loc[method,m],best,atol=1e-12,rtol=0)
        assert pm(line)==expected,(family,method,line,expected)
report['table3']={'methods':stopping_methods,'rows_verified':len(actual),'metric_cells_verified':len(actual)*len(METRICS)}
forced_source=(ROOT/'tables/supp_forced_decision.tex').read_text()
assert r'Forced decision (\%)' in forced_source and r'\mathbf' not in forced_source
forced_rows=table_lines('supp_forced_decision.tex')
assert len(forced_rows)==len(stopping_families)
for family,line in zip(stopping_families,forced_rows):
    assert line.split('&')[0].strip()==family
    expected=[]
    for method in stopping_methods:
        timing=decision_stats.loc[(family,method)]
        expected.append((f'{timing.at_budget_mean*100:.2f}',f'{timing.at_budget_sd*100:.2f}'))
    assert pm(line)==expected,(family,line,expected)
report['forced_decision_table']={'number':'S2','rows_verified':len(forced_rows),'cells_verified':len(forced_rows)*len(stopping_methods),'denominator':'All test reads'}
# Check the paired host-time direction and the all-read budget fractions used in Section 3.3.
paired=stopping[stopping.method=='facp'].merge(stopping[stopping.method.isin(['d0.01','d0.05'])],on=['family','dataset','rep'],suffixes=('_facp','_threshold'),validate='one_to_many')
assert len(paired)==24
assert (paired.host_pore_threshold>paired.host_pore_facp).all()
np.testing.assert_allclose(decisions[['early_mean','at_budget_mean','no_decision_mean']].sum(axis=1),1,atol=1e-12,rtol=0)
report['table3']['longer_host_time_paired_units_per_new_threshold']=12
# Fixed-probability prefix data are available only at the supplied 0.01 percentage-point precision.
prefix=read('fig2a_p05_summary')
assert len(prefix)==128 and prefix.groupby('family').size().eq(32).all()
assert prefix.n.eq(3).all()
for r in prefix.itertuples(index=False):
    assert f'{r.acc_p05_mean*100:.2f} ± {r.acc_p05_sd*100:.2f}'==numbers['fig2a_full'][r.family][str(r.length)]
report['fig2a_p05_points_verified']=128
report['fig2a_precision_note']='Source CSV lacks acc_p05; use the DOCX/companion JSON means and SDs at 0.01 percentage-point precision.'
lso=read('lso_rows');lso=lso[lso.version.isin(['new','base'])]
for prefix in ['taxon','host','bact']:
    np.testing.assert_allclose(lso[prefix+'_fold']-lso[prefix+'_full'],lso['d_'+prefix],atol=1e-12)
report['paired_lso_rows_verified']=len(lso)
for r in read('figS3_composition').itertuples(index=False):
    if r.partition in ['train','dev_a (new)','test']:
        assert f'{r.pct:.2f}'==numbers['figS_composition'][r.fold][r.partition][int(r.group[-1])-1]

# Tables and figure numbering are locked to the current manuscript snapshot.
manifest=json.loads((ROOT/'provenance/publication_manifest.json').read_text())
expected_tables={
    'Table 1':'main_overall', 'Table 2':'closed_loop', 'Table 3':'supp_stopping',
    'Table S1':'supp_partitions_summary', 'Table S2':'supp_forced_decision',
    'Table S3':'supp_baseline_facp', 'Table S4':'supp_budget_summary',
    'Table S5':'supp_simulation_summary', 'Table S6':'supp_lso_summary',
    'Table S7':'supp_all_bacterial_rejection',
}
expected_figures={
    'Figure 1':'fig01_overview', 'Figure 2':'fig02_overall', 'Figure 3':'fig05_budget',
    'Figure S1':'figS_encoding', 'Figure S2':'fig03_encoder_cache',
    'Figure S3':'figS_taxon_holdouts', 'Figure S4':'figS_beta',
}
assert len(manifest['tables'])==len(expected_tables)
assert {r['label']:r['file'] for r in manifest['tables']}=={
    label:'tables/'+name+'.tex' for label,name in expected_tables.items()}
assert {p.stem for p in (ROOT/'tables').glob('*.tex')}==set(expected_tables.values()), 'Obsolete or missing active tables'
for record in manifest['tables']:
    path=ROOT/record['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'], record['label']
assert len(manifest['figures'])==len(expected_figures)
assert {r['label']:r['file'] for r in manifest['figures']}=={
    label:'figures/'+name+'.pdf' for label,name in expected_figures.items()}
# PDF metadata can change on redraw. Presence and readable formats are checked
# here; the plotting code separately verifies numerical summaries and layout.
for record in manifest['figures']:
    for suffix in ['.pdf','.png','.svg']:
        path=(ROOT/record['file']).with_suffix(suffix)
        assert path.is_file() and path.stat().st_size>1000, str(path)
        content=path.read_bytes()
        if suffix=='.pdf': assert content.startswith(b'%PDF-'), str(path)
        elif suffix=='.png': assert content.startswith(b'\x89PNG\r\n\x1a\n'), str(path)
        else: assert b'<svg' in content[:2000], str(path)
for suffix in ['.pdf','.png','.svg']:
    assert {p.stem for p in (ROOT/'figures').glob('*'+suffix)}==set(expected_figures.values()), 'Obsolete or missing active '+suffix+' figures'
macros=dict(re.findall(r'^\\newcommand\{\\(\w+)\}\{(.*)\}$',(ROOT/'numbers.tex').read_text(),re.M))
assert macros==manifest['numeric_macros'], 'Numeric macros differ from manuscript'
# Independently verify the three timing macros from paired raw repeats.
timing=read('timing_raw')
timing=timing[(timing.scenario=='mixed')&(timing.chunk==1600)&(timing.batch==512)]
paired=timing.pivot(index='rep',columns='method',values='seconds')
assert len(paired)==3 and paired.notna().all().all()
for name,values,digits in [
    ('TimingMixedInc',paired.incremental*1000,1),
    ('TimingMixedFull',paired.scratch*1000,1),
    ('TimingMixedRatio',paired.scratch/paired.incremental,2),
]:
    expected=r'\ensuremath{'+f'{values.mean():.{digits}f} \\pm {values.std(ddof=1):.{digits}f}'+'}'
    assert macros[name]==expected, (name,macros[name],expected)
counts=read('tableS1_counts')
assert len(counts)==12
for family,count in [('Simulated',10000),('R9',10000),('R10 mixed',10000),('R10 sputum',20000)]:
    assert counts.loc[counts.family==family,'dev_a'].eq(count).all(), family
assert len(counts[counts.family=='R9'].drop_duplicates(['dataset','train','dev_a','test']))==1
report['tables']={'main':3,'supplementary':7,'match_manuscript_hashes':len(manifest['tables'])}
report['figures']={'main':3,'supplementary':4,'formats':['pdf','png','svg']}
report['numeric_macros_verified']=len(macros)
report['partition_rows_in_table']=10
report['sample_sd_ddof']=1
report['status']='passed'
(ROOT/'review').mkdir(exist_ok=True)
(ROOT/'review/publication_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
