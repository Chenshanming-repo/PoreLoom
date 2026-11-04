#!/usr/bin/env python3
"""Aggregate archived experimental units as mean ± sample SD (n=3), without rerunning experiments."""
from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/poreloom-recomb-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from matplotlib.collections import PathCollection
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
FAMS=['Simulated','R9','R10 mixed','R10 sputum']
XL=['Sim.','R9','R10\nmixed','R10\nsputum']
MS=['poreloom','readcurrent','nanodeep_upstream','squigglenet']
NAMES=['PoreLoom','ReadCurrent','NanoDeep','SquiggleNet']
COL=dict(zip(MS,['#007E79','#4477AA','#D89A28','#A463A4']))
RENDER=[]
BAR_SUMMARIES=[]
BAR_CHECKS=[]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'lines.linewidth':1.2,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'poreloom-recomb2027','savefig.facecolor':'white'})
def data(n):return pd.read_csv(ROOT/'figure_data'/f'{n}.csv')
def panel(ax,c):
    ax.text(0,1.04,c,transform=ax.transAxes,fontweight='bold',fontsize=12,va='bottom')
    ax.grid(axis='y',color='#E5E9EA',linewidth=.55,zorder=0)
    ax.set_axisbelow(True)
def save(fig,name):
    fig.canvas.draw()
    for a in fig.axes:
        for axis in (a.xaxis, a.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks():
                if not lo <= tick.get_loc() <= hi:
                    tick.label1.set_visible(False); tick.label2.set_visible(False)
    fig.canvas.draw()
    for a in fig.axes:assert not a.get_title(),name
    bounds=[];renderer=fig.canvas.get_renderer();width,height=fig.canvas.get_width_height()
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text():continue
        bb=t.get_window_extent(renderer)
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1>width+1 or bb.y1>height+1:
            # Hidden colorbar tick labels are not publication labels.
            if t.get_text() in ['−0.5','0.0'] and t not in fig.texts:continue
            bounds.append(t.get_text())
    for extension in ['pdf', 'svg', 'png']:
        fig.savefig(OUT/f'{name}.{extension}', dpi=300)
    assert not bounds, (name, bounds)
    RENDER.append(dict(figure=name,aggregation='equal-weight mean ± sample SD, ddof=1, n=3',subplot_titles=[a.get_title() for a in fig.axes],out_of_canvas_text=bounds))
    plt.close(fig)
def legend(fig,keys,names,colors,y=.99,ncol=4):
    fig.legend([Line2D([],[],marker='o',lw=1,color=colors[k],ms=4) for k in keys],names,loc='upper center',bbox_to_anchor=(.5,y),ncol=ncol,frameon=False,columnspacing=1.2,handlelength=1.2)
def groups(ax,d,key,scale=1,methods=MS,colors=COL,methodcol='method'):
    for mi,m in enumerate(methods):
        for fi,f in enumerate(FAMS):
            v=d[(d[methodcol]==m)&(d.family==f)][key].to_numpy()*scale;x=fi+(mi-(len(methods)-1)/2)*.17
            assert len(v) == 3 and np.isfinite(v).all()
            ax.errorbar(x,np.mean(v),yerr=np.std(v,ddof=1),fmt='o',ms=3.2,color=colors[m],capsize=2,elinewidth=.9,capthick=.8,zorder=4)
    ax.set_xticks(range(4),XL);ax.set_xlim(-.55,3.55)
def main_figure():
    # Fixed-window outcomes and enrichment are in main Tables 1 and 2.
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.75))
    prefix, positions = data('fixed_prefix'), data('decision_position_cdf')
    prefix_lengths = [432, 768, 1200, 1968, 3000, 4464, 6000]
    prefix = prefix[prefix.usable_samples.isin(prefix_lengths)]
    family_colors = {'Simulated': '#4477AA', 'R9': '#A463A4', 'R10 mixed': '#007E79', 'R10 sputum': '#C76A44'}
    for a, letter, frame, metric in [(axes[0], 'a', prefix, 'acc'),
                                    (axes[1], 'b', positions[positions.category == 'all'], 'cdf')]:
        panel(a, letter)
        for family in FAMS:
            family_data = frame[frame.family == family]
            if metric == 'cdf':
                # Align exact empirical steps before averaging the three units.
                unit_cdfs = family_data.pivot(index='usable_samples', columns='unit', values='cdf').sort_index().ffill().fillna(0)
                assert unit_cdfs.shape[1] == 3 and unit_cdfs.notna().all().all()
                q = unit_cdfs.agg(['mean', 'std'], axis=1)
            else:
                assert family_data.groupby('usable_samples').unit.nunique().eq(3).all()
                q = family_data.groupby('usable_samples')[metric].agg(['mean', 'std'])
            x, mean, sd = q.index.to_numpy(), q['mean'].to_numpy()*100, q['std'].to_numpy()*100
            low, high = mean-sd, mean+sd
            a.plot(x, mean, color=family_colors[family], lw=1.5,
                   marker='o' if metric == 'acc' else None, markersize=3.2, markeredgewidth=0,
                   drawstyle='steps-post' if metric == 'cdf' else 'default')
            a.fill_between(x, low, high, color=family_colors[family], alpha=.12,
                           step='post' if metric == 'cdf' else None, linewidth=0)
        a.set(xlim=(0, 6300), xticks=[0, 3000, 6000])
    axes[0].set(ylabel='Accuracy (%)', xlabel='Prefix length (usable samples)', ylim=(55, 100), yticks=[60, 80, 100])
    axes[1].set(ylabel='Decided reads (%)', xlabel='Usable samples at decision', ylim=(0, 102), yticks=[0, 50, 100])
    axes[1].axvline(3000, color='#92999D', lw=.7, ls='--', zorder=0)
    fig.legend([Line2D([], [], color=family_colors[f], lw=1.5) for f in FAMS],
               ['Simulated', 'R9', 'R10 mixed', 'R10 sputum'], loc='upper center',
               bbox_to_anchor=(.5, 1), ncol=4, frameon=False, columnspacing=1.0, handlelength=1.1)
    fig.subplots_adjust(left=.105, right=.985, bottom=.2, top=.8, wspace=.37)
    save(fig, 'fig02_overall')


def encoder_timing():
    d=data('encoder');t=data('timing_raw')
    assert t.groupby(['scenario','chunk','batch','method']).rep.nunique().eq(3).all()
    t=t[t.scenario=='aligned'].groupby(['scenario','chunk','batch','method','n']).seconds.agg(['mean','std','count']).reset_index()
    assert t['count'].eq(3).all()
    t[['mean','std']] *= 1000
    fig,ax=plt.subplots(1,3,figsize=(6.5,2.65),gridspec_kw={'width_ratios':[1.25,1.25,1]})
    colors={'main':COL['poreloom'],'window_cp':'#808A93'}
    for a,c,key,scale,lab in zip(ax[:2],'ab',['acc','host_pore'],[100,1],['Accuracy (%)','Pore time per\nhost read (s)']):
        panel(a,c)
        groups(a,d,key,scale,['main','window_cp'],colors,'variant')
        a.set_ylabel(lab,labelpad=1)
        a.set_xticks(range(4),XL)
        a.set_xlim(-.35,3.35)
        a.tick_params(axis='x',labelsize=8,pad=4)
        plt.setp(a.get_xticklabels(),rotation=0,ha='center',va='top',multialignment='center')
    ax[0].set_ylim(85,98);ax[1].set_ylim(bottom=0)
    a=ax[2];panel(a,'c')
    for m,lab,col in [('incremental','Feature reuse',COL['poreloom']),('scratch','Full recomputation','#808A93')]:
        q=t[(t.scenario=='aligned')&(t.chunk==1600)&(t.batch==512)&(t.method==m)&(t.n<=19200)].sort_values('n')
        a.plot(q.n/1000,q['mean'],'o-',ms=3,color=col,label=lab)
        a.fill_between(q.n/1000,q['mean']-q['std'],q['mean']+q['std'],color=col,alpha=.13)
    a.set(xlabel='Prefix length\n(thousand samples)',ylabel='Encoder time (ms)',ylim=(0,650));a.legend(frameon=False,loc='upper left',bbox_to_anchor=(0,1.04),fontsize=7,handlelength=1.0)
    a.yaxis.labelpad=1
    legend(fig,['main','window_cp'],['Streaming-prefix encoder','Sliding-window encoder'],colors,ncol=2)
    fig.subplots_adjust(left=.09,right=.97,bottom=.28,top=.76,wspace=.35)
    save(fig,'fig03_encoder_cache')
def budget_figure():
    d=data('budget');fig,ax=plt.subplots(1,3,figsize=(6.5,2.5));cc={'r10_run01':'#4477AA','sputum_g1':'#C76A44'}
    for a,c,key,sc,lab in zip(ax,'abc',['acc','frr','host_pore'],[100,100,1],['Accuracy (%)','Targets rejected (%)','Pore time per\nhost read (s)']):
        panel(a,c)
        for ds in cc:
            q=d[d.dataset==ds]
            assert q.groupby('budget').replicate.nunique().eq(3).all()
            z=q.groupby('budget')[key].agg(['mean','std'])
            a.plot(z.index/1000,z['mean']*sc,color=cc[ds],lw=1.8,marker='o',ms=3)
            a.fill_between(z.index/1000,(z['mean']-z['std'])*sc,(z['mean']+z['std'])*sc,color=cc[ds],alpha=.13,linewidth=0)
        a.set_xlabel('Maximum budget\n(thousand samples)');a.set_ylabel(lab);a.set_xticks([6,9,12,15,18]);a.tick_params(axis='x',labelsize=10)
    legend(fig,list(cc),['R10 mixed, group 1','R10 sputum, group 1'],cc,ncol=2)
    fig.subplots_adjust(left=.10,right=.988,bottom=.29,top=.78,wspace=.56);save(fig,'fig05_budget')
def generalization():
    l=data('lso');fig,ax=plt.subplots(1,2,figsize=(6.8,2.9))
    folds=['Prevotella','Streptococcus','Fusobacterium','Actinomycetota'];labels=['Prev.','Strep.','Fuso.','4 genera']
    a=ax[0];panel(a,'a')
    for fi,f in enumerate(folds):
        for mi,m in enumerate(MS):
            q=l[(l.fold==f)&(l.method==m)];x=fi+(mi-1.5)*.18
            assert len(q) == 3 and q.group.nunique() == 3
            vals=[100*q.frr_main.mean(),100*q.frr_fold.mean()];a.plot([x-.035,x+.035],vals,color=COL[m],lw=.7)
            a.errorbar(x-.035,vals[0],yerr=100*q.frr_main.std(ddof=1),fmt='o',mfc='white',mec=COL[m],color=COL[m],ms=3.5,capsize=1.2,elinewidth=.7,capthick=.6,zorder=3)
            a.errorbar(x+.035,vals[1],yerr=100*q.frr_fold.std(ddof=1),fmt='o',color=COL[m],ms=3.5,capsize=1.2,elinewidth=.7,capthick=.6,zorder=3)
    a.set(xticks=range(4),xticklabels=labels,ylabel='Held-out targets rejected (%)',ylim=(0,102));a.tick_params(axis='x',labelsize=10)
    a=ax[1];panel(a,'b')
    for fi,f in enumerate(folds):
        for mi,m in enumerate(MS):
            q=l[(l.fold==f)&(l.method==m)];x=fi+(mi-1.5)*.18
            assert len(q) == 3 and np.allclose(q.dhost,q.host_fold-q.host_main)
            a.errorbar(x,q.dhost.mean()*100,yerr=q.dhost.std(ddof=1)*100,fmt='o',color=COL[m],ms=3.5,capsize=1.7,elinewidth=.8,capthick=.7)
    a.axhline(0,color='#888888',lw=.7);a.set(xticks=range(4),xticklabels=labels,ylabel='ΔHosts rejected (pp)');a.tick_params(axis='x',labelsize=10)
    legend(fig,MS,NAMES,COL)
    fig.text(.5,.815,'Open: full training    Filled: taxon holdout',ha='center',fontsize=8)
    fig.subplots_adjust(left=.105,right=.985,bottom=.22,top=.72,wspace=.42);save(fig,'fig06_generalization')



def main():
    (ROOT/'review').mkdir(exist_ok=True)
    from draw_overview import draw
    from plot_beta import render as render_beta, saved as saved_beta
    from analyze_matched_rejection import render as render_matched, saved as saved_matched
    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        draw(OUT)
    main_figure();encoder_timing();budget_figure();generalization()
    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        render_beta(*saved_beta())
    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        render_matched(*saved_matched())
    t=data('timing_raw')
    mixed=t[(t.scenario=='mixed')&(t.chunk==1600)&(t.batch==512)]
    assert len(mixed)==6
    timing={method:{'mean_ms':float(1000*q.seconds.mean()),'sd_ms':float(1000*q.seconds.std(ddof=1)),'n':len(q)} for method,q in mixed.groupby('method')}
    paired=mixed.pivot(index='rep',columns='method',values='seconds')
    ratios=paired.scratch/paired.incremental
    timing['paired_speed_ratio']={'mean':float(ratios.mean()),'sd':float(ratios.std(ddof=1)),'n':len(ratios)}
    timing['ratio_of_means']=timing['scratch']['mean_ms']/timing['incremental']['mean_ms']
    report={'figures':RENDER,'mixed_prefix_timing':timing,'source_directory':'figure_data','sample_sd_ddof':1,'experimental_units_per_condition':3,'individual_repeat_traces':False}
    (ROOT/'review/figure_mean_sd_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
