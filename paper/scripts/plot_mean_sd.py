#!/usr/bin/env python3
"""Render the revised manuscript from frozen source data; mean ± sample SD (n=3)."""
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
(ROOT/'review').mkdir(exist_ok=True)
FAMS=['Simulated','R9','R10 mixed','R10 sputum']
XL=['Sim.','R9','R10\nmixed','R10\nsputum']
MS=['poreloom','readcurrent','nanodeep_upstream','squigglenet']
NAMES=['PoreLoom','ReadCurrent','NanoDeep','SquiggleNet']
COL=dict(zip(MS,['#007E79','#4477AA','#D89A28','#A463A4']))
RENDER=[]
BAR_SUMMARIES=[]
BAR_CHECKS=[]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'lines.linewidth':1.2,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'poreloom-recomb2027','savefig.facecolor':'white'})
LEGACY_OFFLINE = ROOT / 'data/revision_20261104'
OFFLINE = ROOT / 'data/revision_20261109_tnorm15/offline'

def data(n):
    source = LEGACY_OFFLINE if n == 'timing_raw' else OFFLINE
    return pd.read_csv(source/f'{n}.csv')
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
    fig.savefig(OUT/f'{name}.pdf')
    fig.savefig(OUT/f'{name}.svg')
    fig.savefig(OUT/f'{name}.png',dpi=200)
    fig.savefig(ROOT/'review'/f'{name}_mean_sd.png',dpi=200)
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

def shape_preserving_curve(x, y, query):
    """PCHIP interpolation through each supplied value, without overshoot.

    Interior derivatives use the weighted harmonic mean of adjacent secants;
    one-sided endpoint derivatives are limited to preserve local shape.
    This is display interpolation, not statistical smoothing or refitting.
    """
    x, y, query = (np.asarray(v, dtype=float) for v in (x, y, query))
    h = np.diff(x)
    assert len(x) >= 3 and np.all(h > 0)
    assert query.min() >= x[0] and query.max() <= x[-1]
    secant = np.diff(y) / h
    slopes = np.zeros_like(y)
    for i in range(1, len(x)-1):
        left, right = secant[i-1], secant[i]
        if left * right > 0:
            w1, w2 = 2*h[i]+h[i-1], h[i]+2*h[i-1]
            slopes[i] = (w1+w2)/(w1/left+w2/right)
    for i, h0, h1, d0, d1 in [(0,h[0],h[1],secant[0],secant[1]),
                              (-1,h[-1],h[-2],secant[-1],secant[-2])]:
        slope = ((2*h0+h1)*d0-h0*d1)/(h0+h1)
        if np.sign(slope) != np.sign(d0):
            slope = 0.
        elif np.sign(d0) != np.sign(d1) and abs(slope) > 3*abs(d0):
            slope = 3*d0
        slopes[i] = slope
    i = np.clip(np.searchsorted(x, query, side='right')-1, 0, len(x)-2)
    t = (query-x[i])/h[i]
    return ((2*t**3-3*t**2+1)*y[i] + (t**3-2*t**2+t)*h[i]*slopes[i]
            + (-2*t**3+3*t**2)*y[i+1] + (t**3-t**2)*h[i]*slopes[i+1])

def main_figure():
    # Fixed-window outcomes and enrichment are reported in Tables 1 and 2.
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.75))
    prefix = pd.read_csv(LEGACY_OFFLINE / 'fig2a_p05_summary.csv').rename(columns={'length': 'usable_samples', 'acc_p05_mean': 'mean', 'acc_p05_sd': 'std'})
    positions = pd.read_csv(OFFLINE / 'fig2b_cdf_units.csv').rename(columns={'used': 'usable_samples', 'frac': 'cdf'})
    decision_summary = pd.read_csv(OFFLINE / 'fig2b_decisions_summary.csv').set_index('family')
    positions['unit'] = positions.dataset + '/r' + positions.rep.astype(str)
    # All 32 requested lengths, fixed p=0.5; supplied summary precision is documented.
    assert prefix.groupby('family').size().eq(32).all()
    family_colors = {'Simulated': '#4477AA', 'R9': '#A463A4', 'R10 mixed': '#007E79', 'R10 sputum': '#C76A44'}
    for a, letter, frame, metric in [(axes[0], 'a', prefix, 'acc'),
                                    (axes[1], 'b', positions, 'cdf')]:
        panel(a, letter)
        for family in FAMS:
            family_data = frame[frame.family == family]
            if metric == 'cdf':
                # Align exact empirical steps before averaging the three units.
                unit_cdfs = family_data.pivot(index='usable_samples', columns='unit', values='cdf').sort_index().ffill().fillna(0)
                assert unit_cdfs.shape[1] == 3 and unit_cdfs.notna().all().all()
                q = unit_cdfs.agg(['mean', 'std'], axis=1)
            else:
                assert family_data.n.eq(3).all()
                q = family_data.set_index('usable_samples')[['mean', 'std']].sort_index()
            x, mean, sd = q.index.to_numpy(), q['mean'].to_numpy()*100, q['std'].to_numpy()*100
            if metric == 'cdf':
                # Interpolate up to the exact left limit at the budget, then
                # retain the probability mass at 6000 as a vertical jump.
                # For each unit F(6000-) = 1 - fraction decided at budget.
                mean[-1] = 100*(1-decision_summary.loc[family,'at_budget_mean'])
                sd[-1] = 100*decision_summary.loc[family,'at_budget_sd']
            smooth_x = np.unique(np.r_[np.linspace(x[0],x[-1],1200),x])
            smooth_mean = shape_preserving_curve(x,mean,smooth_x)
            smooth_sd = shape_preserving_curve(x,sd,smooth_x)
            # The display curve passes through every reported summary value.
            np.testing.assert_allclose(shape_preserving_curve(x,mean,x),mean,atol=1e-10)
            assert np.all(np.diff(smooth_mean) >= -1e-10)
            assert np.all(smooth_sd >= -1e-10)
            if metric == 'cdf':
                smooth_x = np.r_[smooth_x,6000.]
                smooth_mean = np.r_[smooth_mean,100.]
                smooth_sd = np.r_[smooth_sd,0.]
            a.plot(smooth_x,smooth_mean,color=family_colors[family],lw=1.5)
            a.fill_between(smooth_x,np.maximum(0,smooth_mean-smooth_sd),
                           np.minimum(100,smooth_mean+smooth_sd),
                           color=family_colors[family],alpha=.12,linewidth=0)
        a.set(xlim=(0, 6300), xticks=[0, 3000, 6000])
    axes[0].set(ylabel='Accuracy (%)', xlabel='Prefix length (usable samples)', ylim=(55, 100), yticks=[60, 80, 100])
    axes[1].set(ylabel='Decided reads (%)', xlabel='Usable samples at decision', ylim=(0, 102), yticks=[0, 50, 100])
    axes[1].axvline(3000, color='#92999D', lw=.7, ls='--', zorder=0)
    fig.legend([Line2D([], [], color=family_colors[f], lw=1.5) for f in FAMS],
               ['Simulated', 'R9', 'R10 mixed', 'R10 sputum'], loc='upper center',
               bbox_to_anchor=(.5, 1), ncol=4, frameon=False, columnspacing=1.0, handlelength=1.1)
    fig.subplots_adjust(left=.105, right=.985, bottom=.2, top=.8, wspace=.37)
    fig.savefig(OUT/'fig02_overall.png',dpi=200)
    save(fig, 'fig02_overall')
    RENDER[-1].update({'display_interpolation':'shape-preserving cubic Hermite (PCHIP), no markers',
                       'all_observed_summary_values_preserved':True,
                       'cdf_budget_jump_preserved':True})


def encoder_timing():
    t=data('timing_raw')
    assert t.groupby(['scenario','chunk','batch','method']).rep.nunique().eq(3).all()
    t=t[t.scenario=='aligned'].groupby(['scenario','chunk','batch','method','n']).seconds.agg(['mean','std','count']).reset_index()
    assert t['count'].eq(3).all()
    t[['mean','std']] *= 1000
    # Encoding outcomes appear in Fig. S1; retain only
    # the feature-reuse timing comparison in this single-panel figure.
    fig,a=plt.subplots(figsize=(3.6,2.2))
    a.grid(axis='y',color='#E5E9EA',linewidth=.55,zorder=0)
    a.set_axisbelow(True)
    for m,lab,col in [('incremental','Feature reuse',COL['poreloom']),('scratch','Full recomputation','#808A93')]:
        q=t[(t.scenario=='aligned')&(t.chunk==1600)&(t.batch==512)&(t.method==m)&(t.n<=19200)].sort_values('n')
        a.plot(q.n/1000,q['mean'],'o-',ms=3,color=col,label=lab)
        a.fill_between(q.n/1000,q['mean']-q['std'],q['mean']+q['std'],color=col,alpha=.13)
    a.set(xlabel='Prefix length\n(thousand samples)',ylabel='Encoder time (ms)',ylim=(0,650))
    a.legend(frameon=False,loc='upper left',fontsize=8,handlelength=1.3)
    a.yaxis.labelpad=3
    fig.subplots_adjust(left=.2,right=.97,bottom=.29,top=.96)
    save(fig,'fig03_encoder_cache')
    RENDER[-1].update({'panels':1,'metrics':['encoder_time_ms'],
                       'series':['Feature reuse','Full recomputation'],
                       'description':'Single-panel encoder timing comparison; former accuracy and host-pore-time panels removed.'})
def budget_figure():
    d=pd.read_csv(OFFLINE / 'budget_units.csv').rename(columns={'rep':'replicate'});fig,ax=plt.subplots(1,3,figsize=(6.5,2.5));cc={'r10_run01':'#4477AA','sputum_g1':'#C76A44'}
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



def main():
    main_figure();encoder_timing();budget_figure()
    t=data('timing_raw')
    mixed=t[(t.scenario=='mixed')&(t.chunk==1600)&(t.batch==512)]
    assert len(mixed)==6
    timing={method:{'mean_ms':float(1000*q.seconds.mean()),'sd_ms':float(1000*q.seconds.std(ddof=1)),'n':len(q)} for method,q in mixed.groupby('method')}
    paired=mixed.pivot(index='rep',columns='method',values='seconds')
    ratios=paired.scratch/paired.incremental
    timing['paired_speed_ratio']={'mean':float(ratios.mean()),'sd':float(ratios.std(ddof=1)),'n':len(ratios)}
    timing['ratio_of_means']=timing['scratch']['mean_ms']/timing['incremental']['mean_ms']
    report={'figures':RENDER,'mixed_prefix_timing':timing,
            'source_directories':{'timing':str(LEGACY_OFFLINE), 'fixed_threshold_prefix':str(LEGACY_OFFLINE), 'offline':str(OFFLINE)},
            'sample_sd_ddof':1,'experimental_units_per_condition':3,'individual_repeat_traces':False}
    (ROOT/'review/figure_mean_sd_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ['figures', 'mixed_prefix_timing']},indent=2))
if __name__=='__main__': main()
