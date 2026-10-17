#!/usr/bin/env python3
"""Publication plots from delivery-local CSVs. No subplot titles or experimental reruns."""
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
PC={'main':'#007E79','fixed_len':'#777777','const_thr':'#4477AA','calib_thr':'#C76A44'}
PN={'main':'FACP','fixed_len':'Fixed length','const_thr':'Constant threshold','calib_thr':'Length-calibrated'}
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
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=300)
    RENDER.append(dict(figure=name,subplot_titles=[a.get_title() for a in fig.axes],out_of_canvas_text=bounds))
    plt.close(fig)
def legend(fig,keys,names,colors,y=.99,ncol=4):
    fig.legend([Line2D([],[],marker='o',lw=1,color=colors[k],ms=4) for k in keys],names,loc='upper center',bbox_to_anchor=(.5,y),ncol=ncol,frameon=False,columnspacing=1.2,handlelength=1.2)
def groups(ax,d,key,scale=1,methods=MS,colors=COL,methodcol='method'):
    for mi,m in enumerate(methods):
        for fi,f in enumerate(FAMS):
            v=d[(d[methodcol]==m)&(d.family==f)][key].to_numpy()*scale;x=fi+(mi-(len(methods)-1)/2)*.17
            ax.scatter(x+np.linspace(-.025,.025,len(v)),v,s=11,color=colors[m],alpha=.48,zorder=3)
            ax.plot(x,np.mean(v),'_',ms=8,mew=1.7,color=colors[m],zorder=4)
    ax.set_xticks(range(4),XL);ax.set_xlim(-.55,3.55)
def bar_summary(values, figure, panel, group, method, variant, metric):
    v = np.asarray(values, dtype=float)
    assert len(v) == 3 and np.isfinite(v).all(), (figure, group, method)
    avg, low, high = v.mean(), v.min(), v.max()
    BAR_SUMMARIES.append(dict(figure=figure, panel=panel, group=group, method=method,
                          variant=variant, metric=metric, n=len(v),
                          mean=avg, minimum=low, maximum=high))
    return avg, np.array([[avg-low], [high-avg]])


def bar_method_legend(fig, methods, names, colors, y):
    fig.legend([Patch(facecolor=colors[m], edgecolor='none') for m in methods],
               names, loc='upper center', bbox_to_anchor=(.5, y),
               ncol=len(methods), frameon=False, columnspacing=1.05,
               handlelength=1.0, handletextpad=.5)


def save_bar_figure(fig, name, expected_bars, point_axes=()):
    assert not any(isinstance(x, PathCollection) for x in fig.findobj()), 'Scatter points found'
    marker_lines = [(a, x) for a in fig.axes for x in a.lines
                    if x.get_marker() not in (None, '', 'None', '_', '|')]
    assert all(a in point_axes for a, x in marker_lines), 'Point marker outside permitted curve panel'
    bar_count = sum(len(c.patches) for a in fig.axes for c in a.containers
                    if hasattr(c, 'patches'))
    assert bar_count == expected_bars, (name, bar_count)
    if name.startswith('fig04'):
        assert all(a.get_xlim()[0] <= 0 <= a.get_xlim()[1] for a in fig.axes)
    if name.startswith('fig02'):
        assert all(a.get_ylim()[0] == 0 for a in fig.axes if any(hasattr(c, 'patches') for c in a.containers))
    save(fig, name)
    assert not RENDER[-1]['out_of_canvas_text'], RENDER[-1]
    BAR_CHECKS.append(dict(figure=name, bars=bar_count, scatter_points=0,
                       line_marker_points=sum(len(x.get_xdata()) for a, x in marker_lines),
                       error_bars='minimum to maximum across three units; not confidence intervals',
                       zero_included_in_all_bar_axes=True,
                       no_clipped_text=True))


def main_figure():
    d, w, iy = data('main'), data('window_extensions'), data('icarust')
    fig, axes = plt.subplots(3, 2, figsize=(6.5, 6.4))
    methods = [('poreloom', 'streaming')] + [(m, v) for m in MS[1:] for v in ['w3000', 'w6000']]
    bar_width, tool_gap = .09, .055
    offsets, cursor = [], 0.0
    for mi, (method, _) in enumerate(methods):
        if mi and method != methods[mi-1][0]:
            cursor += tool_gap
        offsets.append(cursor+bar_width/2)
        cursor += bar_width
    offsets = np.array(offsets)-cursor/2
    for i in range(len(methods)-1):
        expected_gap = 0 if methods[i][0] == methods[i+1][0] else tool_gap
        assert np.isclose(offsets[i+1]-offsets[i]-bar_width, expected_gap)
    for a, letter, metric, scale, label in zip(axes.flat, 'abc',
            ['host_reject', 'frr', 'host_pore'], [100, 100, 1],
            ['Host rejection (%)', 'Target FRR (%)', 'Host pore occupancy (s)']):
        panel(a, letter)
        for fi, family in enumerate(FAMS):
            for mi, (method, variant) in enumerate(methods):
                q = d[(d.method == method) & (d.family == family)] if method == 'poreloom' else w[(w.method == method) & (w.family == family) & (w.variant == variant)]
                mean, err = bar_summary(q[metric] * scale, 'Figure 2', letter, family, method, variant, metric)
                a.bar(fi+offsets[mi], mean, width=bar_width,
                      facecolor='white' if variant == 'w3000' else COL[method],
                      edgecolor=COL[method], linewidth=.8, zorder=3,
                      yerr=err, error_kw=dict(ecolor='#4F565B', elinewidth=.55, capsize=1.2, capthick=.55))
        a.set(xticks=range(4), xticklabels=XL, xlim=(-.53, 3.53), ylabel=label)
        a.set_ylim(bottom=0)
        a.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axes[0, 0].set(ylim=(0, 105), yticks=[0, 25, 50, 75, 100])
    axes[0, 1].set_ylim(0, 23)
    axes[1, 0].set_ylim(0, 4.5)
    a = axes[1, 1]
    panel(a, 'd')
    for mi, method in enumerate(MS):
        strategy = 'PoreLoom' if method == 'poreloom' else 'nanodeep' if method == 'nanodeep_upstream' else method
        for fi, fraction in enumerate([10, 100, 300]):
            q = iy[(iy.strategy == strategy) & (iy.permille == fraction)]
            variant = 'streaming' if method == 'poreloom' else 'w3000'
            mean, err = bar_summary(q.relative_yield, 'Figure 2', 'd', f'{fraction/10:g}%', method, variant, 'relative_yield')
            a.bar(fi+(mi-1.5)*.18, mean, width=.15,
                  facecolor=COL[method] if method == 'poreloom' else 'white',
                  edgecolor=COL[method], linewidth=.85, zorder=3,
                  yerr=err, error_kw=dict(ecolor='#4F565B', elinewidth=.6, capsize=1.6, capthick=.6))
    a.axhline(1, color='#858C91', ls='--', lw=.8, zorder=4)
    a.set(xticks=range(3), xticklabels=['1', '10', '30'], xlabel='Input target fraction (%)',
          ylabel='Relative target yield', ylim=(0, 9))
    prefix, positions = data('fixed_prefix'), data('decision_position_cdf')
    prefix_lengths = [432, 768, 1200, 1968, 3000, 4464, 6000]
    prefix = prefix[prefix.usable_samples.isin(prefix_lengths)]
    family_colors = {'Simulated': '#4477AA', 'R9': '#A463A4', 'R10 mixed': '#007E79', 'R10 sputum': '#C76A44'}
    for a, letter, frame, metric in [(axes[2, 0], 'e', prefix, 'acc'),
                                    (axes[2, 1], 'f', positions[positions.category == 'all'], 'cdf')]:
        panel(a, letter)
        for family in FAMS:
            family_data = frame[frame.family == family]
            if metric == 'cdf':
                # Align exact empirical steps before averaging the three units.
                unit_cdfs = family_data.pivot(index='usable_samples', columns='unit', values='cdf').sort_index().ffill().fillna(0)
                q = unit_cdfs.agg(['mean', 'min', 'max'], axis=1)
            else:
                q = family_data.groupby('usable_samples')[metric].agg(['mean', 'min', 'max'])
            x, mean, low, high = q.index.to_numpy(), q['mean'].to_numpy()*100, q['min'].to_numpy()*100, q['max'].to_numpy()*100
            a.plot(x, mean, color=family_colors[family], lw=1.5,
                   marker='o' if metric == 'acc' else None, markersize=3.2, markeredgewidth=0,
                   drawstyle='steps-post' if metric == 'cdf' else 'default')
            a.fill_between(x, low, high, color=family_colors[family], alpha=.12,
                           step='post' if metric == 'cdf' else None, linewidth=0)
        a.set(xlim=(0, 6300), xticks=[0, 3000, 6000])
    axes[2, 0].set(ylabel='Accuracy (%)', xlabel='Signal observed', ylim=(55, 100), yticks=[60, 80, 100])
    axes[2, 1].set(ylabel='Cumulative\ndecisions (%)', xlabel='Decision position', ylim=(0, 102), yticks=[0, 50, 100])
    axes[2, 1].axvline(3000, color='#92999D', lw=.7, ls='--', zorder=0)
    fig.legend([Line2D([], [], color=family_colors[f], lw=1.5) for f in FAMS],
               ['Simulated', 'R9', 'R10 mixed', 'R10 sputum'], loc='upper center',
               bbox_to_anchor=(.5, .375), ncol=4, frameon=False, columnspacing=1.0, handlelength=1.1)
    bar_method_legend(fig, MS, NAMES, COL, 1)
    fig.legend([Patch(facecolor='white', edgecolor='#555555'), Patch(facecolor='#777777', edgecolor='#777777')],
               ['3000-sample window (a-d)', '6000-sample window (a-c)'],
               loc='upper center', bbox_to_anchor=(.5, .962), ncol=2,
               frameon=False, columnspacing=1.5, handlelength=1)
    fig.subplots_adjust(left=.105, right=.985, bottom=.135, top=.895, hspace=.72, wspace=.37)
    save_bar_figure(fig, 'fig02_overall', 96, point_axes=(axes[2, 0],))
    BAR_CHECKS[-1]['accuracy_curve_samples'] = prefix_lengths
    BAR_CHECKS[-1]['decision_cdf_support'] = 'Union of observed decision positions; unit CDFs carried forward before equal averaging'
    BAR_CHECKS[-1]['bar_grouping'] = dict(panels='a-c', window_pair_gap=0,
                                    tool_gap=tool_gap, bar_width=bar_width,
                                    dataset_group_gap=1-cursor)


def read_case_figure():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.45))
    cases = data('read_cases')
    action_y = {'Reject': -1, 'Continue': 0, 'Accept': 1}
    for a, letter, case in zip(axes, 'ab', ['Host', 'Target']):
        panel(a, letter)
        q = cases[(cases.case == case) & (cases.method == 'poreloom')]
        y = q.action.map(action_y)
        a.plot(q.time_seconds, y, color=COL['poreloom'], lw=1.3, zorder=3)
        a.plot(q.time_seconds.iloc[:-1], y.iloc[:-1], 'o', color=COL['poreloom'],
               mfc='white', ms=4.6, mew=1.1, zorder=4)
        a.plot(q.time_seconds.iloc[-1], y.iloc[-1], 'o', color=COL['poreloom'], ms=5.2, zorder=4)
        for m, marker, offset in zip(MS[1:], ['s', 'D', '^'], [-.28, 0, .28]):
            b = cases[(cases.case == case) & (cases.method == m)].iloc[0]
            a.plot(b.time_seconds, action_y[b.action]+offset, marker=marker,
                   color=COL[m], ms=4.4, mew=.5, zorder=4)
        a.set(xlim=(.4, 2.22), xticks=[.5, 1, 1.5, 2], ylim=(-1.42, 1.42),
              yticks=[-1, 0, 1], yticklabels=['Reject', 'Continue', 'Accept'],
              xlabel='Time since read start (s)', ylabel=f'{case} read')
        a.tick_params(axis='y', labelsize=9)
    legend(fig, MS, NAMES, COL)
    fig.subplots_adjust(left=.135, right=.985, bottom=.24, top=.76, wspace=.49)
    save(fig, 'figS04_read_cases')


def encoder_timing():
    d=data('encoder');t=data('timing_medians');r=data('timing_ratios')
    fig,ax=plt.subplots(2,2,figsize=(6.5,4.8))
    colors={'main':COL['poreloom'],'window_cp':'#808A93'}
    for a,c,key,scale,lab in zip(ax[0],'ab',['acc','host_pore'],[100,1],['ACC (%)','Host pore occupancy (s)']):
        panel(a,c);groups(a,d,key,scale,['main','window_cp'],colors,'variant');a.set_ylabel(lab)
    ax[0,0].set_ylim(85,98);ax[0,1].set_ylim(bottom=0)
    a=ax[1,0];panel(a,'c')
    for m,lab,col in [('incremental','Cached',COL['poreloom']),('scratch','Full prefix','#808A93')]:
        q=t[(t.scenario=='aligned')&(t.chunk==1600)&(t.batch==512)&(t.method==m)&(t.n<=19200)].sort_values('n')
        a.plot(q.n/1000,q.median_ms,'o-',ms=3,color=col,label=lab)
        a.fill_between(q.n/1000,q.min_ms,q.max_ms,color=col,alpha=.13)
    a.set(xlabel='Received samples\n(thousands)',ylabel='Update time (ms)',ylim=(0,650));a.legend(frameon=False,loc='upper left')
    a=ax[1,1];panel(a,'d');a.grid(False)
    q=r[r.scenario=='mixed'].pivot(index='chunk',columns='batch',values='speed_ratio');vals=q.to_numpy()
    im=a.imshow(vals,cmap='RdBu',norm=TwoSlopeNorm(vmin=0,vcenter=1,vmax=8),aspect='auto')
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):a.text(j,i,f'{vals[i,j]:.1f}',ha='center',va='center',fontsize=10,color='white' if vals[i,j]>5 else '#222222')
    a.set(xticks=range(5),xticklabels=q.columns.astype(str),yticks=range(6),yticklabels=[f'{v/4000:g}' for v in q.index],xlabel='Batch size (reads)',ylabel='Chunk duration (s)')
    cb=fig.colorbar(im,ax=a,fraction=.045,pad=.03);cb.set_label('Full / cached',fontsize=10);cb.ax.tick_params(labelsize=10)
    legend(fig,['main','window_cp'],['Streaming prefix + FACP','Sliding window + FACP'],colors,ncol=2)
    fig.subplots_adjust(left=.105,right=.90,bottom=.17,top=.87,hspace=.65,wspace=.43);save(fig,'fig03_encoder_cache')
def decision_figure():
    d, native = data('stopping'), data('native_cp')
    paired = native.pivot(index=['unit', 'family', 'method'], columns='state', values=['acc', 'host_reject', 'frr', 'host_pore'])
    delta = pd.DataFrame({m: paired[m]['C-p']-paired[m]['Native'] for m in ['acc', 'host_reject', 'frr', 'host_pore']}).reset_index()
    delta.to_csv(ROOT/'figure_data/native_cp_deltas.csv', index=False, float_format='%.15g')
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.95))
    for row, frame, methods, colors, keycol, letters in [
            (0, d, list(PC), PC, 'variant', 'abc'),
            (1, delta, MS[1:], COL, 'method', 'def')]:
        labels = ['Host rejection (%)', 'Target FRR (%)', 'Host occupancy (s)'] if row == 0 else ['ΔHost rejection (pp)', 'ΔTarget FRR (pp)', 'ΔHost occupancy (s)']
        for col, (metric, scale, label, letter) in enumerate(zip(['host_reject', 'frr', 'host_pore'], [100, 100, 1], labels, letters)):
            a = axes[row, col]
            panel(a, letter)
            a.grid(False)
            a.grid(axis='x', color='#E5E9EA', linewidth=.55, zorder=0)
            for fi, family in enumerate(FAMS):
                if fi % 2 == 0:
                    a.axhspan(fi-.46, fi+.46, color='#F5F7F8', zorder=0)
                for mi, method in enumerate(methods):
                    q = frame[(frame[keycol] == method) & (frame.family == family)]
                    mean, err = bar_summary(q[metric] * scale, 'Figure 4', letter, family, method,
                                        'same_encoder' if row == 0 else 'Cp_minus_native_W3000', metric)
                    y = fi+(mi-(len(methods)-1)/2)*.21
                    a.barh(y, mean, height=.15, color=colors[method], edgecolor='none', zorder=3,
                           xerr=err, error_kw=dict(ecolor='#3F484E', elinewidth=.65, capsize=1.2, capthick=.6))
            a.set(yticks=range(4), yticklabels=['Sim.', 'R9', 'R10 mixed', 'R10 sputum'] if col == 0 else [],
                  ylim=(3.55, -.55), xlabel=label)
            a.tick_params(axis='y', length=0, pad=7)
            a.spines['left'].set_visible(False)
            a.xaxis.set_major_locator(MaxNLocator(nbins=3))
            if row == 0:
                a.set_xlim(left=0)
            else:
                a.axvline(0, color='#8D969C', lw=.8, zorder=4)
    axes[0, 0].set(xlim=(0, 103), xticks=[0, 50, 100])
    axes[0, 1].set(xlim=(0, 11.2), xticks=[0, 5, 10])
    axes[0, 2].set(xlim=(0, 2.9), xticks=[0, 1, 2])
    bar_method_legend(fig, list(PC), [PN[k] for k in PC], PC, 1)
    bar_method_legend(fig, MS[1:], NAMES[1:], COL, .525)
    fig.text(.5, .015, 'Bars: mean; whiskers: min-max of 3 units (not confidence intervals)',
             ha='center', va='bottom', fontsize=8, color='#586168')
    fig.subplots_adjust(left=.17, right=.985, bottom=.145, top=.875, hspace=.95, wspace=.23)
    save_bar_figure(fig, 'fig04_decisions', 84)


def budget_figure():
    d=data('budget');fig,ax=plt.subplots(1,3,figsize=(6.5,2.5));cc={'r10_run01':'#4477AA','sputum_g1':'#C76A44'}
    for a,c,key,sc,lab in zip(ax,'abc',['acc','frr','host_pore'],[100,100,1],['ACC (%)','Target FRR (%)','Host occupancy (s)']):
        panel(a,c)
        for ds in cc:
            q=d[d.dataset==ds]
            for rep,m in zip(range(3),['o','s','^']):
                z=q[q.replicate==rep].sort_values('budget');a.plot(z.budget/1000,z[key]*sc,color=cc[ds],alpha=.33,marker=m,ms=3,lw=.6)
            z=q.groupby('budget')[key].mean();a.plot(z.index/1000,z*sc,color=cc[ds],lw=1.8)
        a.set_xlabel('Budget\n(thousand samples)');a.set_ylabel(lab);a.set_xticks([6,9,12,15,18]);a.tick_params(axis='x',labelsize=10)
    legend(fig,list(cc),['R10 mixed, group 1','R10 sputum, group 1'],cc,ncol=2)
    fig.subplots_adjust(left=.10,right=.988,bottom=.29,top=.78,wspace=.56);save(fig,'fig05_budget')
def generalization():
    l=data('lso');fig,ax=plt.subplots(1,2,figsize=(6.5,2.55))
    folds=['Prevotella','Streptococcus','Fusobacterium','Actinomycetota'];labels=['Prev.','Strep.','Fuso.','4 genera']
    a=ax[0];panel(a,'a')
    for fi,f in enumerate(folds):
        for mi,m in enumerate(MS):
            q=l[(l.fold==f)&(l.method==m)];x=fi+(mi-1.5)*.18
            vals=[100*q.frr_main.mean(),100*q.frr_fold.mean()];a.plot([x-.03,x+.03],vals,color=COL[m],lw=.8)
            a.scatter(x-.03,vals[0],facecolor='white',edgecolor=COL[m],s=20,zorder=3)
            a.scatter(x+.03,vals[1],color=COL[m],s=20,zorder=3)
    a.set(xticks=range(4),xticklabels=labels,ylabel='Held-out target FRR (%)',ylim=(0,102));a.tick_params(axis='x',labelsize=10)
    a=ax[1];panel(a,'b')
    for fi,f in enumerate(folds):
        for mi,m in enumerate(MS):
            q=l[(l.fold==f)&(l.method==m)];x=fi+(mi-1.5)*.18
            a.scatter(x+np.linspace(-.025,.025,3),q.dhost*100,color=COL[m],s=9,alpha=.4);a.plot(x,q.dhost.mean()*100,'_',ms=7,mew=1.6,color=COL[m])
    a.axhline(0,color='#888888',lw=.7);a.set(xticks=range(4),xticklabels=labels,ylabel='ΔHost rejection (pp)');a.tick_params(axis='x',labelsize=10)
    legend(fig,MS,NAMES,COL)
    fig.subplots_adjust(left=.105,right=.985,bottom=.22,top=.75,wspace=.35);save(fig,'fig06_generalization')

def transfer_figure():
    t=data('transfer');fig,ax=plt.subplots(1,2,figsize=(6.5,2.75))
    for a,c,key,lab in zip(ax,'ab',['target_frr','acc'],['Target FRR (%)','ACC (%)']):
        panel(a,c)
        for di,di_name in enumerate(['Sputum to mixed','Mixed to sputum']):
            for mi,m in enumerate(MS):
                q=t[(t.direction==di_name)&(t.method==m)];x=di+(mi-1.5)*.17
                v1=q['indomain_'+key].mean()*100;v2=q[key].mean()*100
                a.plot([x-.025,x+.025],[v1,v2],color=COL[m],lw=.8)
                a.scatter(x-.025,v1,facecolor='white',edgecolor=COL[m],s=23,zorder=3);a.scatter(x+.025,v2,color=COL[m],s=23,zorder=3)
                a.scatter(x+.025+np.linspace(-.02,.02,len(q)),q[key]*100,color=COL[m],s=7,alpha=.25)
        a.set(xticks=[0,1],xticklabels=['Sputum →\nmixed','Mixed →\nsputum'],ylabel=lab,ylim=(0,102),xlim=(-.5,1.5));a.tick_params(axis='x',labelsize=10)
    legend(fig,MS,NAMES,COL)
    fig.subplots_adjust(left=.105,right=.985,bottom=.28,top=.77,wspace=.35);save(fig,'figS03_transfer')
def supplement():
    s=data('budget_sensitivity');fig,ax=plt.subplots(1,2,figsize=(6.5,2.7))
    cs={.2:'#4477AA',.4:'#007E79',1.0:'#C76A44'}
    for a,c,ds in zip(ax,'ab',['r10_run01','sputum_g1']):
        panel(a,c)
        for dt in cs:
            q=s[(s.dataset==ds)&(s.interval==dt)&(s.service=='nominal')].groupby('budget').acc.mean()
            a.plot(q.index/1000,100*q,'o-',ms=3,color=cs[dt],label=f'{dt:g} s')
        q=s[(s.dataset==ds)&(s.interval==.4)&(s.service=='zero')].groupby('budget').acc.mean()
        a.plot(q.index/1000,100*q,'s--',ms=3,color='#777777',label='0.4 s; zero service')
        a.set(xlabel='Budget\n(thousand samples)',ylabel='Command ACC (%)')
    ax[0].legend(frameon=False,ncol=2,fontsize=10,loc='lower right');fig.subplots_adjust(left=.10,right=.985,bottom=.27,top=.85,wspace=.35);save(fig,'figS01_cadence')
    # Full timing grid at the final complete synchronous chunk, retaining slowdowns.
    r=data('timing_ratios');r=r[(r.scenario=='aligned')&(r.n%r.chunk==0)]
    r=r.sort_values('n').groupby(['chunk','batch']).tail(1)
    fig,ax=plt.subplots(1,2,figsize=(6.5,2.7))
    for a,c,kind in zip(ax,'ab',['incremental','speed_ratio']):
        panel(a,c);a.grid(False);q=r.pivot(index='chunk',columns='batch',values=kind);v=q.to_numpy()
        im=a.imshow(v,aspect='auto',cmap='YlGnBu' if kind=='incremental' else 'RdBu',norm=None if kind=='incremental' else TwoSlopeNorm(vmin=0,vcenter=1,vmax=8.1))
        for i in range(6):
            for j in range(5):a.text(j,i,f'{v[i,j]:.0f}' if kind=='incremental' else f'{v[i,j]:.1f}',ha='center',va='center',fontsize=10,color='white' if v[i,j]>(500 if kind=='incremental' else 5) else '#111111')
        a.set(xticks=range(5),xticklabels=q.columns.astype(str),yticks=range(6),yticklabels=[f'{i/4000:g}' for i in q.index],xlabel='Batch size (reads)',ylabel='Chunk duration (s)')
    fig.subplots_adjust(left=.10,right=.985,bottom=.27,top=.85,wspace=.35);save(fig,'figS02_timing_grid')
def write_bar_outputs():
    pd.DataFrame(BAR_SUMMARIES).to_csv(ROOT/'figure_data/bar_summaries.csv', index=False, float_format='%.15g')
    (ROOT/'review/bar_figure_render.json').write_text(json.dumps(BAR_CHECKS, indent=2)+'\n')

def main():
    main_figure();encoder_timing();decision_figure();budget_figure();generalization();supplement();transfer_figure();read_case_figure()
    write_bar_outputs()
    (ROOT/'review/figure_render.json').write_text(json.dumps(RENDER,indent=2)+'\n')
    print(json.dumps(RENDER,indent=2))
if __name__=='__main__':main()
