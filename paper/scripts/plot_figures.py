#!/usr/bin/env python3
"""Reproduce the three main and four supplementary figures from saved CSVs."""
from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/poreloom-bioinformatics-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'lines.linewidth':1.2,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'poreloom-bioinformatics','savefig.facecolor':'white'})
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
    assert not bounds, (name, bounds)
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
def main_figure():
    prefix, positions = data('fixed_prefix'), data('decision_position_cdf')
    prefix = prefix[prefix.usable_samples.isin([432, 768, 1200, 1968, 3000, 4464, 6000])]
    colors = {'Simulated': '#4477AA', 'R9': '#A463A4', 'R10 mixed': '#007E79', 'R10 sputum': '#C76A44'}
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
    for ax, letter, frame, metric in [(axes[0], 'a', prefix, 'acc'),
                                     (axes[1], 'b', positions[positions.category == 'all'], 'cdf')]:
        panel(ax, letter)
        for family in FAMS:
            subset = frame[frame.family == family]
            if metric == 'cdf':
                # Carry empirical CDFs forward on their union before averaging.
                units = subset.pivot(index='usable_samples', columns='unit', values='cdf').sort_index().ffill().fillna(0)
                q = units.agg(['mean', 'min', 'max'], axis=1)
            else:
                q = subset.groupby('usable_samples')[metric].agg(['mean', 'min', 'max'])
            x = q.index.to_numpy()
            ax.plot(x, q['mean'].to_numpy()*100, color=colors[family], lw=1.5,
                    marker='o' if metric == 'acc' else None, markersize=3.2,
                    drawstyle='steps-post' if metric == 'cdf' else 'default')
            ax.fill_between(x, q['min'].to_numpy()*100, q['max'].to_numpy()*100,
                            color=colors[family], alpha=.12, linewidth=0,
                            step='post' if metric == 'cdf' else None)
        ax.set(xlim=(0, 6300), xticks=[0, 3000, 6000], xlabel='Usable samples')
        ax.axvline(3000, color='#92999D', lw=.7, ls='--', zorder=0)
    axes[0].set(ylabel='Accuracy (%)', ylim=(55, 100), yticks=[60, 80, 100])
    axes[1].set(ylabel='Cumulative decisions (%)', ylim=(0, 102), yticks=[0, 50, 100])
    legend(fig, FAMS, FAMS, colors)
    fig.subplots_adjust(left=.10, right=.985, bottom=.21, top=.78, wspace=.38)
    save(fig, 'fig02_overall')


def encoder_timing():
    encoder, timing = data('encoder'), data('timing_medians')
    fig = plt.figure(figsize=(6.5, 4.7))
    grid = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    colors = {'main': COL['poreloom'], 'window_cp': '#808A93'}
    for ax, letter, key, scale, label in zip(axes[:2], 'ab', ['acc', 'host_pore'],
                                          [100, 1], ['Accuracy (%)', 'Mean host pore time (s)']):
        panel(ax, letter)
        groups(ax, encoder, key, scale, ['main', 'window_cp'], colors, 'variant')
        ax.set_ylabel(label)
    axes[0].set_ylim(85, 98)
    axes[1].set_ylim(bottom=0)
    ax = axes[2]
    panel(ax, 'c')
    for method, label, color in [('incremental', 'Feature reuse', COL['poreloom']), ('scratch', 'Full prefix', '#808A93')]:
        q = timing[(timing.scenario == 'aligned') & (timing.chunk == 1600) &
                   (timing.batch == 512) & (timing.method == method) & (timing.n <= 19200)].sort_values('n')
        ax.plot(q.n/1000, q.median_ms, 'o-', ms=3, color=color, label=label)
        ax.fill_between(q.n/1000, q.min_ms, q.max_ms, color=color, alpha=.13)
    ax.set(xlabel='Received samples (thousands)', ylabel='Update time (ms)', ylim=(0, 650))
    ax.legend(frameon=False, loc='upper left')
    legend(fig, ['main', 'window_cp'], ['Streaming prefix + FACP', 'Sliding window + FACP'], colors, ncol=2)
    fig.subplots_adjust(left=.11, right=.985, bottom=.12, top=.86, hspace=.75, wspace=.40)
    save(fig, 'fig03_encoder_cache')


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

def main():
    from draw_overview import draw
    from plot_beta import render as render_beta, saved as saved_beta
    from analyze_matched_rejection import render as render_matched, saved as saved_matched
    draw(OUT)
    main_figure()
    encoder_timing()
    budget_figure()
    generalization()
    render_beta(*saved_beta())
    summary, audit = saved_matched()
    render_matched(summary, audit)
    (ROOT/'review/stopping_matched_render.json').write_text(json.dumps(audit['figure'], indent=2)+'\n')
    (ROOT/'review/figure_render.json').write_text(json.dumps(RENDER, indent=2)+'\n')
    print(json.dumps(RENDER, indent=2))


if __name__ == '__main__':
    main()
