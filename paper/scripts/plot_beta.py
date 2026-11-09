#!/usr/bin/env python3
"""Figure S4: fitted beta policies and the shared accuracy-tolerance interval."""
from plot_mean_sd import OFFLINE,FAMS,panel,save,plt,pd,np
from matplotlib.lines import Line2D

def main():
    d=pd.read_csv(OFFLINE/'figS_beta_units.csv');colors=['#4477AA','#A463A4','#007E79','#C76A44']
    accuracy=d.groupby(['beta','family']).acc.mean().unstack().reindex(columns=FAMS)
    within_one_pp=(accuracy.max()-accuracy).le(.01+1e-12).all(axis=1)
    passing=accuracy.index[within_one_pp].to_numpy()
    assert len(passing) and 1.0 in passing
    beta_lo,beta_hi=passing.min(),passing.max()
    # Shade the span of tested weights satisfying the criterion in every dataset.
    assert within_one_pp.loc[beta_lo:beta_hi].all()
    fig,axes=plt.subplots(2,2,figsize=(6.5,6.1))
    for ax,letter,metric,scale,label in zip(axes.flat,'abcd',['acc','frr','host_reject','host_pore'],[100,100,100,1],['Accuracy (%)','Targets rejected (%)','Hosts rejected (%)','Pore time per host read (s)']):
        panel(ax,letter)
        ax.axvspan(beta_lo,beta_hi,color='#D4D7D9',alpha=.50,lw=0,zorder=0)
        for family,color in zip(FAMS,colors):
            q=d[d.family==family].groupby('beta')[metric].agg(['mean','std','count']).sort_index()
            assert len(q)==42 and q['count'].eq(3).all()
            x=q.index.to_numpy();y=q['mean'].to_numpy()*scale;sd=q['std'].to_numpy()*scale
            ax.plot(x,y,color=color,lw=1.5)
            ax.fill_between(x,y-sd,y+sd,color=color,alpha=.12,linewidth=0)
        ax.axvline(1,color='#92999D',lw=.8,ls='--',zorder=0)
        ax.set(xscale='log',xlim=(.03,300),xlabel=r'$\beta$ (log scale)',ylabel=label)
        ax.set_xticks([.03,.3,1,3,30,300],['0.03','0.3','1','3','30','300']);ax.tick_params(axis='x',labelsize=9)
    axes[0,0].set(ylim=(50,101));axes[0,1].set(ylim=(0,65));axes[1,0].set(ylim=(0,103));axes[1,1].set(ylim=(0,10))
    fig.legend([Line2D([],[],color=c,lw=1.5) for c in colors],FAMS,loc='upper center',bbox_to_anchor=(.5,1.0),ncol=4,frameon=False,columnspacing=1,handlelength=1.1)
    fig.subplots_adjust(left=.105,right=.98,bottom=.092,top=.88,wspace=.38,hspace=.48)
    save(fig,'figS_beta')
    print('Rendered all 504 unit-level outcomes; beta=1 marks the main policy.')
    print('Accuracy-tolerance beta interval:',beta_lo,beta_hi)
    print('Effective normalization interval (s):',15*beta_lo,15*beta_hi)

if __name__=='__main__':main()
