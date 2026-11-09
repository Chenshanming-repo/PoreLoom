#!/usr/bin/env python3
"""Figure S1: both encoders in the existing manuscript style."""
from plot_mean_sd import OFFLINE,FAMS,XL,panel,save,plt,pd,np
from matplotlib.lines import Line2D
def main():
    d=pd.read_csv(OFFLINE/'tableS4_units.csv');fig,axes=plt.subplots(2,2,figsize=(6.5,4.5))
    for ax,letter,metric,scale,label in zip(axes.flat,'abcd',['acc','host_reject','frr','host_pore'],[100,100,100,1],['Accuracy (%)','Hosts rejected (%)','Targets rejected (%)','Pore time per host read (s)']):
        panel(ax,letter)
        for offset,prefix,color,marker in [(-.13,'w','#808A93','s'),(.13,'s','#007E79','o')]:
            q=d.groupby('family')[prefix+'_'+metric].agg(['mean','std','count']).reindex(FAMS);assert q['count'].eq(3).all()
            ax.errorbar(np.arange(4)+offset,q['mean']*scale,yerr=q['std']*scale,fmt=marker,ms=4,capsize=2.5,color=color,elinewidth=1)
        ax.set(xticks=np.arange(4),xticklabels=XL,xlim=(-.5,3.5),ylabel=label);ax.margins(y=.15)
    # Include the R9 sliding-window mean and its SD in panel b.
    axes[0,1].set(ylim=(80,100),yticks=[80,85,90,95,100])
    fig.legend([Line2D([],[],marker='s',lw=0,color='#808A93'),Line2D([],[],marker='o',lw=0,color='#007E79')],['Sliding-window','Streaming-prefix'],loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,1.01))
    fig.subplots_adjust(left=.105,right=.98,bottom=.12,top=.86,wspace=.38,hspace=.65);save(fig,'figS_encoding')
if __name__=='__main__':main()
