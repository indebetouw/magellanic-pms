#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 14:50:07 2021

@author: toneill
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

'''
#############################################
Utility script to create a Hess diagram
#############################################
'''

def hess_bin(filt1=None, # first filter
              filt2=None, # second filter)
              xbins=None,ybins=None,# bin edges
              ax=None, # mpl subplot to use
              fig=None): # mplt figure to use
    
    ##### plots filter 1 - filter 2 on x-axis, and filter 1 on yaxis
    # returns counts/bin
        
    # define filt 1 - filt 2 (e.g., V - I)
    f1minf2 = filt1 - filt2
    # counts how many stars in each bin
    binned = stats.binned_statistic_2d(f1minf2,\
                     filt1,None,'count',bins=[xbins,ybins],\
                     expand_binnumbers=True)
    # retrieve bin edges  
    xedges = binned.x_edge
    yedges = binned.y_edge
    
    # Poisson errors for each cell
    errs = np.sqrt(binned.statistic)
    
    # plot hess 
    im = ax.imshow(binned.statistic.T,extent=[xedges[0],xedges[-1],\
               yedges[-1],yedges[0]],cmap='Greys')
    fig.colorbar(im,label='Counts')
    
    # return dictionary with relevant info
    hess_dict = {'counts':binned.statistic.T,
                 'errs':errs,
                 'xedges':xedges,
                 'yedges':yedges}
            
    return hess_dict

if __name__ == '__main__': 
    
    ##### example use case
    xs = np.random.normal(0,1,size=1000)
    ys = np.random.normal(0,1,size=1000)
    
    # define bin sizes
    # here, selected to make life easier by having "square" bins)
    xbins = np.arange(np.min(xs),np.max(ys),0.2)
    ybins = np.linspace(np.min(xs),np.max(ys),len(xbins))
    
    # create fig & plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    hdict = hess_bin(filt1=xs,filt2=ys,
             xbins=xbins,ybins=ybins,
             ax=ax,fig=fig)




