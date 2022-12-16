#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 14:27:00 2021

@author: toneill
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy import stats

'''
#############################################
Utility script to create a 2d kernel density estimation
#############################################
'''

def twoD_kde(x, # x positions (e.g. RA)
             y, # y positions (e.g. Dec)
             ax=None, # mpl subplot to use
             fig=None, # mpl figure to use
             discrete=False,cmap_kde=cmr.flamingo_r): # plot continous kde if False, plot non-cont if True

    # Define cell size & extent of intial kde build
    deltaX = (max(x) - min(x))/10
    deltaY = (max(y) - min(y))/10
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY
    #print(xmin, xmax, ymin, ymax)
    
    #### NOTE: this step can be very time consuming
    # Create meshgrid
    xx, yy = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]    
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    # initiate kernel
    kernel = stats.gaussian_kde(values)#,bw_method='silverman')
    # perform density estimation
    f = np.reshape(kernel(positions).T, xx.shape) 
        
    # plot
    # cset command creates a cartoonish representation of kde levels
    # imshow command shows the actual (continous) kde
    
    if discrete:
        cset = ax.contourf(xx, yy, f, cmap='inferno')
    if discrete == False:
        # sigma contours
        cont_levs = np.arange(0.5*np.std(f),6*np.std(f),0.5*np.std(f))
        im =ax.imshow((np.rot90(f)), cmap=cmap_kde, extent=[xmin, xmax, ymin, ymax],zorder=0)
        #ax.contour(xx, yy, f, cmap='Blues_r',linewidths=1,levels=cont_levs)#,label=lab)
    #ax.clabel(cset, inline=1, fontsize=10)
    
    ax.set_xlim(xmax,xmin)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.minorticks_on()
    
    # force fix aspect ratio of subplot, since gets wonky 
    # if x and y extents are unequal
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    
    ### create colorbar
    #ax_divider = make_axes_locatable(ax)
    # Add an axes above the main axes.
    #cax = ax_divider.append_axes("top", size="1%",pad="0%")#size="7%", 
    #cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    # Change tick position to top (with the default tick position "bottom", ticks
    # overlap the image).
    #cax.xaxis.set_ticks_position("top")   
    # fix aspect ratios and add colorbar
    #fig.colorbar(im,ax=ax,pad=cbar_pad,label='Density')#,orientation='horizontal')#,label='Density')
    
    return

if __name__ == '__main__': 
    
    ##### example use case
    
    xs = np.random.normal(0,10,size=500)
    ys = np.random.normal(0,10,size=500)
    
    # create fig
    fig = plt.figure()
    ax = fig.add_subplot(111)
    twoD_kde(xs,ys,ax=ax,fig=fig,discrete=False)


