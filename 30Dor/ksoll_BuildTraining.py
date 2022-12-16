#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import os,pickle,sys
if not os.path.abspath("../code") in sys.path:
    sys.path.append(os.path.abspath("../code"))
from scipy import stats
from make_2dKDE import twoD_kde
from make_Hess import hess_bin
from make_GMM import fit_gmm
from sklearn.neighbors import NearestNeighbors



'''
#############################################
Build training set based on Ksoll+ 2018 procedures
#############################################

includes:
    1. construct KDE of on-sky positions, select stars in high density areas
    2. constructing hess diagrams

to do:
    - Integrate extinction correction for CMD
            - later: improve extinction correction procedure from just UMS
    - Create ref. line between LMS/PMS suspected pops
            - later: change to more complicated 0.5-10 Myr isochrone scheme
    - Calculate distances to ref lines
    - Fit 1-d bimodal GMM to distances, quantify fit through EM, assign P(PMS) probs
    - etc etc
'''

def SED_dist(X, Y):
    """Compute the squared Euclidean distance between X and Y."""
    return sum((i-j)**2 for i, j in zip(X, Y))

def nearest_neighbor_bf(*, query_points, reference_points):
    """Use a brute force algorithm to solve the
    "Nearest Neighbor Problem".
    """
    return {
        query_p: min(
            reference_points,
            key=lambda X: SED_dist(X, query_p),
        )
        for query_p in query_points
    }



if __name__ == '__main__': 
    
    # autoset catalog path based on user
    if os.environ['USER'] =='toneill':
        catalogdir = '/Users/toneill/Box/MC/HST/'
    # can expand for others if desired
    else:
        catalogdir="../../MCBox/HST/"
    
    # create fig
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #twoD_kde(RA[z136],Dec[z136],ax=ax1,fig=fig,discrete=False)
    twoD_kde(color136_dered,mag136_dered,
             ax=ax1,fig=fig,discrete=True)
    
    ax1.invert_yaxis()
    
    # make HD
    
    xbins = np.arange(-1,2.5,0.025)
    ybins = np.linspace(12,29,len(xbins))
    
    # create fig & plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    hdict = hess_bin(filt1=mag136_dered,filt2=mag136_dered-color136_dered,
             xbins=xbins,ybins=ybins,
             ax=ax1,fig=fig,cm='viridis')
    #ax1.set_xlabel('F110W - F160W')
    #ax1.set_ylabel('F110W')

    
    
        
    
    ##### MAKE PARALLEL track to assess dist of ages


    plt.figure()
    plt.scatter(color136_dered,mag136_dered,c='k',s=0.02,alpha=0.1)
    
    
    plt.scatter(color136_dered[fin_use_inds],mag136_dered[fin_use_inds],c='c',alpha=0.1,s=0.1)
    plt.gca().invert_yaxis()
    plt.xlabel(colname[0].upper() + " - " + colname[1].upper())
    plt.ylabel(magname[0].upper())
    
    # for 110-160 vs 110
    # not extinction corrected
    '''m = (26.5-13.19)/(1.4-0.3)
    vi_range = np.linspace(1.5,3,1000)
    trans_range = np.linspace(-0.5,2.5,1000)
    intercept = 14'''
    
    
    
    # for 555-775 vs 555
    # extinction corrected
    m = (26.-17.8)/(1+0.3)
    vi_range = np.linspace(1.5,3,1000)
    trans_range = np.linspace(-1,2.5,1000)
    intercept = 18
    
    
    
    
    
    y = m*trans_range + intercept
    plt.plot(trans_range,y,c='r')#3,ls=':')
    
    
    reference_points = [[trans_range[i],
                         y[i]]
                        for i in range(len(y))]
    query_points = [[color136_dered[i],
                         mag136_dered[i]]
                        for i in range(len(mag136_dered))]
    
    
    
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(reference_points)    
    distances, inds = nbrs.kneighbors(query_points)
    
    
    use_inds = (color136_dered <= 2) & (color136_dered >= -0.5) & \
                (mag136_dered <= 24) & (mag136_dered >= 17.5)
    
    fin_use_inds = (distances.flatten() <= 0.7 ) & (use_inds == True)
    
    
    
    
    fit_gmm(X=distances[fin_use_inds],N=[2])
    
    
    # create dictionary with the nearest reference (ISO) point
    # to each query (PMS) point
    # keys are PMS coords
    '''pair_matches = nearest_neighbor_bf(
        reference_points = reference_points,
        query_points = query_points,
    )
    
    transform_ang = -(np.arctan((y-intercept)/trans_range)*180/np.pi)[1] #degrees
    
    diff_vi = np.array([query_points[i][0] - vi_range[0] for i in range(len(query_points))])
    diff_i = np.array([query_points[i][1]-intercept for i in range(len(query_points))])
    d_hypo = np.sqrt(np.array(diff_vi)**2 + np.array(diff_i)**2)
    
    #tx = d_hypo * np.cos(transform_ang*np.pi/180)
    
    tx = diff_vi*np.cos(transform_ang*np.pi/180) - diff_i*np.sin(transform_ang*np.pi/180)'''
    
        
        
    
    
    
    
    
    





