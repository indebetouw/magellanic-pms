#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 15:20:55 2021

@author: toneill
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

'''
#############################################
Utility script to create a univariate Gaussian mixture model
#############################################
'''

from matplotlib import pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

# mostly taken from https://www.astroml.org/book_figures/chapter4/fig_GMM_1D.html
# lots of room to improve this


def fit_gmm(X=None, # one dimensional variable to fit 
            N=2):# # number of components in GMM to fit - could be list to compare multiple
            #fig = None)#, ax1 = None,ax2=None):
    
    models = [None for i in range(len(N))]
    
    for i in range(len(N)):
        models[i] = GaussianMixture(N[i]).fit(X)
    
    # compute the AIC and the BIC
    AIC = [m.aic(X) for m in models]
    BIC = [m.bic(X) for m in models]
    
    #------------------------------------------------------------
    # Plot the results
    #   1) data + best-fit mixture
    #   SKIPPED HERE BUT COULD ADD BACK: 2) AIC and BIC vs number of components
    #   3) probability that a point came from each component
    
    fig = plt.figure(figsize=(10, 4))
    fig.subplots_adjust(left=0.12, right=0.97,
                        bottom=0.21, top=0.9, wspace=0.5)
    
    
    # plot 1: data + best-fit mixture
    ax = fig.add_subplot(121)
    M_best = models[np.argmin(AIC)]
    
    x = np.linspace(-6, 6, 1000)
    logprob = M_best.score_samples(x.reshape(-1, 1))
    responsibilities = M_best.predict_proba(x.reshape(-1, 1))
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]
    
    ax.hist(X, 30, density=True, histtype='stepfilled', alpha=0.4)
    ax.plot(x, pdf, '-k')
    ax.plot(x, pdf_individual, '--k')
    ax.text(0.04, 0.96, "Best-fit Mixture",
            ha='left', va='top', transform=ax.transAxes)
    ax.set_xlabel('$x$')
    ax.set_ylabel('Density')
    
    '''
    # plot 2: AIC and BIC
    ax = fig.add_subplot(132)
    ax.plot(N, AIC, '-k', label='AIC')
    ax.plot(N, BIC, '--k', label='BIC')
    ax.set_xlabel('n. components')
    ax.set_ylabel('information criterion')
    ax.legend(loc=2)'''
    
    
    # plot 3: posterior probabilities for each component
    ax = fig.add_subplot(122)
    
    p = responsibilities
    p = p[:, (1, 0)]  # rearrange order so the plot looks better
    p = p.cumsum(1).T
    
    ax.fill_between(x, 0, p[0], color='gray', alpha=0.3)
    ax.fill_between(x, p[0], p[1], color='gray', alpha=0.5)
    ax.fill_between(x, p[1], 1, color='gray', alpha=0.7)
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 1)
    ax.set_xlabel('$x$')
    ax.set_ylabel(r'$p({\rm class}|x)$')
    
    ax.text(-5, 0.3, 'class 1', rotation='vertical')
    #ax.text(0, 0.5, 'class 2', rotation='vertical')
    ax.text(3, 0.3, 'class 2', rotation='vertical')
    
    plt.show()

    return models


if __name__ == '__main__': 

    # example use
    X = np.concatenate([np.random.normal(-2, 1.5, 350),
                        #random_state.normal(0, 1, 500),
                        np.random.normal(3, 0.8, 150)]).reshape(-1, 1)

    fit_gmm(X=X,N=[2])


