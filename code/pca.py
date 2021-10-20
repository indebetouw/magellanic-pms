"""
@author: toneill
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
import seaborn as sn
import numpy as np
import os
from matplotlib.colors import DivergingNorm
# autoset catalog path based on user
if os.environ['USER'] =='toneill':
    catalogdir = '/Users/toneill/Box/MC/HST/'
else:
    catalogdir="../../MCBox/HST/"
    
# load HTTP catalog
full = pd.read_csv(catalogdir+'trim_HTTP.2015_10_20.1.csv')
for m in ['m_f110w','m_f160w','m_f555w','m_f775u']:
    full.drop(full[full[m]>30].index,inplace=True) 

def scale(df):
    scaler = StandardScaler()
    clus_scaled = scaler.fit_transform(df)
    return clus_scaled

df = full[['m_f110w_dered','m_f160w_dered','m_f555w_dered','m_f775u_dered','AvNN']]
df_scaled = scale(df)

def pca_clumps(clumps_df,ncomps=3,pca_stats=False,plot=True):
    
    # Apply PCA
    pca = PCA(n_components = ncomps)
    pca_fit = pca.fit_transform(df_scaled)
    clus_principal = pd.DataFrame(pca_fit)
    #clus_principal.columns = ['P1','P2','P3','P4','P5','P6','P7','P8']

    if pca_stats == True:
        
        print(clus_principal.head())
        print(pca.explained_variance_ratio_)
    
    if plot == True:
        
        # check correlations
        cor = pd.DataFrame(df).corr()    
        mask = np.triu(np.ones_like(cor,dtype=np.bool))
        plt.figure()
        sn.heatmap(cor,annot=True,mask=mask,vmin=-1,vmax=1)
        plt.xticks(rotation=15)
        plt.yticks(rotation=55)
        plt.title('Correlations')
        
        # effects of each dimension on factor plot
        plt.figure(figsize=(8,4))
        plt.imshow(pca.components_,interpolation='none',cmap='plasma',vmin=-1,vmax=1)
        feature_names = list(clumps_df.columns)
        #plt.gca().set_xticks(np.arange(-.5, len(feature_names)));
        #plt.gca().set_yticks(np.arange(0.5, 9));
        plt.gca().set_xticklabels(feature_names, rotation=60, ha='left', fontsize=10);
        plt.gca().set_yticklabels(['PC1', 'PC2','PC3','PC4'], \
               va='bottom', fontsize=12);
        plt.colorbar(orientation='horizontal', ticks=[pca.components_.min(), 0,
                                                      pca.components_.max()], pad=0.2)
        plt.title('Impact of Observables on PCs')
            
        # cumulative explained variance
        plt.figure()
        plt.plot(np.linspace(1,3,3),np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')    
        plt.title('Cumulative Explained Variance of PCs')
        plt.ylim(0,1)
        #plt.xlim(1,8)
        
        # scree plot
        plt.figure()
        plt.plot(np.linspace(1,3,3),pca.explained_variance_)
        plt.xlabel('Number of Components')
        plt.ylabel('Eigenvalue')    
        plt.title('Scree Plot of PCs')
        #plt.ylim(0,1)
        #plt.xlim(1,8)
        
        # scatter
        #plt.scatter(clus_principal['P1'],clus_principal['P2'])
        
        score = pca_fit[:,0:2]
        coeff = np.transpose(pca.components_[0:2, :])
        labels = list(clumps_df.columns)
        
        # biplot
        plt.figure()
        xs = score[:,0]
        ys = score[:,1]
        n = coeff.shape[0]
        scalex = 1.0/(xs.max() - xs.min())
        scaley = 1.0/(ys.max() - ys.min())

        norm = DivergingNorm(vmin=0, vcenter=0.5,vmax=1) 
        plt.scatter(xs * scalex,ys * scaley,c=full_prob[:,1],
                    cmap='RdYlBu_r',s=0.1,alpha=0.5,norm=norm)

        for i in range(n):
            plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
            if labels is None:
                plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'green', ha = 'center', va = 'center')
            else:
                plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
     
        plt.xlabel("PC{}".format(1))
        plt.ylabel("PC{}".format(2))
        plt.grid()
        #plt.ylim(-1,1)
        #plt.xlim(-0.4,0.8)
        plt.title('Biplot')
                           
    return clus_principal

