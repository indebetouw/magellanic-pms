
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cmasher as cmr
from astropy import units as u
from astropy.coordinates import SkyCoord
from sklearn import linear_model
import seaborn as sns
from load_data import load_phot
import statsmodels.api as sm
import os

poss_mags = ['mag_555','mag_814','mag_125','mag_160']
poss_cols = ['555min814','125min160']

fuse = 'vis.ir'
all_vi = load_phot(region='n159-all',fuse=fuse)
e_vi = load_phot(region='n159-e',fuse=fuse)
w_vi = load_phot(region='n159-w',fuse=fuse)
s_vi = load_phot(region='n159-s',fuse=fuse)
off_vi = load_phot(region='off-point',fuse=fuse)
region_dicts = {'n159e':e_vi,'n159w':w_vi,'n159s':s_vi,'all':all_vi,'off':off_vi}


print(f'{fuse}:')
print(f'n159e: {len(e_vi)}')
print(f'n159w: {len(w_vi)}')
print(f'n159s: {len(s_vi)}')
print(f'all n159: {len(all_vi)}')
print(f'n159off: {len(off_vi)}')

#############

region = 'all'
r_df = region_dicts[region]

col_use = poss_cols[1]

import matplotlib as mpl

cm_use = cmr.get_sub_cmap(cmr.ember,0,1)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))#, sharex=True, sharey=True)
axs = axs.ravel()
for m in range(len(poss_mags)):
    mag_use = poss_mags[m]
    ax = axs[m]
    ax.scatter(r_df[col_use], r_df[mag_use], c='k', s=0.05, alpha=0.6,zorder=0)
    ss = ax.hist2d(x=r_df[col_use],y=r_df[mag_use],bins=200,cmin=10,cmap=cm_use,norm=mpl.colors.LogNorm())
    ax.set_xlabel(f'F{col_use[0:3]}W - F{col_use[6::]}W [mag]', fontsize=13)
    ax.set_ylabel(f'F{mag_use[4::]}W [mag]', fontsize=14)
    if col_use == '125min160':
        ax.set_xlim(-0.5, 2)
    if col_use == '555min814':
        ax.set_xlim(-0.5, 4)
    ax.invert_yaxis()#set_ylim(27, 14)
    cbar = fig.colorbar(ss[3],ax=ax,label='Counts')
    cbar.set_label('Counts',labelpad=-3,fontsize=10)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(direction='in', which='both', labelsize=9)
fig.tight_layout()




fig, ax = plt.subplots(1,1,figsize=(6,6))
ax.scatter(r_df[poss_cols[0]], r_df[poss_cols[1]], c='k', s=0.05, alpha=0.6, zorder=0)
ss = ax.hist2d(x=r_df[poss_cols[0]], y=r_df[poss_cols[1]], bins=300, cmin=5, cmap=cm_use, norm=mpl.colors.LogNorm())
ax.set_xlim(-0.5,4)
ax.set_ylim(-0.25,2)
ax.axhline(y=0,c='k',ls='--')
ax.axvline(x=0,c='k',ls='--')
ax.set_xlabel(f'F{poss_cols[0][0:3]}W - F{poss_cols[0][6::]}W [mag]', fontsize=13)
ax.set_ylabel(f'F{poss_cols[1][0:3]}W - F{poss_cols[1][6::]}W [mag]', fontsize=13)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(direction='in', which='both', labelsize=9)
fig.tight_layout()




