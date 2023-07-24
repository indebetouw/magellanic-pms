
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
visir_dict = region_dicts

fuse = 'vis'
all_vi = load_phot(region='n159-all',fuse=fuse)
e_vi = load_phot(region='n159-e',fuse=fuse)
w_vi = load_phot(region='n159-w',fuse=fuse)
s_vi = load_phot(region='n159-s',fuse=fuse)
off_vi = load_phot(region='off-point',fuse=fuse)
region_dicts = {'n159e':e_vi,'n159w':w_vi,'n159s':s_vi,'all':all_vi,'off':off_vi}
vis_dict = region_dicts

fuse = 'ir'
all_vi = load_phot(region='n159-all',fuse=fuse)
e_vi = load_phot(region='n159-e',fuse=fuse)
w_vi = load_phot(region='n159-w',fuse=fuse)
s_vi = load_phot(region='n159-s',fuse=fuse)
off_vi = load_phot(region='off-point',fuse=fuse)
region_dicts = {'n159e':e_vi,'n159w':w_vi,'n159s':s_vi,'all':all_vi,'off':off_vi}
ir_dict = region_dicts


print(f'{fuse}:')
print(f'n159e: {len(e_vi)}')
print(f'n159w: {len(w_vi)}')
print(f'n159s: {len(s_vi)}')
print(f'all n159: {len(all_vi)}')
print(f'n159off: {len(off_vi)}')

#############
cm_use = cmr.get_sub_cmap(cmr.toxic,0,1)
import matplotlib as mpl


region = 'off'
r_df = region_dicts[region]
col_use = poss_cols[1]

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.ravel()
for m in range(len(poss_mags)):
    mag_use = poss_mags[m]
    ax = axs[m]
    ax.scatter(r_df[col_use], r_df[mag_use], c='k', s=0.1, alpha=0.6,zorder=0)
    ss = ax.hist2d(x=r_df[col_use],y=r_df[mag_use],bins=150,cmin=10,cmap=cm_use)#,norm=mpl.colors.LogNorm())
    ax.set_xlabel(f'F{col_use[0:3]}W - F{col_use[6::]}W [mag]', fontsize=13)
    ax.set_ylabel(f'F{mag_use[4::]}W [mag]', fontsize=14)
    if col_use == '125min160':
        ax.set_xlim(-0.5, 2)
    if col_use == '555min814':
        ax.set_xlim(-0.5, 4)
    ax.invert_yaxis()#set_ylim(27, 14)
    cbar = fig.colorbar(ss[3],ax=ax,label='Counts',pad=0)
    cbar.set_label('Counts',fontsize=10)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(direction='in', which='both', labelsize=9)
fig.tight_layout()
##########################################

xlim_dict = {poss_cols[0]:[-1,4],poss_cols[1]:[-1,2]}
ylim_dict = {poss_mags[0]:[28,17],poss_mags[1]:[28,17],poss_mags[2]:[26,12],poss_mags[3]:[26,12]}
nhbox = {poss_cols[0]:80,poss_cols[1]:100}

cm_use = cmr.get_sub_cmap(cmr.toxic,0,1)


def plot4pan(col_use=poss_cols[0],axs=axs):
    for m in range(len(poss_mags)):
        mag_use = poss_mags[m]
        ax = axs[m]
        try:
            ax.scatter(v_df[col_use], v_df[mag_use], c='grey', s=0.3, alpha=1, zorder=0)#, label='F555W & F814W')##3A86FF
        except:
            print()
        try:
            ax.scatter(ir_df[col_use], ir_df[mag_use], c='grey', s=0.3, alpha=1, zorder=0)# label='F125W & F160W')##BA5A31
        except:
            print()
        try:
            ax.scatter(vir_df[col_use], vir_df[mag_use], c='k', s=0.3, alpha=0.8, zorder=1)#, label='4 band')
            ss = ax.hist2d(x=vir_df[col_use], y=vir_df[mag_use], bins=nhbox[col_use], cmin=10, cmap=cm_use)  # ,norm=mpl.colors.LogNorm())
        except:
            print()
        ax.set_xlabel(f'F{col_use[0:3]}W - F{col_use[6::]}W [mag]', fontsize=13)
        ax.set_ylabel(f'F{mag_use[4::]}W [mag]', fontsize=14)
        xlim_u = xlim_dict[col_use]
        ax.set_xlim(xlim_u[0],xlim_u[1])
        ylim_u = ylim_dict[mag_use]
        ax.set_ylim(ylim_u[0],ylim_u[1])
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(direction='in', which='both', labelsize=9)
    return

sdir = '/Users/toneill/N159/Plots/cmds/'

col_use = poss_cols[1]
region = 'off'
v_df = vis_dict[region]
ir_df = ir_dict[region]
vir_df = visir_dict[region]

fig, axs = plt.subplots(2,4, figsize=(16, 8))
axs = axs.ravel()
plot4pan(poss_cols[0],axs[0:4])
plot4pan(poss_cols[1],axs[4::])
fig.tight_layout()
ax = axs[3]
ax.scatter(100,100,c='grey',label='2 band')
ax.scatter(100,100,c='k',label='4 band',marker='s')
ax.legend(loc='upper right',title=f'{region.upper()}',fontsize=9)
plt.savefig(f'{sdir}bulkcmd_{region}.png',dpi=300)



#7FB069













fig, axs = plt.subplots(2,4, figsize=(16, 10))
axs = axs.ravel()
for m in range(len(poss_mags)):
    mag_use = poss_mags[m]
    ax = axs[m]
    try:
        ax.scatter(v_df[col_use], v_df[mag_use], c='royalblue', s=0.3, alpha=0.6,zorder=0,label='F555W & F814W')
    except:
        print()
    try:
        ax.scatter(ir_df[col_use], ir_df[mag_use], c='firebrick', s=0.3, alpha=0.6,zorder=0,label='F125W & F160W')
    except:
        print()
    try:
        ax.scatter(vir_df[col_use], vir_df[mag_use], c='k', s=0.1, alpha=0.6,zorder=0,label='4 band')
    except:
        print()
    ax.legend(loc='upper right')
    ax.set_xlabel(f'F{col_use[0:3]}W - F{col_use[6::]}W [mag]', fontsize=13)
    ax.set_ylabel(f'F{mag_use[4::]}W [mag]', fontsize=14)
    if col_use == '125min160':
        ax.set_xlim(-0.5, 2)
    if col_use == '555min814':
        ax.set_xlim(-0.5, 4)
    ax.invert_yaxis()#set_ylim(27, 14)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(direction='in', which='both', labelsize=9)


fig.tight_layout()












##########################################

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




