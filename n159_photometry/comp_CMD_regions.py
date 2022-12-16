
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import pandas as pd
import vaex as vx
import os
from astropy.table import Table
from astropy.wcs import wcs
import subprocess
import cmasher as cmr
from astropy import units as u
from astropy.coordinates import SkyCoord
#photdir = '/Users/toneill/N159/photometry/new/'
savephotdir = '/Users/toneill/N159/photometry/reduced/'


#e_vi = pd.read_csv(photdir+region+'new_new_xmatch_0.2sec.555.814_f814ref.csv')
#w_vi = pd.read_csv(photdir+'n159-w_xmatch_0.3sec.555.814_f814ref.csv')
#w_vi = pd.read_csv(photdir+'n159-w_xmatch_0.1sec.555.814_f814ref_offsetByMeanClus.csv')
#s_vi = pd.read_csv(photdir+'n159-s_xmatch_0.1sec.555.814_f814ref.csv')

w_5coords = SkyCoord(w_vi['ra_555']*u.deg,w_vi['dec_555']*u.deg ,frame='fk5')
w_8coords = SkyCoord(w_vi['ra_814']*u.deg,w_vi['dec_814']*u.deg ,frame='fk5')

off_w = w_5coords.spherical_offsets_to(w_8coords)
off_w_ra = off_w[0].to(u.arcsec).value
off_w_dec = off_w[1].to(u.arcsec).value

w_vi['deltaRA'] = off_w_ra
w_vi['deltaDec'] = off_w_dec
w_vi['magOff'] = np.sqrt(off_w_ra**2 + off_w_dec**2)
#s_vi.to_csv('n159-s_xmatch_0.1sec.555.814_f814ref_offset.csv')

##########################################################

#e_vi.columns = np.append('index',w_vi.columns[0:-1])

#usecrit = [e_vi['magOff'] <= 0.2]

plt.figure()
plt.scatter(w_vi['deltaRA'].values,w_vi['deltaDec'].values,s=0.5)#,c=e_vi['magOff'],cmap='tab10',s=0.1)#,cmap='tab10')
#plt.scatter(w_vi['deltaRA'][clus_offs],w_vi['deltaDec'][clus_offs],c='r',s=0.1)#,cmap='tab10')
plt.xlabel('$\delta$ RA [ " ] ')
plt.ylabel('$\delta$ Dec [ " ] ')
#plt.title('N159w with corrected 555w coords')
#plt.colorbar(label='Mag of Offset ["]')
plt.tight_layout()




plt.figure()
plt.scatter(w_vi['ra_555'].values,w_vi['dec_555'].values,s=0.5,cmap='tab10')
plt.xlabel('RA')
plt.ylabel('Dec')
plt.title('N159w')
plt.colorbar(label='Mag of Offset ["]')
plt.tight_layout()


# w_5coords.spherical_offsets_to(w_8coords)
# so is offset from 555 to 814
clus_offs = (off_w_ra <= -0.15 ) & (off_w_ra >= -0.19) & (off_w_dec <= -0.16) & (off_w_dec >= -0.21)
mean_ra_off = (-0.15+-0.19)/2 # arcsec
mean_dec_off = (-0.16 + -0.21)/2 # arcsec

np.mean(w_vi['deltaRA'][clus_offs])
np.mean(w_vi['deltaDec'][clus_offs])

#target_star = SkyCoord(86.75309*u.deg, -31.5633*u.deg, frame='icrs')
#>>> target_star.spherical_offsets_by(1.3*u.arcmin, -0.7*u.arcmin)


clus_offs = (off_w_ra <= 0.013) & (off_w_ra >= - 0.0095) & (off_w_dec <= 0.021) & (off_w_dec >= 0.004)

plt.figure()
plt.scatter(w_vi['ra_555'],w_vi['dec_555'],s=0.05)#,c='r')
plt.scatter(w_vi['ra_555'][clus_offs],w_vi['dec_555'][clus_offs],s=0.2,c='r')
plt.xlabel('RA')
plt.ylabel('Dec')
plt.title('N159e')
#plt.colorbar(label='Mag of Offset ["]')
plt.tight_layout()


####################################################

def plot_scatter(r_df,ax=None,kwargs=None):
    if ax is not None:
        ax.scatter(r_df['555min814'],r_df['mag_555'],**kwargs)
        ax.invert_yaxis()
    if ax is None:
        plt.scatter(r_df['555min814'], r_df['mag_555'], **kwargs)

e_vi = pd.read_csv(savephotdir+'n159-e_reduce.phot.csv')
w_vi = pd.read_csv(savephotdir+'n159-w_reduce.phot.csv')
s_vi = pd.read_csv(savephotdir+'n159-s_reduce.phot.csv')


plotargs = {'s':0.3,'c':'royalblue','alpha':0.5}
fig, [ax1,ax2,ax3] = plt.subplots(1,3,figsize=(12,5))
plot_scatter(e_vi,ax=ax1,kwargs=plotargs)
plot_scatter(w_vi,ax=ax2,kwargs=plotargs)
plot_scatter(s_vi,ax=ax3,kwargs=plotargs)
ax1.set_xlim(-1,4)
ax2.set_xlim(-1, 4)
ax3.set_xlim(-1,4)
ax1.set_title('N159e')
ax2.set_title('N159w')
ax3.set_title('N159s')
[ax.set_xlabel('F555w - F814w') for ax in [ax1,ax2,ax3]]
[ax.set_ylabel('F555w') for ax in [ax1,ax2,ax3]]
fig.tight_layout()

plt.savefig(savephotdir+'compCMDs.png',dpi=300)




fig, [ax1,ax2,ax3]= plt.subplots(1,3,figsize=(10,4))
for i in range(3):
    df = [e_vi,w_vi,s_vi][i]
    ax = [ax1,ax2,ax3][i]
    ax.hist(df['mag_555'],bins=35,label='F555w',color='firebrick',alpha=0.8)
    ax.hist(df['mag_814'],bins=35,label='F814w',alpha=0.8,color='cornflowerblue')
    ax.legend(loc='upper left')
ax1.set_title('N159E')
ax2.set_title('N159W')
ax3.set_title('N159S')
fig.tight_layout()

plt.savefig(savephotdir+'hist_compVI.png',dpi=300)

