
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
from astropy.coordinates import SkyCoord
from astropy import units as u

photdir = '/Users/toneill/N159/photometry/'


#zarhar = Table.read(photdir+'zar10arcmin.fit',format='ascii')
zar =  fits.open(photdir+'zar10arcmin.fit')[1].data
zar = zar[zar['Imag'] > 0]

z_ra = zar['RAJ2000']
z_dec = zar['DEJ2000']
z_v = zar['Vmag']
z_i = zar['Imag']
z_vmini = z_v-z_i

plt.figure()
plt.scatter(z_vmini,z_v,s=0.5,alpha=0.4,c='k')#c=zar['Flag'],cmap='tab20c')#'royalblue')#,c=z_i)
plt.gca().invert_yaxis()
#plt.colorbar()

cmatch_vi = pd.read_csv(photdir+'n159-s_xmatch_0.1sec.555.814_f814ref.csv')
saturated = pd.read_csv(photdir + 'n159-s_xmatch_0.1sec_Saturated.555.814_f814ref.csv')

csat = SkyCoord(ra=saturated['ra_555'] * u.deg, dec=saturated['dec_555'] * u.deg)
czar = SkyCoord(ra=z_ra* u.deg, dec=z_dec* u.deg)
max_sep = 0.1 * u.arcsec
idx, d2d, d3d = csat.match_to_catalog_sky(czar)
sep_constraint = d2d < max_sep
print(np.sum(sep_constraint))
csat_matches = saturated[sep_constraint]
czar_matches = pd.DataFrame(zar[idx[sep_constraint]])
hz_mat = czar_matches.join(csat_matches)
hz_mat['Z_VminI'] = hz_mat['Vmag'] - hz_mat['Imag']

plt.figure()
plt.scatter(hz_mat['Z_VminI'], hz_mat['Vmag'], s=5, alpha=1,c='r',label='Z&H with saturated HST match (N159S ONLY!)')
plt.scatter(zar['Vmag']-zar['Imag'], zar['Vmag'], s=0.3, alpha=0.4,c='grey',label='Full Z&H',zorder=0)
plt.gca().invert_yaxis()
plt.xlabel('V - I [mag]')
plt.ylabel('V [mag]')
plt.legend()
plt.title("Z&H04 within 10' of N159")
plt.ylim(23,10)


##########################

cfull = SkyCoord(ra=cmatch_vi['ra_555'] * u.deg, dec=cmatch_vi['dec_555'] * u.deg)
czar = SkyCoord(ra=z_ra* u.deg, dec=z_dec* u.deg)
max_sep = 0.2 * u.arcsec
idx, d2d, d3d = cfull.match_to_catalog_sky(czar)
sep_constraint = d2d < max_sep
print(np.sum(sep_constraint))
cfull_matches = cmatch_vi[sep_constraint]
czar_matches = pd.DataFrame(zar[idx[sep_constraint]])
hz_mat_full = czar_matches.join(cfull_matches)
hz_mat_full['Z_VminI'] = hz_mat_full['Vmag'] - hz_mat_full['Imag']

plt.figure()
plt.scatter(hz_mat_full['Z_VminI'], hz_mat_full['Vmag'], s=5, alpha=1,c='b',label='Z&H with non-saturated HST match (N159S ONLY!)')
plt.scatter(zar['Vmag']-zar['Imag'], zar['Vmag'], s=0.3, alpha=0.4,c='grey',label='Full Z&H',zorder=0)
plt.gca().invert_yaxis()
plt.xlabel('V - I [mag]')
plt.ylabel('V [mag]')
plt.legend()
plt.title("Z&H04 within 10' of N159")
plt.ylim(23,10)


fig = plt.figure(figsize=(10,3))
ax1 = fig.add_subplot(131)
plt.scatter(hz_mat_full['Z_VminI'], hz_mat_full['555min814'], s=5, alpha=1,c='b')#,label='Z&H with non-saturated HST match (N159S ONLY!)')
plt.xlabel('Z&H V - I')
plt.ylabel('Dolphot F555W - F814W')
plt.plot([-2,4],[-2,4],c='k',ls=':')
plt.title('V - I comparison')

ax2 = fig.add_subplot(132)
plt.scatter(hz_mat_full['Vmag'], hz_mat_full['mag_555'], s=5, alpha=1,c='b')#,label='Z&H with non-saturated HST match (N159S ONLY!)')
plt.xlabel('Z&H V [mag]')
plt.ylabel('Dolphot F555W [mag]')
plt.plot([15,25],[15,25],c='k',ls=':')
plt.title('V comp')

ax3 = fig.add_subplot(133)
plt.scatter(hz_mat_full['Imag'], hz_mat_full['mag_814'], s=5, alpha=1,c='b')#,label='Z&H with non-saturated HST match (N159S ONLY!)')
plt.xlabel('Z&H I [mag]')
plt.ylabel('Dolphot F8142 [mag]')
plt.plot([15,25],[15,25],c='k',ls=':')
plt.title('I comp')

fig.tight_layout()





