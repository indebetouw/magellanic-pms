
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

savephotdir = '/Users/toneill/N159/photometry/reduced/'
e_vi = pd.read_csv(savephotdir+'n159-e_reduce.phot.cutblue.csv')
w_vi = pd.read_csv(savephotdir+'n159-w_reduce.phot.cutblue.csv')
s_vi = pd.read_csv(savephotdir+'n159-s_reduce.phot.cutblue.csv')
all_vi = pd.read_csv(savephotdir+'n159-all_reduce.phot.cutblue.csv')
region_dicts = {'n159e':e_vi,'n159w':w_vi,'n159s':s_vi,'all':all_vi}

r_df = region_dicts['all']

#zarhar = Table.read(photdir+'zar10arcmin.fit',format='ascii')
zar =  fits.open(photdir+'old/zar10arcmin.fit')[1].data
zar = zar[zar['Imag'] > 0]

z_ra = zar['RAJ2000']
z_dec = zar['DEJ2000']
z_v = zar['Vmag']
z_i = zar['Imag']
z_vmini = z_v-z_i

zar_coords = SkyCoord(ra=z_ra * u.deg, dec=z_dec * u.deg)
hst_coords = SkyCoord(ra=r_df['ra_814'] * u.deg, dec=r_df['dec_814'] * u.deg)

max_sep = 15 * u.arcsec
idx, d2d, d3d = zar_coords.match_to_catalog_sky(hst_coords)
sep_constraint = d2d < max_sep
print(np.sum(sep_constraint))
zar_matches = zar[sep_constraint]

#pd.DataFrame(zar_matches).to_csv(photdir+'zaritsky04_matchN159.csv',index=False)

z_ra = zar_matches['RAJ2000']
z_dec = zar_matches['DEJ2000']
z_v = zar_matches['Vmag']
z_i = zar_matches['Imag']
z_b = zar_matches['Bmag']


z_ev = zar_matches['e_Vmag']
z_ei = zar_matches['e_Imag']
z_eb = zar_matches['e_Bmag']
#z_u = zar_matches['Umag']

z_vmini = z_v-z_i
z_vminb = z_v - z_b


z_evmini = z_ev-z_ei
z_evminb = z_ev - z_eb

def convert_bvri_hst(m_bvri,c_bvri,c0=None,c1=None,c2=None,zeropt=0):
    #https://arxiv.org/pdf/astro-ph/0507614.pdf
    #https://iopscience.iop.org/article/10.1086/444553/pdf
    return -(c0 + c1 * c_bvri + c2 * c_bvri**2) + m_bvri + zeropt


z_555 =convert_bvri_hst(z_v,z_vmini, c0=25.719, c1=-0.088, c2=0.043,zeropt=25.724)
#z_555[z_vmini > 0.4] = convert_bvri_hst(z_v[z_vmini > 0.4],z_vmini[z_vmini > 0.4],
#                                        c0=25.735, c1=-0.106, c2=0.013,zeropt=25.724)
z_814 = convert_bvri_hst(z_i,z_vmini, c0=25.489, c1=0.041, c2=-0.093,zeropt=25.501)
#z_814[z_vmini > 0.1] = convert_bvri_hst(z_v[z_vmini > 0.1],z_vmini[z_vmini > 0.1],
#                                        c0=25.496, c1=-0.014, c2=0.015,zeropt=25.501)
z_5min8 = z_555 - z_814

z_e555 =convert_bvri_hst(z_ev,z_evmini, c0=25.719, c1=-0.088, c2=0.043,zeropt=25.724)
z_e814 = convert_bvri_hst(z_ei,z_evmini, c0=25.489, c1=0.041, c2=-0.093,zeropt=25.501)


z_475 = convert_bvri_hst(z_b,z_vminb, c0 = 26.145, c1=0.220, c2=-0.05, zeropt=26.168)
z_e475 = convert_bvri_hst(z_eb,z_evminb, c0 = 26.145, c1=0.220, c2=-0.05, zeropt=26.168)




zar_matches = pd.DataFrame(zar_matches)
zar_matches['f555w'] = z_555
zar_matches['f814w'] = z_814
zar_matches['555min814'] = z_5min8
zar_matches['f475w'] = z_475

def vega_flux(m_hst,F_vega):
    #https://hst-docs.stsci.edu/acsdhb/chapter-5-acs-data-analysis/5-1-photometry
    return 10**(-1/2.5 * m_hst) * F_vega

flux_814 = vega_flux(z_814,2440.74)  * 1000 #m jansky
flux_555 = vega_flux(z_555,3662.24)  * 1000 #m jansky
flux_475 = vega_flux(z_475,4004.42)  * 1000 #m jansky

flux_e814 = (z_e814/z_814) * flux_814
flux_e555 =  (z_e555/z_555) * flux_555
flux_e475 = (z_e475/z_475) * flux_475

zar_matches['flux814'] = flux_814
zar_matches['e_flux814'] = flux_e814
zar_matches['flux555'] = flux_555
zar_matches['e_flux555'] = flux_e555
zar_matches['flux475'] = flux_475
zar_matches['e_flux475'] = flux_e475


zar_matches['mag_555'] = zar_matches['f555w']
zar_matches = zar_matches[zar_matches['555min814'] > -0.2]

zar_matches.to_csv(photdir+'zaritsky_15arcsec_555_814.csv',index=False)


def plot_colmag(df,ax=None,**kwargs):
    if ax == None:
        ax = plt.gcf()
    ax.scatter(df['555min814'],df['mag_555'],**kwargs)



fig = plt.figure(figsize=(11,6))
ax1 = fig.add_subplot(1,5,(1,2))
plot_colmag(r_df,ax=ax1,s=0.8,alpha=0.5,c='royalblue')#,label='HST')
plot_colmag(zar_matches,ax=ax1,s=1,alpha=0.6,c='firebrick',marker='s')
ax1.invert_yaxis()
ax1.set_xlim(-0.3,4)
ax1.scatter(-10,20,c='royalblue',label='HST')
ax1.scatter(-10,20,c='firebrick',label='ZH04',marker='s')
ax1.set_ylabel('F555W [mag]',fontsize=12,labelpad=5)
ax1.set_xlabel('F555W - F814W [mag]',fontsize=12)
ax1.plot(zams_x,zams_y,c='k',label='ZAMS',lw=2)
ax1.xaxis.set_ticks_position('both')
ax1.yaxis.set_ticks_position('both')
ax1.set_ylim(27.5,11.5)
ax1.tick_params(direction='in', which='both', labelsize=10)
ax1.legend(fontsize=12)

ax2 = fig.add_subplot(1,5,(3,5))#,projection=wcs.WCS(ne_8[0].header))
ax2.scatter(zar_matches['RAJ2000'],zar_matches['DEJ2000'],s=1.25,alpha=0.9,c='firebrick',zorder=2,marker='s')#,
            #transform=ax2.get_transform('fk5'),label='ZH04')
ax2.scatter(r_df['ra_814'],r_df['dec_814'],c='royalblue',s=0.3,alpha=0.4)#,
           # transform=ax2.get_transform('fk5'),label='HST')
#ax2.legend()
ax2.set_xlabel('RA',fontsize=12)
ax2.set_ylabel('Dec',fontsize=12)
ax2.tick_params('x',labelrotation=0,labelsize=8)
ax2.tick_params('y',labelrotation=45,labelsize=7)
ax2.xaxis.set_ticks_position('both')
ax2.yaxis.set_ticks_position('both')
ax2.tick_params(direction='in', which='both',size=5)
ax2.invert_xaxis()

fig.tight_layout()
plt.savefig('cmd_hst_zh04.png',dpi=300)


#https://sedfitter.readthedocs.io/en/stable/data.html
# n = 3
'''zar_sedfit = pd.DataFrame({'name':np.arange(0,len(zar_matches)),
                           'ra':zar_matches['RAJ2000'], 'dec':zar_matches['DEJ2000'],
                        'flag814':np.repeat(1,len(zar_matches)),
                        'flag555':np.repeat(1,len(zar_matches)),
                        'flag475':np.repeat(1,len(zar_matches))
                           })
zar_sedfit[zar_matches.columns[-6::]] = zar_matches[zar_matches.columns[-6::]]

np.savetxt(photdir+'zaritsky_sedfitter.txt',zar_sedfit.values)'''

#zar_pd = pd.read_csv(photdir+'zaritsky_555_814.csv')

#zar_sed = np.loadtxt(photdir+'zaritsky_sedfitter.txt')
#np.savetxt(photdir+'zaritsky_sedfitter2.txt',zar_sed.values,
          # fmt = '%.18e %.18e %.18e %i %i %i %.18e %.18e %.18e %.18e %.18e %.18e',)
#           fmt =' '.join([ '%.18e']*3 + ['%i']*3 + ['%.18e']*6))

          # '%.18e'*3 + '%i'*3 + '%.18e'*6)

#zar_sed2 = np.genfromtxt(photdir+'zaritsky_sedfitter2.txt',dtype=None)
'''zar_sed = pd.DataFrame(zar_sed)
zar_sed[3] = zar_sed[3].astype('int')
zar_sed[4] = zar_sed[4].astype('int')
zar_sed[5] = zar_sed[4].astype('int')
zar_sed[:,3] = [int(zar_sed[:,3][i]) for i in range(len(zar_sed))]
zar_sed[:,4] = zar_sed[:,4].astype('int')
zar_sed[:,5] = zar_sed[:,5].astype('int')'''
