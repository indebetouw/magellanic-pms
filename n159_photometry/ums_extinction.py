


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

zar_matches = pd.read_csv(photdir+'zaritsky_15arcsec_555_814.csv')

med_slopes = {'n159e':2.764,'n159w':2.574,'n159s':2.438,'all':2.53}


from astropy.io import ascii

#isos = ascii.read('/Users/toneill/Desktop/cmds/output971128150195.dat')

'''isos = ascii.read('/Users/toneill/N159/isochrones/output38030050070.dat')
ages = np.unique(isos['logAge'])
isos = isos[isos['logAge'] == 8]
zams_x = isos['F555Wmag'] - isos['F814Wmag']
zams_y = isos['F555Wmag']'''

z_5min8 = zar_matches['555min814'].values
z_555 = zar_matches['mag_555'].values


isos = pd.read_csv('/Users/toneill/N159/isochrones/n159_v2_TA-DA_syn_phot.txt',
                        skiprows=18,sep='\s+')#, sep='\s+')
ages = np.unique(isos['logage(yr)'])
isos = isos[isos['logage(yr)'].values==np.max(ages)]
zams_x = (isos['acswfc_f555w'] - isos['acswfc_f814w']).values #- 1.8
zams_y = isos['acswfc_f555w'].values #+ 18.48
# isos.columns




zams_l0 = zams_x < 0

polparams = np.polyfit(zams_x[zams_l0],zams_y[zams_l0],4)
poly = np.poly1d(polparams)
xinterp = np.linspace(-0.4,zams_x[-1],100)
zams_extend = poly(xinterp)

zams_x = np.append(zams_x, xinterp[zams_extend >= 8][::-1])
zams_y = np.append(zams_y, zams_extend[zams_extend >= 8][::-1])

isntnan = np.isfinite(zams_x)
zams_x = zams_x[isntnan]
zams_y = zams_y[isntnan]

plt.figure()
plt.scatter(z_5min8,z_555,s=1,alpha=0.8,c='k')#c=zar['Flag'],cmap='tab20c')#'royalblue')#,c=z_i)
#plt.scatter(r_df['555min814'],r_df['mag_555'],s=0.5,alpha=0.4,c='r')#c=zar['Flag'],cmap='tab20c')#'royalblue')#,c=z_i)
#plt.scatter(h_ums_x,h_ums_y,s=1,c='r')
plt.gca().invert_yaxis()
plt.plot(zams_x,zams_y,c='k')
#plt.colorbar()
plt.title('z&h')
#plt.plot(xinterp,poly(xinterp),c='b')


###########################################

def ransac_intercept(X, y):
    ransac = linear_model.RANSACRegressor(max_trials=1000,
                                          base_estimator=linear_model.LinearRegression(fit_intercept=True))
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    m, c = float(ransac.estimator_.coef_), float(ransac.estimator_.intercept_)

    return [m, c, inlier_mask * 1]

right_zams = np.full(len(z_5min8),False)
for i in range(len(z_5min8)):
    zix = z_5min8[i]
    ziy = z_555[i]
    d_zams = np.sqrt( (zams_x - zix)**2 + (zams_y - ziy)**2)
    close_zams = np.argmin(d_zams)
    if zix >= zams_x[close_zams]:
        right_zams[i] = True


'''xy_cut = [0.875, 1.5, 18.46,20.8]
rc_inds = (z_555 <= xy_cut[3]) & (z_555 >= xy_cut[2]) & (z_5min8 <= xy_cut[1]) & (z_5min8 >= xy_cut[0])  # & line_side
rc_df = zar_matches[rc_inds]

nrun = 1000
ran_mc = np.vstack([ransac_intercept(rc_df['555min814'].values.reshape(-1, 1) ,
                                     rc_df['mag_555'].values) for i in range(nrun)])
ran_m = ran_mc[:, 0]
ran_c = ran_mc[:, 1]
ran_in = ran_mc[:, 2]
raP_in = np.sum(ran_in, axis=0) / nrun

in_rc = raP_in >= 0.5

plt.scatter(rc_df['555min814'],rc_df['mag_555'],c=raP_in)

plt.scatter(rc_df['555min814'][in_rc],rc_df['mag_555'][in_rc],c='r')

in_rc_full = np.full(len(zar_matches),True)
in_rc_full[rc_df.index] = ~in_rc'''


left_rc = zar_matches['555min814'] <= 0.85#0.875# np.min(rc_df['555min814'])
zh_use = right_zams & left_rc# in_rc_full


zar_use = zar_matches[zh_use]
z_ums = zh_use
umsbool = z_ums
#################################

plt.figure()
plt.scatter(z_5min8,z_555,s=0.5,alpha=0.4,c='k')#c=zar['Flag'],cmap='tab20c')#'royalblue')#,c=z_i)
#plt.scatter(r_df['555min814'],r_df['mag_555'],s=0.5,alpha=0.4,c='r')#c=zar['Flag'],cmap='tab20c')#'royalblue')#,c=z_i)
#plt.scatter(h_ums_x,h_ums_y,s=1,c='r')
plt.gca().invert_yaxis()
plt.plot(zams_x,zams_y,c='k')
#plt.colorbar()
plt.title('z&h')
plt.scatter(z_5min8[umsbool],z_555[umsbool],s=1,alpha=1)#c=zar['Flag'],cmap='tab20c')#'royalblue')#,c=z_i)


############################################################
ums_ra_full = zar_use['RAJ2000'].values
ums_dec_full = zar_use['DEJ2000'].values

av0_list = []

for k in range(4):

    region =list(region_dicts.keys())[k]

    r_df = region_dicts[region]

    zar_coords = SkyCoord(ra=ums_ra_full * u.deg, dec=ums_dec_full * u.deg)
    hst_coords = SkyCoord(ra=r_df['ra_814'] * u.deg, dec=r_df['dec_814'] * u.deg)

    max_sep = 15 * u.arcsec
    idx, d2d, d3d = zar_coords.match_to_catalog_sky(hst_coords)
    sep_constraint = d2d < max_sep
    print(np.sum(sep_constraint))
    r_zar = zar_use[sep_constraint]

    slope_rv = med_slopes[region]

    ums_x = r_zar['555min814'].values
    ums_y = r_zar['mag_555'].values

    xrange = np.linspace(-1.8, -0.2, 100)

    dered_umsx = []
    dered_umsy = []
    ums_dzams = []

    for i in range(len(ums_x)):
        print(100 * i / len(ums_x))
        xi = ums_x[i]
        yi = ums_y[i]
        y_rv = slope_rv * (xrange - xi) + yi
        d_zams_allj = []
        for j in range(len(xrange)):
            d_zams = np.sqrt((zams_x - xrange[j]) ** 2 + (zams_y - y_rv[j]) ** 2)
            d_zams_allj.append(np.min(d_zams))
        nearj = np.argmin(d_zams_allj)
        zams_closest = np.argmin(np.sqrt((zams_x - xrange[nearj]) ** 2 + (zams_y - y_rv[nearj]) ** 2))
        near_zamsj = np.argmin(d_zams)
        d_zams = np.sqrt((xi - zams_x[zams_closest]) ** 2 + (yi - zams_y[zams_closest]) ** 2)
        corr_x = xi - zams_x[zams_closest]
        corr_y = yi - zams_y[zams_closest]
        ums_dzams.append(d_zams)
        dered_umsx.append(zams_x[zams_closest])
        dered_umsy.append(zams_y[zams_closest])

    ums_dzams = np.array(ums_dzams)
    ums_ra = r_zar['RAJ2000'].values
    ums_dec = r_zar['DEJ2000'].values

    h_ra = r_df['ra_814'].values
    h_dec = r_df['dec_814'].values

    nnear = 10
    eps = 10. / 3600

    UAV = ums_dzams
    # separate dfs into nx2 arrays for KNN
    UCoords = [[ums_ra[i], ums_dec[i]] for i in range(len(ums_ra))]
    TCoords = np.array([[h_ra[i], h_dec[i]] for i in range(len(h_dec))])
    av0 = kNN_regr_Av(UCoords, TCoords, UAV, eps, nnear, ncore=7)

    av0_list.append(av0)

####################################


from skimage import measure
alma_co = fits.open('/Users/toneill/N159/alma/12CO_combined.regrid.gtr0.2K.maximum.fits')
alma_cont = np.where(alma_co[0].data >= -100000, 0, 1)
contours = measure.find_contours(alma_cont, 0.9)


fig = plt.figure(figsize=(9, 9))
ax1 = fig.add_subplot(221, projection=wcs.WCS(alma_co[0].header))
ax2 = fig.add_subplot(222, projection=wcs.WCS(alma_co[0].header))
ax3 = fig.add_subplot(223, projection=wcs.WCS(alma_co[0].header))
ax4 = fig.add_subplot(224, projection=wcs.WCS(alma_co[0].header))
axs = [ax1, ax2, ax3, ax4]

 for k in range(4):
    region = list(region_dicts.keys())[k]

    r_df = region_dicts[region]
    h_ra = r_df['ra_814'].values
    h_dec = r_df['dec_814'].values

    r_avs = av0_list[k]

    ax = axs[k]

    ax.contour(alma_co[0].data, levels=[5], colors=['grey'],  zorder=3)
    sss = ax.scatter(h_ra, h_dec, c=r_avs, cmap=cmr.ember_r, s=15, alpha=0.5, transform=ax.get_transform('fk5'), zorder=0,vmin=0.65,vmax=2.65)
    # ax.scatter(ums_ra, ums_dec, c='c', marker='x',s=1, transform=ax.get_transform('fk5'),zorder=2)
    ax.set_title(f'{region.upper()}')
    #ax.set_xlim(115, 330)
    #ax.set_ylim(-99, 255)

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], c='k',  lw=1)

ax1.set_xlim(120,250)
ax1.set_ylim(90,220)

ax2.set_xlim(195,325)
ax2.set_ylim(70,210)

ax3.set_xlim(115,255)
ax3.set_ylim(-85, 55)

ax4.set_xlim(115,335)
ax4.set_ylim(-80,225)

[ax.set_xlabel('RA') for ax in [ax3,ax4]]
[ax.set_ylabel('Dec',labelpad=0) for ax in [ax1,ax3]]
[ax.set_xlabel('a ',fontsize=0) for ax in [ax1,ax2]]
[ax.set_ylabel(' ') for ax in [ax2,ax4]]

[ax.tick_params('y',labelrotation=45,labelsize=7) for ax in [ax1,ax2,ax3,ax4]]
[ax.tick_params('x',labelsize=7) for ax in [ax1,ax2,ax3,ax4]]

fig.tight_layout()

fig.subplots_adjust(right=0.9,bottom=0.05,left=0.07,wspace=0.14)

sss = ax4.scatter(h_ra[-1], h_dec[-1], c=r_avs[-1], cmap=cmr.ember_r,
                  s=10, alpha=1, transform=ax4.get_transform('fk5'), zorder=0,
                 vmin=0.65, vmax=2.65)

cbar_ax = fig.add_axes([0.915, 0.05, 0.02, 0.91])
cbar = fig.colorbar(sss, cax=cbar_ax)
cbar.set_label(label='$A_{555}$ [mag]',fontsize=13,labelpad=3)
cbar.ax.tick_params(labelsize=8)

fig.tight_layout()
fig.subplots_adjust(right=0.9,bottom=0.05,left=0.07,wspace=0.14)

ax1.set_title('N159W')
ax2.set_title('N159E')
plt.savefig('red_maps.png',dpi=300)

#########################


import seaborn as sns
#sns.set_palette('Set2')
sns.set_palette('OrRd_r')

region_names = list(region_dicts.keys())

cols_kdes = ['darkviolet','hotpink','red','blue']
alphas = [0.2, 0.4, 0.15]
zorders = [0, 3, 2]

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
[sns.kdeplot(av0_list[i],label=f'{region_names[i].upper()}: Median = {np.median(av0_list[i]):.3}',
        fill=True,linewidth=2,alpha=alphas[i],zorder=zorders[i],bw_adjust=1.5,color=cols_kdes[i]) for i in range(len(av0_list)-1)]
#[plt.axvline(x=np.median(av0_list[i]),ls='--',lw=1,c=cols_kdes[i]) for i in range(len(av0_list)-1)]
plt.legend(loc='upper left',fontsize=12,framealpha=1)
plt.xlabel('$A_{555}$ [mag]',fontsize=13,labelpad=10)
plt.ylabel('Probability density',fontsize=13,labelpad=10)
#plt.title('KDEs of RANSAC Slopes')
plt.tight_layout()
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(direction='in', which='both', labelsize=12)
#plt.xlim(2.3,2.9)

plt.savefig('a555_kdes.png',dpi=300)

##### NEED TO FIX THIS FOR F814W
# de Marchi 2016 extinction law
# https://academic.oup.com/mnras/article/455/4/4373/1264525
R_BV = [4.48, 3.03, 1.83, 1.22]  # at V,I,J,H
# R_BV = [4.48, 3.74, 1.83, 1.22] # at V,R,J,H
label = ['f555w', 'f814w', 'f110w', 'f160w']

##############################################
import copy

fnames = ['n159-e_reduce.phot.cutblue', 'n159-w_reduce.phot.cutblue',
          'n159-s_reduce.phot.cutblue',  'n159-all_reduce.phot.cutblue' ]
for k in range(4):
    region = list(region_dicts.keys())[k]
    r_df = region_dicts[region]
    r_avs = av0_list[k]

    trim_cat = copy.deepcopy(r_df)
    trim_cat['AvNN'] = r_avs
    trim_cat['mag_555_dered'] = trim_cat['mag_555'] - r_avs * R_BV[0] / R_BV[0]
    trim_cat['mag_814_dered'] = trim_cat['mag_814'] - r_avs * R_BV[1] / R_BV[0]

    trim_cat.to_csv(photdir + fnames[k] +'.dered.csv', index=False)


dered_e = pd.read_csv(photdir + fnames[0] +'.dered.csv')
dered_w = pd.read_csv(photdir + fnames[1] +'.dered.csv')

plt.figure()
plt.scatter(z_5min8,z_555,s=0.1,alpha=0.4,c='k',label='all')
#plt.plot(line_X + exp_VminI, line_Y + exp_mV, color="royalblue", linewidth=2, ls='--',
#        label=f"Median RANSAC")
sss = plt.scatter(ums_x,ums_y,label='ums',c=ums_dzams ,cmap=cmr.ember_r)#'inferno_r')
plt.colorbar(sss, label='Distance to ZAMS along Rv [mag]')
plt.gca().invert_yaxis()
plt.plot(zams_x,zams_y,c='k',label='zams (20 myr)')
plt.xlabel('v-i')
plt.ylabel('v')
plt.legend()
plt.title(f'{region}')


plt.figure()
#plt.scatter(r_df['ra_555'], r_df['dec_555'], c='grey', s=0.1, alpha=0.5)
#plt.scatter(z_ra,z_dec,s=0.3)
sss = plt.scatter(ums_ra,ums_dec,s=10, c=ums_dzams, cmap=cmr.ember_r)#'plasma_r')#,vmin=0,vmax=3)#,  s=30, alpha=0.5)
#plt.scatter(h_ums_df['ra_555'],h_ums_df['dec_555'],s=10)
plt.colorbar(sss, label='Distance to ZAMS along Rv [mag]')
plt.gca().invert_xaxis()
plt.xlabel('RA')
plt.ylabel('dec')
plt.title('Z&H')



h_ra = r_df['ra_814'].values
h_dec = r_df['dec_814'].values

# weighting params for KNN from Ksoll 2018
nnear = 10
eps = 10. / 3600

UAV = ums_dzams

# separate dfs into nx2 arrays for KNN
# TO DO: improve how done for TCoords - takes forever since list
UCoords = [[ums_ra[i], ums_dec[i]] for i in range(len(ums_ra))]
TCoords = np.array([[h_ra[i], h_dec[i]] for i in range(len(h_dec))])
# start = timer()
av0 = kNN_regr_Av(UCoords, TCoords, UAV, eps, nnear, ncore=7)


alma_co = fits.open('/Users/toneill/N159/alma/12CO_combined.regrid.gtr0.2K.maximum.fits')

fig = plt.figure()
ax = fig.add_subplot(111, projection=wcs.WCS(alma_co[0].header))
#ax.imshow(alma_co[0].data,origin='lower',vmin=5)
ax.contour(alma_co[0].data, levels=[5],colors=['c'], label='CO',zorder=3)
sss = ax.scatter(h_ra,h_dec,c=av0,cmap=cmr.ember_r,s=3,alpha=0.5,transform=ax.get_transform('fk5'),zorder=0)
#ax.scatter(ums_ra, ums_dec, c='c', marker='x',s=1, transform=ax.get_transform('fk5'),zorder=2)
plt.colorbar(sss,label='Inferred Av')
ax.set_title(f'{region}')
ax.set_xlim(115,330)
ax.set_ylim(-99,255)

from skimage import measure
alma_co = fits.open('/Users/toneill/N159/alma/12CO_combined.regrid.gtr0.2K.maximum.fits')
alma_cont = np.where(alma_co[0].data >= -100000, 0, 1)
contours = measure.find_contours(alma_cont, 0.9)
for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], c='k',
            lw=0.5)  # ,ls=':')

import seaborn as sns
plt.figure()
sns.kdeplot(av0,fill=True,color='darkviolet',label='HST')
sns.kdeplot(ums_dzams,fill=True,color='firebrick',label='Z&H UMS')
plt.xlabel('Inferred Av [mag]')
plt.legend()
plt.title('all, nn=10')
plt.tight_layout()

#############################################

##### NEED TO FIX THIS FOR F814W
# de Marchi 2016 extinction law
# https://academic.oup.com/mnras/article/455/4/4373/1264525
R_BV = [4.48, 3.03, 1.83, 1.22] # at V,I,J,H
#R_BV = [4.48, 3.74, 1.83, 1.22] # at V,R,J,H
label = ['f555w','f814w', 'f110w', 'f160w']

import copy

trim_cat = copy.deepcopy(r_df)
trim_cat['AvNN'] = av0
trim_cat['mag_555_dered'] = trim_cat['mag_555'] - av0*R_BV[0]/R_BV[0]
trim_cat['mag_814_dered'] = trim_cat['mag_814'] - av0*R_BV[1]/R_BV[0]

trim_cat.to_csv(photdir+'dered_hstphot_all.csv',index=False)

