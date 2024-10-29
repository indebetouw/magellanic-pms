
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cmasher as cmr
from astropy import units as u
from astropy.coordinates import SkyCoord
from sklearn import linear_model
import seaborn as sns

####################################################
savephotdir = '/Users/toneill/N159/photometry/reduced/'
#e_vi = pd.read_csv(savephotdir+'n159-e_reduce.phot.cutblue.csv')
#w_vi = pd.read_csv(savephotdir+'n159-w_reduce.phot.cutblue.csv')
#s_vi = pd.read_csv(savephotdir+'n159-s_reduce.phot.cutblue.csv')
#all_vi = pd.read_csv(savephotdir+'n159-all_reduce.phot.cutblue.csv')

from n159_photometry.load_data import load_phot

fuse = 'vis.ir'
all_vi = load_phot(region='n159-all',fuse=fuse)
e_vi = load_phot(region='n159-e',fuse=fuse)
w_vi = load_phot(region='n159-w',fuse=fuse)
s_vi = load_phot(region='n159-s',fuse=fuse)

region_dicts = {'n159e':e_vi,'n159w':w_vi,'n159s':s_vi,'all':all_vi}

####################################################
# define theoretical RC position
## https://openaccess.inaf.it/bitstream/20.500.12386/32297/1/Nataf_2021_ApJ_910_121.pdf
## table 3 and conclusions

exp_MI = -0.39
mu_LMC = 18.48
exp_mI = mu_LMC + exp_MI
exp_VminI = 0.93
# (V-I) = 0.93 = expMV - expMI
# expMV = (V-I) + expMI
    # = 0.93 - 0.39 = 0.54
exp_MV = exp_VminI + exp_MI
exp_mV = mu_LMC + exp_MV

def convert_bvri_hst(m_bvri,c_bvri,c0=None,c1=None,c2=None,zeropt=0):
    #https://arxiv.org/pdf/astro-ph/0507614.pdf
    #https://iopscience.iop.org/article/10.1086/444553/pdf
    return -(c0 + c1 * c_bvri + c2 * c_bvri**2) + m_bvri + zeropt

exp_555 =convert_bvri_hst(exp_mV,exp_VminI, c0=25.719, c1=-0.088, c2=0.043,zeropt=25.724)
exp_814 = convert_bvri_hst(exp_mI,exp_VminI, c0=25.489, c1=0.041, c2=-0.093,zeropt=25.501)
exp_555min814 = exp_555 - exp_814




#################################

v_mmax = 22
b_ymin = 18.5
c_xmin = 0.85
c_xmax = 2.5

bounds = [c_xmin,c_xmax,b_ymin,v_mmax]

xyb_dict = {'n159e': [0.5, 2.5, 19.2, 24], 'n159w': [0.5, 2.5, 19.2, 24],
            'n159s': [0.5, 2.5, 19.2, 24], 'all': [0.5, 2.5, 19.2, 24]}
xycut_dict = {'n159e': [0.8, 3, 19, v_mmax], 'n159w': [0.8, 3, 19, v_mmax], 'n159s': [0.8, 3, 19, v_mmax], 'all': [0.8, 3, 19, v_mmax]}

#xyb_dict = {'n159e': bounds, 'n159w': bounds,
#            'n159s':bounds, 'all': bounds}
xycut_dict = {'n159e':bounds, 'n159w':bounds, 'n159s': bounds, 'all':bounds}


def ransac_nointercept(X, y):
    ransac = linear_model.RANSACRegressor(max_trials=1000,
                                          estimator=linear_model.LinearRegression(fit_intercept=False))
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    m, c = float(ransac.estimator_.coef_[0]), float(ransac.estimator_.intercept_)

    return [m, c],[ inlier_mask * 1]

cmap_use = cmr.get_sub_cmap('inferno', 0, 0.9)

nrun = 1000

fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2,figsize=(9,9),sharey=True,sharex=True)
axs = [ax2,ax1,ax3,ax4]
saveslopes = []

for i in range(4):
    region = list(region_dicts.keys())[i]
    #region = 'n159e'
    r_df = region_dicts[region]
    xs = r_df['555min814']
    ys = r_df['mag_555']

    xy_bounds = xyb_dict[region]
    d_rcline = (xs - xy_bounds[0]) * (xy_bounds[3]-xy_bounds[2]) - (ys-xy_bounds[2])*(xy_bounds[1]-xy_bounds[0])
    line_side = d_rcline >= 0

    xy_cut = xycut_dict[region]
    rc_inds = (ys <= xy_cut[3]) & (ys >= xy_cut[2]) & (xs <= xy_cut[1]) & (xs >= xy_cut[0]) #& line_side
    rc_df = r_df[rc_inds]

    X0 = rc_df['555min814'].values.reshape(-1, 1) - exp_555min814
    y0 = rc_df['mag_555'].values - exp_555

    ran_mc = [ransac_nointercept(X0, y0) for i in range(nrun)]
    ran_mc_params =np.array( [ran_mc[i][0] for i in range(nrun)])
    ran_mc_inlist = np.array([ran_mc[i][1] for i in range(nrun)])
    ran_m = ran_mc_params[:, 0]
    ran_c = ran_mc_params[:, 1]
    ran_in = ran_mc_inlist
    P_in = np.sum(ran_in, axis=0) / nrun
    saveslopes.append(ran_m)

    line_X = np.arange(-0.2, 1.45, 0.05)#X0.min(), X0.max() - 0.5, 0.05)  # [:, np.newaxis]
    line_Y = np.median(ran_m) * line_X + np.median(ran_c)
    #rc_pcheck = P_in >= 0.9

    lineX_plot = line_X + exp_VminI
    lineY_plot = line_Y + exp_mV
    partialplot = True
    if partialplot:

        ax = axs[i]
        ax.scatter(exp_VminI, exp_mV, edgecolor='firebrick', facecolor='None',marker='8', s=100, zorder=3, alpha=0.8)#label='Expected RC Center',
        scats = ax.scatter(rc_df['555min814'].values, rc_df['mag_555'].values, c=P_in, cmap=cmap_use, vmin=0, vmax=1, s=0.8,
                           marker='o')
        ax.scatter(xs, ys, c='grey', zorder=0, s=0.3, alpha=0.5)

        ax.plot(lineX_plot, lineY_plot, color="firebrick", linewidth=2, ls='-',
                 label=f"RANSAC Slope = {np.median(ran_m):.2f}",zorder=5)


        arrprop = dict(arrowstyle="-|>,head_width=0.4,head_length=0.8",
                    shrinkA=0, shrinkB=0,color='firebrick',lw=2)

        ax.annotate("", xy=(lineX_plot[-1],lineY_plot[-1]),
                                    xytext=(lineX_plot[0],lineY_plot[0]), arrowprops=arrprop,
                    label=f"RANSAC Slope = {np.median(ran_m):.2f}", zorder=6        )
        ax.legend(loc='upper right',fontsize=8)

ax1.invert_yaxis()
ax1.set_xlim(-0.3,3.5)

region_names = list(region_dicts.keys())
for i in range(4):
    ax = axs[i]
    ax.set_title(region_names[i].upper())
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(direction='in', which='both', labelsize=9)

[ax.set_xlabel('F555W - F814W',fontsize=12) for ax in [ax3,ax4]]
[ax.set_ylabel('F555W',fontsize=12) for ax in [ax1,ax3]]

fig.tight_layout()

fig.subplots_adjust(right=0.9)

cbar_ax = fig.add_axes([0.915, 0.05, 0.025, 0.91])
cbar = fig.colorbar(scats, cax=cbar_ax)
cbar.set_label(label='P( Red Clump )',fontsize=10,labelpad=5)
cbar.ax.tick_params(labelsize=8)

plt.savefig('ransac_rc.png',dpi=300)

##################################################################

import seaborn as sns
#sns.set_palette('Set2')
sns.set_palette('OrRd_r')

cols_kdes = ['darkviolet','hotpink','red']

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
[sns.kdeplot(saveslopes[i],label=f'{region_names[i].upper()}: Median = {np.median(saveslopes[i]):.3}',
        fill=True,linewidth=1.75,bw_adjust=2,color=cols_kdes[i]) for i in range(len(saveslopes)-1)]
[plt.axvline(x=np.median(saveslopes[i]),ls='--',lw=1,c=cols_kdes[i]) for i in range(len(saveslopes)-1)]
plt.legend(loc='upper right',fontsize=12,framealpha=1)
plt.xlabel('RANSAC Slope',fontsize=13,labelpad=10)
plt.ylabel('Probability density',fontsize=13,labelpad=10)
#plt.title('KDEs of RANSAC Slopes')
plt.tight_layout()
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(direction='in', which='both', labelsize=12)
plt.xlim(2.3,2.9)

plt.savefig('ransac_kdes.png',dpi=300)


med_slopes = [np.round(np.median(saveslopes[i]),3) for i in range(len(saveslopes))]
print(med_slopes)
#[2.696, 2.526, 2.471, 2.543]

### OLD
###med_slopes = {'n159e':2.764,'n159w':2.574,'n159s':2.438,'all':2.53}


##################################################################
    normplot = False
    if normplot:

        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(121)
        plt.scatter(exp_VminI, exp_mV, c='r', marker='P', s=100, label='Expected RC Center', zorder=3, alpha=0.5)
        scats = ax.scatter(rc_df['555min814'].values, rc_df['mag_555'].values, c=P_in, cmap='RdYlBu_r', vmin=0, vmax=1, s=0.8,
                           marker='o', label='Red Clump Candidates')
        ax.scatter(xs, ys, c='grey', zorder=0, s=1, alpha=0.5)
        fig.colorbar(scats, ax=ax, label='P(RC)')
        plt.plot(line_X + exp_VminI, line_Y + exp_mV, color="k", linewidth=2, ls='-',
                 label=f"Median RANSAC, slope = {np.median(ran_m):.2f}")
        plt.legend(loc='lower left',fontsize=13)
        plt.gca().invert_yaxis()
        plt.xlabel('F555 - F814')
        plt.ylabel('F555')
        plt.xlim(-0.5, 3.5)
        plt.ylim(27.5, 18)
        plt.title(f'{region} Transformed RANSAC Regression')

        ax1 = fig.add_subplot(222)
        ax1.hist(ran_m, bins=15, facecolor='crimson',alpha=0.5)
        ax1.set_xlabel('Slope')
        ax1.set_title('Slope estimates')
        ax1.axvline(x=np.median(ran_m), c='k', lw=1, \
                    label='Slope =  %.2f' % np.median(ran_m) + ' $\pm$ %.2f' % np.std(ran_m))
        ax1.legend(loc='upper left')
        ax1.set_xlim(2.15,3.15)

        ax2 = fig.add_subplot(224)
        plt.scatter(0, 0, c='r', marker='P', s=100, label='Expected RC Center', zorder=3, alpha=0.5)
        scats = ax2.scatter(X0, y0, c=P_in, cmap='RdYlBu_r', vmin=0, vmax=1, s=3, marker='o', label='Red Clump Candidates')
        # scats = ax.scatter(X[rc_pcheck],y[rc_pcheck],c='r',s=6,marker='o',label='Red Clump Samples')
        #fig.colorbar(scats, ax=ax2, label='P(RC)')
        plt.plot(line_X, line_Y, color="k", linewidth=2, ls='-', label="Mean RANSAC")
        plt.legend()
        plt.axvline(x=0,c='grey',ls='--',lw=0.5)
        plt.axhline(y=0,c='grey',ls='--',lw=0.5)
        plt.xlabel('(F555 - F814) - (V-I)$_{RC,0,LMC}$')
        plt.ylabel('(F555) - V$_{RC,0,LMC}$')
        plt.xlim(-0.5, 2.5)
        plt.ylim(4.5, -0.5)

        fig.tight_layout()

        #plt.savefig('/Users/toneill/N159/Plots/red/'+region+'_transformed_ransac_rectangle_max22.png',dpi=300)

    rc_cand = rc_df[P_in >= 0.5]

    line_X = np.arange(X0.min()-0.5, X0.max() - 0.5, 0.00001)  # [:, np.newaxis]
    line_Y = np.median(ran_m) * line_X + np.median(ran_c)
    line_Yperp = -1/np.median(ran_m) * line_X

    exp_line_X = line_X + exp_VminI
    exp_line_Yperp = line_Yperp + exp_mV

    rc_xs = rc_cand['555min814'].values
    rc_ys = rc_cand['mag_555'].values
    rc_dexp = np.sqrt((rc_xs-exp_VminI)**2 + (rc_ys-exp_mV)**2) # mag

    rc_dalongperp = []
    for i in range(len(rc_xs)):
        d_pperpline = np.sqrt((rc_xs[i]-exp_line_X)**2 + (rc_ys[i]-exp_line_Yperp)**2)
        closest_perp = np.argmin(d_pperpline)
        #plt.scatter(rc_xs[i],rc_ys[i])
        #plt.scatter(exp_line_X[closest_perp],exp_line_Yperp[closest_perp])
        slope_closeperp = (exp_line_Yperp[closest_perp]-rc_ys[i])/(exp_line_X[closest_perp]-rc_xs[i])
        #plt.plot([exp_line_X[closest_perp],rc_xs[i]],[exp_line_Yperp[closest_perp],rc_ys[i]])
        d_alongperp = np.sqrt((exp_line_Yperp[closest_perp]-rc_ys[i])**2 + (exp_line_X[closest_perp]-rc_xs[i])**2)
        rc_dalongperp.append(d_alongperp)

    rc_cand['dperp'] = rc_dalongperp

    col_exx = rc_xs -exp_VminI
    rc_Av = rc_cand['dperp'] * col_exx
    plt.figure()
    plt.hist(rc_Av)


    plt.figure()
    plt.scatter(rc_dexp,rc_dalongperp)
    plt.plot([0,3.2],[0,3.2],c='k')

    import seaborn as sns
    sns.kdeplot(rc_dalongperp,fill=True,cut=0,bw_adjust=0.8)
    plt.xlabel('Distance to red clump along Rv')
    plt.title('n159e')

    plt.figure()
    plt.scatter(P_in[P_in >= 0.5],rc_dalongperp)
    plt.xlabel('Probability RC')
    plt.ylabel('Dist along Rv')
    # create fig
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #twoD_kde(rc_cand['555min814'],rc_cand['mag_555'], ax=ax, fig=fig, discrete=True)
    ax.invert_yaxis()
    #ax.scatter(exp_VminI, exp_mV, c='r', marker='P', s=200, label='Expected RC Center', zorder=3, alpha=0.5)
    ax.plot(line_X + exp_VminI, line_Y + exp_mV, color="royalblue", linewidth=2, ls='--',
             label=f"Median RANSAC, slope = {np.median(ran_m):.2f}")
    ax.plot(line_X + exp_VminI, line_Yperp + exp_mV, color="r", linewidth=2, ls='--')
    sss = ax.scatter(rc_cand['555min814'],rc_cand['mag_555'],c=rc_dalongperp,s=2,alpha=1,cmap='inferno')
    plt.colorbar(sss,label='Distance to red clump along Rv [mag]')
    #plt.plot([0,23],[23,0],c='k',ls=':')
    plt.title('n159e')

    plt.figure()
    plt.scatter(r_df['ra_555'],r_df['dec_555'],c='grey',s=0.1,alpha=0.5)
    sss = plt.scatter(rc_cand['ra_555'],rc_cand['dec_555'],c=rc_dalongperp,
                      cmap='inferno_r',edgecolor='None',s=50,alpha=0.5)
    plt.colorbar(sss,label='Distance to red clump along Rv [mag]')
    #plt.plot([0,23],[23,0],c='k',ls=':')
    plt.title('n159e')

    from astropy.io import ascii
    isos = ascii.read('output971128150195.dat')

    #isos.columns

    plt.plot(isos['Vmag']-isos['Imag'],isos['Vmag']+mu_LMC)
    plt.gca().invert_yaxis()


    ras = r_df['ra_555']
    decs = r_df['dec_555']

    ########## use NGC 602 isochrones for now
    # used mu = 18.91
    pisa_phot = pd.read_csv('/Users/toneill/NGC602/Isochrones/tadaAV025.dist18.91.txt',
                            skiprows=18, sep='\s+')
    # delim_whitespace=True)
    pisa_phot['555min814'] = pisa_phot['acswfc_f555w'] - pisa_phot['acswfc_f814w']
    # pisa_phot = pisa_phot.dropna().reset_index(drop=True)
    # divide by age of isochrone track to make colors
    unique_ages = np.unique(pisa_phot['log_age_yr'])
    pisa_zams = pisa_phot[pisa_phot['log_age_yr'] == np.min(unique_ages)]


    zams_x = pisa_zams['555min814']-0.9
    zams_y = pisa_zams['acswfc_f555w']-18.91+mu_LMC

    fake_ums = (xs < 0.9) & (xs>0) & (ys<21)

    plt.figure()
    plt.plot(zams_x,zams_y,c='r',lw=2)
    plt.scatter(r_df['555min814'],r_df['mag_555'],c='grey',s=0.5)
    plt.plot(line_X + exp_VminI, line_Y + exp_mV, color="royalblue", linewidth=2, ls='--')
    plt.gca().invert_yaxis()
    plt.title('fake ZAMS and UMS')
    plt.scatter(xs[fake_ums],ys[fake_ums],c='g')


    slope_rv = np.median(ran_m)

    ums_df = r_df[fake_ums]

    ums_x = ums_df['555min814'].values
    ums_y = ums_df['mag_555'].values

    xrange = np.linspace(-0.5,1.,50)

    dered_umsx = []
    dered_umsy = []
    ums_dzams = []

    for i in range(len(ums_df)):
        print(100*i/len(ums_df))
        xi = ums_df['555min814'].values[i]
        yi = ums_df['mag_555'].values[i]
        y_rv = slope_rv*(xrange-xi) + yi
        d_zams_allj = []
        for j in range(len(xrange)):
            d_zams = np.sqrt((zams_x-xrange[j])**2 + (zams_y-y_rv[j])**2)
            d_zams_allj.append(np.min(d_zams))
        nearj = np.argmin(d_zams_allj)
        zams_closest = np.argmin(np.sqrt((zams_x - xrange[nearj]) ** 2 + (zams_y - y_rv[nearj]) ** 2))
        near_zamsj = np.argmin(d_zams)
        d_zams = np.sqrt((xi-zams_x[zams_closest])**2+(yi-zams_y[zams_closest])**2)
        corr_x = xi-zams_x[zams_closest]
        corr_y = yi-zams_y[zams_closest]
        ums_dzams.append(d_zams)
        dered_umsx.append(zams_x[zams_closest])
        dered_umsy.append(zams_y[zams_closest])



    plt.figure()
    plt.scatter(r_df['ra_555'], r_df['dec_555'], c='grey', s=0.1, alpha=0.5)
    sss = plt.scatter(ums_df['ra_555'], ums_df['dec_555'], c=ums_dzams,
                          cmap='Reds', edgecolor='None', s=500, alpha=0.5)
    plt.colorbar(sss, label='Distance to ZAMS along Rv [mag]')

    av0   = np.array([kNN_extinction(ums_df['ra_555'], ums_df['dec_555'],ums_dzams,eps,nnear,ri,di) \
                  for ri,di in zip(RA[z0],Dec[z0])])

    # weighting params for KNN from Ksoll 2018
    nnear = 20
    eps = 10. / 3600

    URA = ums_df['ra_555'].values
    UDec = ums_df['dec_555'].values
    UAV =ums_dzams

    RA = r_df['ra_555'].values
    Dec = r_df['dec_555'].values

    # separate dfs into nx2 arrays for KNN
    # TO DO: improve how done for TCoords - takes forever since list
    UCoords = [[URA[i], UDec[i]] for i in range(len(URA))]
    TCoords = np.array([[RA[i], Dec[i]] for i in range(len(RA))])
    # start = timer()
    av0 = kNN_regr_Av(UCoords, TCoords, UAV, eps, nnear, ncore=7)

    RCCoords = [[rc_cand['ra_555'].values[i], rc_cand['dec_555'].values[i]] for i in range(len(rc_cand))]
    av0_rc = kNN_regr_Av(RCCoords, TCoords, rc_dalongperp, eps, nnear, ncore=7)

    plt.figure()
    avscat = plt.scatter(RA,Dec,c=av0,cmap='inferno_r',s=2,alpha=0.5)
    plt.colorbar(avscat,label='NN Av')
    plt.gca().invert_xaxis()
    plt.xlabel('RA')
    plt.ylabel('Dec')
    plt.title('n159e crappy fake zams/ums NN Av')

    plt.figure()
    #avscat = plt.scatter(RA,Dec,c=av0,cmap='inferno_r',s=2,alpha=0.5)
    avscat = plt.scatter(rc_cand['ra_555'],rc_cand['dec_555'],c=rc_dalongperp,
                      cmap='inferno_r',edgecolor='None',s=10,alpha=0.5)
    plt.colorbar(avscat,label='NN Av')
    plt.gca().invert_xaxis()
    plt.xlabel('RA')
    plt.ylabel('Dec')
    plt.title('n159e red clump values')


    alma_co = fits.open('/Users/toneill/N159/alma/12CO_combined.regrid.gtr0.2K.maximum.fits')

    fig = plt.figure()q
    ax = fig.add_subplot(121,projection=wcs.WCS(alma_co[0].header))
    ax.contour(alma_co[0].data,cmap='Blues_r',label='CO')
    #ax.scatter(RA,Dec,c=av0,cmap='inferno_r',s=3,alpha=0.5,transform=ax.get_transform('fk5'))
    ax.scatter(URA,UDec,c='blue',marker='x',transform=ax.get_transform('fk5'))
    #ax.invert_yaxis()
    plt.title('n159e crappy fake zams/ums NN Av, CO contour')
    ax2 = fig.add_subplot(122,projection=wcs.WCS(alma_co[0].header))
    ax2.contour(alma_co[0].data,cmap='Blues_r',label='CO')
    ax2.scatter(rc_cand['ra_555'],rc_cand['dec_555'],c='r',marker='x',transform=ax2.get_transform('fk5'))
    #ax2.scatter(RA,Dec,c=av0_rc,cmap='inferno_r',s=3,alpha=0.5,transform=ax2.get_transform('fk5'))
    ax2.set_title('NN red clump')
    ax.invert_yaxis()
    ax2.invert_yaxis()
    '''plt.figure()
        plt.plot(zams_x, zams_y, c='r', lw=2,marker='s')
        plt.plot(xrange,y_rv,marker='o')
        plt.scatter(xi,yi,c='m')
        plt.gca().invert_yaxis()'''

        # y - y0 = m*(x-x0)
        # y = m*(x-x0) + y0
    #plt.scatter(ras[fake_ums],decs[fake_ums],c='g')


#########################################################a################################


sns.set_palette('Set2')

rnames = list(region_dicts.keys())

plt.figure(figsize=(8,6))
[sns.kdeplot(saveslopes[i],alpha=0.3,label=f'{rnames[i]}, median = {np.median(saveslopes[i]):.3}',
        fill=True,linewidth=1.5,bw_adjust=1.3) for i in range(len(saveslopes)-1)]
[plt.axvline(x=np.median(saveslopes[i]),c='k',ls='--',lw=0.5) for i in range(len(saveslopes)-1)]
plt.legend(loc='upper right')
plt.xlabel('Slope',fontsize=13)
plt.ylabel('Kernel density',fontsize=13)
plt.title('KDEs of RANSAC Slopes')
plt.tight_layout()
plt.savefig('/Users/toneill/N159/Plots/red/' + 'kdes_transformed_ransac_rectangle_max22.png', dpi=300)

from scipy import stats

list(region_dicts.keys())

## east vs west – east has significantly greater slopes than west
stats.ks_2samp(saveslopes[0],saveslopes[1],alternative='less')

## east vs south – east has significantly greater slopes than west
stats.ks_2samp(saveslopes[0],saveslopes[2],alternative='less')

## west vs south – west has significantly greater slopes than south
stats.ks_2samp(saveslopes[1],saveslopes[2],alternative='less')





