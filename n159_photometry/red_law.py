
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

def ransac_nointercept(X, y,fit_int=False):
    ransac = linear_model.RANSACRegressor(max_trials=1000,
                                          base_estimator=linear_model.LinearRegression(fit_intercept=fit_int))
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    m, c = float(ransac.estimator_.coef_), float(ransac.estimator_.intercept_)
    return [m, c, inlier_mask * 1]

cmap_use = cmr.get_sub_cmap('inferno', 0, 0.9)

os.chdir('/Users/toneill/N159/Plots/extinct_law/')

#### define previos results

## de marchi 30Dor
r136_invlam = [2.944,1.846,1.519,1.298,0.868,0.651,1.111,0.508,0.534,0.299,0.229,0.212]
r136_r57 = [4.69,3.29,2.73,2.31,1.38,0.9,1.87,0.63,0.69,0.35,0.29,0.28]
r136_r57_err = [0.25,0.18,0.19,0.18,0.15,0.13,0.17,0.12,0.13,0.11,0.11,0.11]

r136_r092 = [3.72,2.58,2.17,1.79,1.08,0.7,1.47,0.49,0.54,0.26,0.22,0.2]
r136_r092_err = [0.28,0.14,0.13,0.12,0.11,0.11,0.11,0.11,0.12,0.11,0.11,0.1]

lowess_57 = sm.nonparametric.lowess(r136_r57, r136_invlam,frac=0.2)
lowess_092 = sm.nonparametric.lowess(r136_r092, r136_invlam,frac=0.2)

####################################################
# load data

fuse = 'vis.ir'
all_vi = load_phot(region='n159-all',fuse=fuse)
e_vi = load_phot(region='n159-e',fuse=fuse)
w_vi = load_phot(region='n159-w',fuse=fuse)
s_vi = load_phot(region='n159-s',fuse=fuse)
off_vi = load_phot(region='off-point',fuse=fuse)
region_dicts = {'n159e':e_vi,'n159w':w_vi,'n159s':s_vi,'all':all_vi,'off':off_vi}

'''plt.figure()
plt.scatter(off_vi['555min814'],off_vi['mag_555'],s=2,alpha=0.5,c='k')
plt.gca().invert_yaxis()
plt.xlim(-0.1,2.5)
plt.ylim(27,18)


fuse = 'vis.ir'
all_vi = load_phot(region='n159-all',fuse=fuse)

fuse = 'vis'
all_vis = load_phot(region='n159-all',fuse=fuse)

fuse = 'ir'
all_ir = load_phot(region='n159-all',fuse=fuse)


plt.figure()
plt.scatter(all_ir['125min160'],all_ir['mag_160'],s=2,alpha=0.5,c='k')
plt.gca().invert_yaxis()
plt.xlabel('125 - 160')
plt.ylabel('160')


plt.figure()
plt.scatter(all_ir['125min160'],all_ir['mag_125'],s=2,alpha=0.5,c='k')
plt.gca().invert_yaxis()
plt.xlabel('125 - 160')
plt.ylabel('125')


plt.figure()
plt.scatter(all_vis['555min814'],all_vis['mag_814'],s=2,alpha=0.5,c='k')
plt.gca().invert_yaxis()
plt.xlabel('555 - 814')
plt.ylabel('814')

plt.figure()
plt.scatter(all_vis['555min814'],all_vis['mag_555'],s=2,alpha=0.5,c='k')
plt.gca().invert_yaxis()
plt.xlabel('555 - 814')
plt.ylabel('555')



klim = (all_vi['555min814'] >= 0.9) &  (all_vi['555min814'] <= 2.5) & (all_vi['mag_555'] >= 18) & (all_vi['mag_555'] <= 22.5)
plt.figure()
#hist = sns.histplot(all_vi[klim],x='555min814',y='mag_555',kde=True)
sns.kdeplot(x=all_vi['555min814'][klim],y=all_vi['mag_555'][klim],fill=True,cut=0)
plt.gca().invert_yaxis()
#plt.scatter(all_vi['555min814'],all_vi['mag_555'],s=2,alpha=0.5,c='k')
plt.xlim(0.8,2.5)
plt.ylim(23,19)'''

########
# define filter props

lambda_refs = {'mag_555':5361.03,'mag_814':8045.53,'mag_125':12486.07,
               'mag_160':15370.34}
filts_rep = list(lambda_refs.keys())

inv_lams = {}
for i in range(len(filts_rep)):
    lref = lambda_refs[filts_rep[i]]
    lmic = lref*u.angstrom.to(u.micron)
    inv_lams[filts_rep[i]] = 1/lmic



##############################################
# define expected center and boundaries of RC

### NOTE: RIGHT NOW, EXPECTED FOR 125 and 160 ARE BY EYE
exp_cols = {'555min814':0.925,'125min160':0.4}
exp_mags = {'mag_555':19.07,'mag_814':18.14,'mag_125':17.65,'mag_160':17.2}

col_bounds = {'555min814':[0.85,2.5],'125min160':[0.35,1]}
mag_bounds = {'mag_555':[18.5,22],'mag_814':[17.5,21],'mag_125':[16.5,19],'mag_160':[16.5,19]}


##############################################
# function to fit rc slope

def fit_rc(r_df,col_use='555min814',mag_use='mag_555',nrun=500):#,exp_cols=exp_cols,exp_mags=exp_mags,col_bounds=col_bounds)

    xs = r_df[col_use].values.ravel()
    ys = r_df[mag_use].values.ravel()

    exp_col = exp_cols[col_use]
    exp_mag = exp_mags[mag_use]
    col_cut = col_bounds[col_use]
    mag_cut = mag_bounds[mag_use]
    rc_inds = (ys <= mag_cut[1]) & (ys >= mag_cut[0]) & (xs <= col_cut[1]) & (xs >= col_cut[0])
    rc_df = r_df[rc_inds].reset_index()
    X0 = rc_df[col_use].values.reshape(-1, 1) - exp_col
    y0 = rc_df[mag_use].values - exp_mag

    ran_mc = np.vstack([ransac_nointercept(X0, y0,fit_int=False) for i in range(nrun)])
    ran_m = ran_mc[:, 0]
    ran_c = ran_mc[:, 1]
    ran_in = ran_mc[:, 2]
    P_in = np.sum(ran_in, axis=0) / nrun

    rc_df['P_rc'] = P_in
    #saveslopes.append(ran_m)

    return rc_df, ran_m, ran_c

### need to cycle through
# region
#   colors (555m814 and 125m160)
#       mags (125, 160, 555, 814)



poss_cols = ['555min814','125min160']
poss_mags = ['mag_555','mag_814','mag_125','mag_160']


def red_curv(r_df,col_use='555min814',plot=False,nrun=500,region='all'):

    col_slopes = np.full((len(poss_mags),3),np.nan)

    if plot:
        fig, axs = plt.subplots(2,2,figsize=(10,10),sharex=True,sharey=True)
        axs = axs.ravel()

    for m in range(len(poss_mags)):
        print(m)

        mag_use = poss_mags[m]
        rc_df, m_dist, c_dist = fit_rc(r_df,col_use=col_use,mag_use=mag_use,nrun=nrun)

        col_slopes[m][0] = np.mean(m_dist)
        col_slopes[m][1] = np.std(m_dist)
        col_slopes[m][2] = inv_lams[mag_use]

        #c_int = np.mean(c_dist)

        rc_df.to_csv(f'redclump_cands/rc_{region}_{col_use}_{mag_use}.csv',index=False)

        if plot:
            ax = axs[m]
            ax.scatter(r_df[col_use],r_df[mag_use],c='gray',s=1.5,alpha=0.3)
            ax.scatter(rc_df[col_use],rc_df[mag_use],c=rc_df['P_rc'],cmap=cmap_use,s=0.8,alpha=1,
                       label='$R_{\\lambda} =$ '+f'{col_slopes[m][0]:.2f} $\\pm$ {col_slopes[m][1]:.2f}')
            ax.set_xlabel(col_use,fontsize=13)
            ax.set_ylabel(mag_use,fontsize=14)

            exp_col = exp_cols[col_use]
            exp_mag = exp_mags[mag_use]
            ax.scatter(exp_col,exp_mag,edgecolor='r',facecolor='None',marker='s',alpha=0.5,label='Expected RC')
            #ax.scatter(exp_col,exp_mag+c_int,edgecolor='b',facecolor='None',marker='o',alpha=0.5,label='Fit RC')
            ax.legend(loc='lower right')

    if plot:
        if col_use == '125min160':
            ax.set_xlim(-0.5,2)
        if col_use == '555min814':
            ax.set_xlim(-0.5,4)
        ax.set_ylim(27, 16)
        fig.tight_layout()
        #plt.savefig(f'cmd_{region}_{col_use}.png',dpi=300)
        plt.savefig(f'rcfit_{region}_{col_use}.png',dpi=300)
    return col_slopes

clobber = True

if clobber:
    nrun = 1000

    rvs_vis = red_curv(all_vi,col_use='555min814',plot=True,nrun=nrun)
    rvs_ir = red_curv(all_vi,col_use='125min160',plot=True,nrun=nrun)

    e_vis = red_curv(e_vi,col_use='555min814',plot=True,nrun=nrun,region='e')
    e_ir = red_curv(e_vi,col_use='125min160',plot=True,nrun=nrun,region='e')

    w_vis = red_curv(w_vi,col_use='555min814',plot=True,nrun=nrun,region='w')
    w_ir = red_curv(w_vi,col_use='125min160',plot=True,nrun=nrun,region='w')

    s_vis = red_curv(s_vi,col_use='555min814',plot=True,nrun=nrun,region='s')
    s_ir = red_curv(s_vi,col_use='125min160',plot=True,nrun=nrun,region='s')

    off_vis = red_curv(off_vi,col_use='555min814',plot=True,nrun=nrun,region='off')
    off_ir = red_curv(off_vi,col_use='125min160',plot=True,nrun=nrun,region='off')

#if not clobber:

##############
## summary plot

lprops = {'lw':0.75,'capsize':2,'marker':'o','markersize':5}

plt.figure()
plt.errorbar(rvs_vis[:,2],rvs_vis[:,0],yerr=rvs_vis[:,1],label='N159',c='cornflowerblue',zorder=5,**lprops)
'''plt.errorbar(e_vis[:,2],e_vis[:,0],yerr=e_vis[:,1],label='N159E',c='#ed6495',ls=':',**lprops)#,markersize=1)
plt.errorbar(w_vis[:,2],w_vis[:,0],yerr=w_vis[:,1],label='N159W',ls=':',c='#edbc64',**lprops)
plt.errorbar(s_vis[:,2],s_vis[:,0],yerr=s_vis[:,1],label='N159S',ls=':',c='#7864ed',**lprops)'''
plt.errorbar(off_vis[:,2],off_vis[:,0],yerr=off_vis[:,1],label='Off',ls=':',c='r',**lprops)
plt.gca().set_xscale('log')
plt.errorbar(r136_invlam,r136_r57,yerr=r136_r57_err,capsize=2,marker='d',ls='None',label='30Dor (F555W - F775W)',c='gray')
plt.plot(lowess_57[:,0],lowess_57[:,1],c='gray',zorder=0,ls=':')
plt.legend()
plt.ylabel('$A_\\lambda$/ E(F555W - F814W)',fontsize=14)
plt.xlabel('1/$\\lambda$ [$\\mu$m$^{-1}$]',fontsize=14)
plt.title(f'Color: F555W - F814W')
plt.tight_layout()
#plt.savefig('rv.curv_555min814.png',dpi=300)
plt.savefig('rv.curv_zoom_off_555min814.png',dpi=300)

plt.figure()
plt.errorbar(rvs_ir[:,2],rvs_ir[:,0],yerr=rvs_ir[:,1],label='N159',c='cornflowerblue',zorder=5,**lprops)
plt.errorbar(e_ir[:,2],e_ir[:,0],yerr=e_ir[:,1],label='N159E',ls=':',c='#ed6495',**lprops)
plt.errorbar(w_ir[:,2],w_ir[:,0],yerr=w_ir[:,1],label='N159W',ls=':',c='#edbc64',**lprops)
plt.errorbar(s_ir[:,2],s_ir[:,0],yerr=s_ir[:,1],label='N159S',ls=':',c='#7864ed',**lprops)
plt.errorbar(off_ir[:,2],off_ir[:,0],yerr=off_ir[:,1],label='Off',ls=':',c='r',**lprops)
plt.gca().set_xscale('log')
plt.errorbar(r136_invlam,r136_r092,yerr=r136_r092_err,capsize=2,marker='d',ls='None',label='30Dor (F090W - F200W)',c='gray')
plt.plot(lowess_092[:,0],lowess_092[:,1],c='gray',zorder=0,ls=':')
plt.legend()
plt.ylabel('$A_\\lambda$/ E(F125W - F160W)',fontsize=14)
plt.xlabel('1/$\\lambda$ [$\\mu$m$^{-1}$]',fontsize=14)
plt.title(f'Color: F125W - F160W')
plt.tight_layout()
plt.savefig('rv.curv_125min160.png',dpi=300)
#plt.savefig('rv.curv_zoom_125min160.png',dpi=300)

#####################################################

## interpolate to johnson

from scipy.interpolate import interp1d


targ_curvs = [1.82,1.52,1.24,0.82,0.61]
full_targ_curvs = [2.74,2.25,1.82,1.52,1.24,0.82,0.61,0.46]

targ_names = ['V','R','I','J','H']
targ_RMW = [2.3,1.78,1.29,0.63,0.40]
full_targ_RMW = [3.61,3.05,2.3,1.78,1.29,0.63,0.40,0.26]
targ_R30old = [3.09,2.58,2.09,1.26,0.84]
full_targ_R30old = [4.41,3.78,3.09,2.58,2.09,1.26,0.84,0.52]

old30_invlam = [3.69,2.98,1.88,1.30,0.86,0.65]
old30_r57 = [5.15,4.79,3.35,2.26,1.41,0.95]


interp_rvs = interp1d(rvs_vis[:,2],rvs_vis[:,0],kind='linear',fill_value='extrapolate')
rvs_targ = interp_rvs(targ_curvs)
interp_rvs = interp1d(e_vis[:,2],e_vis[:,0],kind='linear',fill_value='extrapolate')
e_targ = interp_rvs(targ_curvs)
interp_rvs = interp1d(w_vis[:,2],w_vis[:,0],kind='linear',fill_value='extrapolate')
w_targ = interp_rvs(targ_curvs)
interp_rvs = interp1d(s_vis[:,2],s_vis[:,0],kind='linear',fill_value='extrapolate')
s_targ = interp_rvs(targ_curvs)


#r136_invlam,r136_r57
interp_rvs = interp1d(r136_invlam,r136_r57,kind='linear',fill_value='extrapolate')
new_targ = interp_rvs(full_targ_curvs)

plt.figure()
plt.plot(targ_curvs,rvs_targ,label='N159',c='cornflowerblue',zorder=5,marker='o',ls='-')
plt.plot(targ_curvs,targ_R30old,label='30Dor (2016)',c='gray',marker='d',ls=':')
plt.plot(full_targ_curvs,new_targ,label='30Dor (2023)',c='g',**targ_lprops,marker='d')
#plt.plot(old30_invlam,old30_r57,label='30Dor (2016)',c='orange',**targ_lprops,marker='d')
plt.plot(full_targ_curvs,full_targ_RMW,label='MW',c='firebrick',marker='s',ls='--')
plt.gca().set_xscale('log')
plt.legend()
plt.ylabel('$A_\\lambda$/ E(V - I)',fontsize=14)
plt.xlabel('1/$\\lambda$ [$\\mu$m$^{-1}$]',fontsize=14)
plt.title(f'Color: V - I')
plt.tight_layout()
plt.savefig('rv.curv_VminI.png',dpi=300)


plt.figure()
plt.plot(targ_curvs,rvs_targ/targ_RMW,label='N159 / MW',c='cornflowerblue',zorder=5,marker='o',ls='-')
plt.plot(targ_curvs,rvs_targ/targ_R30old,label='N159 / 30Dor (2016)',c='orange',zorder=5,marker='o',ls='-')
plt.plot(targ_curvs,rvs_targ/new_targ[2:-1],label='N159 / 30Dor (2023)',c='r',zorder=5,marker='o',ls='-')
plt.plot(full_targ_curvs,np.array(full_targ_R30old)/full_targ_RMW,label='30Dor (2016) / MW',c='gray',marker='d',ls='-')
plt.plot(full_targ_curvs,np.array(new_targ)/full_targ_RMW,label='30Dor (2023) / MW',c='g',marker='d',ls='-')
#plt.plot(targ_curvs,rvs_targ-targ_RMW,label='N159',c='cornflowerblue',zorder=5,ls=':')
#plt.plot(full_targ_curvs,np.array(full_targ_R30old)-full_targ_RMW,label='30Dor (2016)',c='gray',ls=':')
#plt.plot(targ_curvs,targ_RMW,label='MW',c='firebrick',marker='s',ls=':')
plt.gca().set_xscale('log')
plt.legend()
plt.ylabel('Ratio $R_\\lambda$ / MW',fontsize=14)
plt.xlabel('1/$\\lambda$ [$\\mu$m$^{-1}$]',fontsize=14)
plt.title(f'Color: V - I')
plt.tight_layout()
plt.ylim(0.5,2.3)
plt.axhline(y=1,c='k',ls='--')
plt.savefig('rv.curv_ratio_VminI.png',dpi=300)


plt.figure()
plt.plot(targ_curvs,rvs_targ/targ_RMW,label='N159 / MW',c='cornflowerblue',zorder=5,marker='o',ls='-')
plt.plot(targ_curvs,e_targ/targ_RMW,label='N159E / MW',c='#ed6495',zorder=5,marker='o',ls='-')
plt.plot(targ_curvs,w_targ/targ_RMW,label='N159W / MW',c='#edbc64',zorder=5,marker='o',ls='-')
plt.plot(targ_curvs,s_targ/targ_RMW,label='N159-S / MW',c='#7864ed',zorder=5,marker='o',ls='-')
plt.gca().set_xscale('log')
plt.legend()
plt.ylabel('Ratio $R_\\lambda$ / MW',fontsize=14)
plt.xlabel('1/$\\lambda$ [$\\mu$m$^{-1}$]',fontsize=14)
plt.title(f'Color: V - I')
plt.tight_layout()
plt.ylim(0.5,2.2)
plt.axhline(y=1,c='k',ls='--')
plt.savefig('rv.curv_ratio_N159only_VminI.png',dpi=300)


###################################################
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=5, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=10, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

def gauss_rc(r_df,col_use='555min814',region='all'):

    col_slopes = np.full((len(poss_mags),3),np.nan)

    for m in range(len(poss_mags)):
        print(m)
        rc_df = pd.read_csv(f'redclump_cands/rc_{region}_{col_use}_{mag_use}.csv')

        X = rc_df[[col_use,mag_use]].values#[rc_df['P_rc']>=0.5]
        gmm = GaussianMixture(n_components=2).fit(X)

        plt.figure()
        plot_gmm(gmm,X)
        plt.gca().invert_yaxis()
        plt.title(f'{region}')

