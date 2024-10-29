
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cmasher as cmr
from astropy import units as u
from astropy.coordinates import SkyCoord
from sklearn import linear_model
import seaborn as sns
from n159_photometry.load_data import load_phot
import statsmodels.api as sm
import os
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Avenir"
matplotlib.rcParams['font.family'] = "sans-serif"


def model_valid(model,*random_data) -> bool:
    return model.coef_[0] > 0

def ransac_nointercept(X, y,fit_int=False):
    ransac = linear_model.RANSACRegressor(max_trials=1000,
              estimator=linear_model.LinearRegression(fit_intercept=fit_int),
                                          is_model_valid=model_valid)
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    m, c = float(ransac.estimator_.coef_), float(ransac.estimator_.intercept_)
    return [m, c], [inlier_mask * 1]

#e_ir = red_curv(e_vi,col_use='125min160',plot=True,nrun=500,region='e')

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
#exp_cols = {'555min814':0.925,'125min160':0.4}
exp_mags = {'mag_555':19.16,'mag_814':18.10,'mag_125':17.69,'mag_160':17.25}
exp_cols = {'555min814':exp_mags['mag_555']-exp_mags['mag_814'],'125min160':exp_mags['mag_125']-exp_mags['mag_160']}

col_bounds = {'555min814':[0.8,4],'125min160':[0.25,2]}
mag_bounds = {'mag_555':[18.5,22.5],'mag_814':[17.5,21.5],'mag_125':[16.5,20.5],'mag_160':[15.5,19.5]}

plt.figure()
plt.scatter(all_vi['125min160'],all_vi['mag_160'],s=3)
plt.gca().invert_yaxis()
##############################################
# function to fit rc slope

def fit_rc(r_df,col_use='555min814',mag_use='mag_555',nrun=500,fit_intercept=False,rc_loc=None):#,exp_cols=exp_cols,exp_mags=exp_mags,col_bounds=col_bounds)

    xs = r_df[col_use].values.ravel()
    ys = r_df[mag_use].values.ravel()

    exp_col = rc_loc[0]#exp_cols[col_use]
    exp_mag = rc_loc[1]#exp_mags[mag_use]
    col_cut = col_bounds[col_use]
    mag_cut = mag_bounds[mag_use]
    rc_inds = (ys <= mag_cut[1]) & (ys >= mag_cut[0]) & (xs <= col_cut[1]) & (xs >= col_cut[0])
    rc_df = r_df[rc_inds].reset_index()
    X0 = rc_df[col_use].values.reshape(-1, 1) - exp_col
    y0 = rc_df[mag_use].values - exp_mag

    ran_mc = [ransac_nointercept(X0, y0,fit_int=fit_intercept) for i in range(nrun)]
    ran_mc_params =np.array( [ran_mc[i][0] for i in range(nrun)])
    ran_mc_inlist = np.array([ran_mc[i][1][0] for i in range(nrun)])
    ran_m = ran_mc_params[:, 0]
    ran_c = ran_mc_params[:, 1]
    ran_in = ran_mc_inlist
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

import seaborn as sns

def red_curv(r_df,col_use='555min814',plot=False,nrun=500,region='all',fit_intercept=False):

    col_slopes = np.full((len(poss_mags),3),np.nan)
    col_rc_locs = np.full((len(poss_mags), 2), np.nan)

    if plot:
        fig, axs = plt.subplots(1,4,figsize=(16,6),sharex=True,sharey=False)
        axs = axs.ravel()

    for m in range(len(poss_mags)):
        print(m)

        mag_use = poss_mags[m]

        rc_loc = find_rc_center(r_df, col_use=col_use, mag_use=mag_use, plot=False, bw=.3)
        rc_df, m_dist, c_dist = fit_rc(r_df,col_use=col_use,mag_use=mag_use,nrun=nrun,fit_intercept=fit_intercept,rc_loc=rc_loc)

        col_slopes[m][0] = np.median(m_dist)
        col_slopes[m][1] = np.std(m_dist)
        col_slopes[m][2] = inv_lams[mag_use]

        col_rc_locs[m] = rc_loc

        #rc_df.to_csv(f'redclump_cands/rc_{region}_{col_use}_{mag_use}.csv',index=False)

        if plot:
            ax = axs[m]

            col_cut = col_bounds[col_use]
            mag_cut = mag_bounds[mag_use]
            rpatch = Rectangle((col_cut[0], mag_cut[0]), col_cut[1] - col_cut[0], mag_cut[1] - mag_cut[0],
                               facecolor='None', edgecolor='k',ls='--')

            ax.scatter(r_df[col_use],r_df[mag_use],c='gray',s=0.5,alpha=0.3)
            scat_cm = ax.scatter(rc_df[col_use],rc_df[mag_use],c=rc_df['P_rc'],cmap=cmap_use,s=0.5,alpha=0.9,
                       vmin=0,vmax=1)
            ax.plot([-100,-100],[-10,-10],c='firebrick',lw=2,label='$R_{\\lambda} =$ '+f'{col_slopes[m][0]:.2f} $\\pm$ {col_slopes[m][1]:.2f}')
            #sns.kdeplot(x=rc_df[col_use],y=rc_df[mag_use],cmap='Blues',ax=ax)
            ax.set_xlabel(f'F{col_use[0:3]}W - F{col_use[6::]}W [mag]', fontsize=14)
            ax.set_ylabel(f'F{mag_use[4::]}W [mag]', fontsize=16)
            ax.scatter(rc_loc[0], rc_loc[1], edgecolor='firebrick', facecolor='None',marker='8', alpha=0.5, s=100,label='RC Center')
            ax.scatter(exp_cols[col_use],exp_mags[mag_use],c='r',marker='x',alpha=0.5,label='DP14 RC Center')

            line_X = np.arange(0, 1, 0.05)
            line_Y = col_slopes[m][0] * line_X + np.median(c_dist)
            max_A = 1 # mag
            use_line = np.where(np.sqrt(line_X**2 + line_Y**2) <= max_A)
            # rc_pcheck = P_in >= 0.9

            lineX_plot = line_X[use_line] + rc_loc[0]
            lineY_plot = line_Y[use_line] + rc_loc[1]

            arrprop = dict(arrowstyle="-|>,head_width=0.4,head_length=0.8",
                           shrinkA=0, shrinkB=0, color='firebrick', lw=2, label='$R_{\\lambda} =$ '+f'{col_slopes[m][0]:.2f} $\\pm$ {col_slopes[m][1]:.2f}')

            ax.annotate("", xy=(lineX_plot[-1], lineY_plot[-1]),
                        xytext=(lineX_plot[0], lineY_plot[0]), arrowprops=arrprop,
                        label='$R_{\\lambda} =$ '+f'{col_slopes[m][0]:.2f} $\\pm$ {col_slopes[m][1]:.2f}')#, zorder=6)

            ax.legend(loc='lower left')
            ax.add_patch(rpatch)

    if plot:
        if col_use == '125min160':
            ax.set_xlim(-1,2.5)
        if col_use == '555min814':
            ax.set_xlim(-1,4.5)
        [ax.set_ylim(29, 15) for ax in axs]
        fig.suptitle(region.upper(),fontsize=20)

        for ax in axs:
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.tick_params(direction='in', which='both',labelsize=12)  # , labelsize=15)
            ax.minorticks_on()
            ax.tick_params(which='major',length=5)
            ax.tick_params(which='minor',length=3)

        fig.tight_layout()
        fig.subplots_adjust(right=0.92)

        cbar_ax = fig.add_axes([0.93, 0.1, 0.015, 0.8])
        cbar = fig.colorbar(scat_cm, cax=cbar_ax)
        cbar.set_label(label='P(Red Clump)', fontsize=14, labelpad=5)
        cbar.ax.tick_params(labelsize=12)
        #plt.savefig(f'cmd_{region}_{col_use}.png',dpi=300)
        #lt.savefig(f'rcfit_{region}_{col_use}.png',dpi=300)
    return col_slopes, col_rc_locs

#################################################


from findthegap.gapper import Gapper
from sklearn.preprocessing import StandardScaler
import time
from matplotlib.patches import Rectangle


def find_rc_center(r_df,col_use='555min814',mag_use='mag_555',bw=.2,gridding_size=80,plot=False):

    xs = r_df[col_use].values.ravel()
    ys = r_df[mag_use].values.ravel()

    exp_col = exp_cols[col_use]
    exp_mag = exp_mags[mag_use]
    col_cut = col_bounds[col_use]
    mag_cut = mag_bounds[mag_use]
    rc_inds = (ys <= mag_cut[1]) & (ys >= mag_cut[0]) & (xs <= col_cut[1]) & (xs >= col_cut[0])
    rc_df = r_df[rc_inds].reset_index()

    data_nonorm = rc_df[[col_use,mag_use]].values
    scaler = StandardScaler()
    scaler.fit(data_nonorm)
    data = scaler.transform(data_nonorm)
    bounds = np.array([[np.min(data[:,d]),np.max(data[:,d])] for d in range(data.shape[1])])

    gapper_base = Gapper(data, bw, bounds)

    grid_data, grid_density = gapper_base.compute_density_grid(gridding_size = gridding_size)
    critical_points = gapper_base.compute_all_critical_points(distance_to_group= .1*bw)
    point_ = data[np.random.randint(data.shape[0])]
    density_ofpt = gapper_base.get_density(point_)
    eig_val_H, eig_vec_H, dens_est, gradient, Hessian = gapper_base.get_g_H_eigvH(point_)
    H_eigval_crits = []
    for pts in critical_points:
        eig_val_H, eig_vec_H, logd_apoint, g, H = gapper_base.get_g_H_eigvH(pts)
        H_eigval_crits.append(eig_val_H)
    H_eigval_crits = np.array(H_eigval_crits)
    max_eigval_H = np.max(H_eigval_crits, axis=1)
    pts_use, = np.where(max_eigval_H<0)

    density_matr = grid_density.reshape((gridding_size, gridding_size))
    grid_linspace = [np.linspace(bounds[d][0], bounds[d][1], gridding_size) for d in range(data.shape[1])]
    meshgrid = np.array(np.meshgrid(*grid_linspace, indexing='ij'))

    if plot:
        plt.figure(figsize=(8,7))
        ctf = plt.contourf(meshgrid[0], meshgrid[1], density_matr, 20, cmap=cm.cividis)
        critsc = plt.scatter(critical_points[:,0][pts_use], critical_points[:,1][pts_use], s=80, c=max_eigval_H[pts_use],
                    cmap = cm.RdBu.reversed(), edgecolor='k', vmin = -(np.max(np.abs(max_eigval_H))),
                             vmax=np.max(np.abs(max_eigval_H)))
        cb = plt.colorbar(label='Maximum eigenvalue of Hessian')
        plt.gca().invert_yaxis()
        plt.show()

    neg_loc_scale = critical_points[pts_use,:]
    neg_loc = scaler.inverse_transform(neg_loc_scale)
    d_exp_rc = np.sqrt((neg_loc[:,0]-exp_cols[col_use])**2 + (neg_loc[:,1] - exp_mags[mag_use])**2)
    rc_loc = neg_loc[np.argmin(d_exp_rc),:]

    #rc_loc_scale = critical_points[np.argmin(max_eigval_H),:]
    #rc_loc = scaler.inverse_transform(rc_loc_scale.reshape(1,-1))[0]
    print(f'Red clump location: {rc_loc[0]:.3f}, {rc_loc[1]:.3f}')

    if plot:
        rpatch = Rectangle((col_cut[0],mag_cut[0]),col_cut[1]-col_cut[0],mag_cut[1]-mag_cut[0],
                           facecolor='None',edgecolor='k')

        plt.figure()
        plt.scatter(xs,ys,c='gray',s=0.1)
        plt.scatter(neg_loc[:,0],neg_loc[:,1],c=d_exp_rc)
        plt.scatter(rc_loc[0],rc_loc[1],c='r',marker='x')
        plt.scatter(exp_cols[col_use],exp_mags[mag_use],c='maroon',marker='s',alpha=0.5)
        cb = plt.colorbar(label='Distance from expected RC position')
        plt.gca().invert_yaxis()
        plt.gca().add_patch(rpatch)
        plt.title(f'Red clump location: {rc_loc[0]:.3f}, {rc_loc[1]:.3f}')

    return rc_loc


col_use = '555min814'
mag_use ='mag_555'
r_df = all_vi

save_loc = []
for r_df in [all_vi,e_vi,w_vi,s_vi, off_vi]:
    rc_loc = find_rc_center(r_df,col_use=col_use,mag_use=mag_use,plot=True)
    save_loc.append(rc_loc)

save_loc = []
for m in range(len(poss_mags)):
    rc_loc = find_rc_center(r_df,col_use=col_use,mag_use=poss_mags[m],plot=True,bw=.3)
    save_loc.append(rc_loc)




##################################################
clobber = True

if clobber:
    nrun = 100

    rvs_vis, rvs_vis_rc = red_curv(all_vi,col_use='555min814',plot=True,nrun=nrun,fit_intercept=False,region='N159')
    rvs_ir, rvs_ir_rc = red_curv(all_vi,col_use='125min160',plot=True,nrun=nrun,region='N159')

    e_vis, e_vis_rc = red_curv(e_vi,col_use='555min814',plot=True,nrun=nrun,region='N159E')
    #plt.savefig('/Users/toneill/N159/Plots/final_plots/n159e_rc.pdf',dpi=300)
    e_ir, e_ir_rc = red_curv(e_vi,col_use='125min160',plot=True,nrun=nrun,region='N159E')
    #plt.savefig('/Users/toneill/N159/Plots/final_plots/n159e_rc_ir.pdf', dpi=300)

    w_vis, w_vis_rc = red_curv(w_vi,col_use='555min814',plot=True,nrun=nrun,region='N159W')
    w_ir, w_ir_rc = red_curv(w_vi,col_use='125min160',plot=True,nrun=nrun,region='N159W')

    s_vis, s_vis_rc = red_curv(s_vi,col_use='555min814',plot=True,nrun=nrun,region='N159S')
    s_ir, s_ir_rc = red_curv(s_vi,col_use='125min160',plot=True,nrun=nrun,region='N159S')

    off_vis, off_vis_rc = red_curv(off_vi,col_use='555min814',plot=True,nrun=nrun,region='Off')
    off_ir, off_ir_rc = red_curv(off_vi,col_use='125min160',plot=True,nrun=nrun,region='Off')

#if not clobber:

    fig, axs = plt.subplots(1,4,figsize=(16,6),sharex=True,sharey=True)
    axs = axs.ravel()

    for m in range(len(poss_mags)):
        print(m)

        mag_use = poss_mags[m]

        if plot:
            ax = axs[m]
            ax.scatter(r_df[col_use],r_df[mag_use],c='gray',s=0.5,alpha=0.3)
            ax.set_xlabel(f'F{col_use[0:3]}W - F{col_use[6::]}W [mag]', fontsize=12)
            ax.set_ylabel(f'F{mag_use[4::]}W [mag]', fontsize=12)
            exp_col = exp_cols[col_use]
            exp_mag = exp_mags[mag_use]
            '''if col_use == '125min160':
                exp_col = 0.54
            if mag_use == 'mag_160':
                exp_mag = 17.15'''
            ax.scatter(exp_col,exp_mag,edgecolor='r',facecolor='None',marker='s',alpha=0.5,label='Expected RC')
            #ax.scatter(exp_col,exp_mag+c_int,edgecolor='b',facecolor='None',marker='o',alpha=0.5,label='Fit RC')
            ax.legend(loc='lower right')

    if plot:
        if col_use == '125min160':
            ax.set_xlim(-1,2)
        if col_use == '555min814':
            ax.set_xlim(-1,4)
        ax.set_ylim(28, 16)
        fig.tight_layout()





##############
## summary plot

lprops = {'lw':1,'capsize':2,'marker':'o','markersize':5}

c_e = 'maroon'
c_w = 'r'
c_s = 'darkorange'
c_off = 'k'
c_all = 'royalblue'

plt.figure()
plt.errorbar(rvs_vis[:,2],rvs_vis[:,0],yerr=rvs_vis[:,1],label='N159',c=c_all,zorder=5,**lprops)
plt.errorbar(e_vis[:,2],e_vis[:,0],yerr=e_vis[:,1],label='N159E',c=c_e,ls=':',**lprops)#,markersize=1)
plt.errorbar(w_vis[:,2],w_vis[:,0],yerr=w_vis[:,1],label='N159W',ls=':',c=c_w,**lprops)
plt.errorbar(s_vis[:,2],s_vis[:,0],yerr=s_vis[:,1],label='N159S',ls=':',c=c_s,**lprops)
plt.errorbar(off_vis[:,2],off_vis[:,0],yerr=off_vis[:,1],label='Off',ls=':',c=c_off,**lprops)
plt.gca().set_xscale('log')
plt.errorbar(r136_invlam,r136_r57,yerr=r136_r57_err,capsize=2,marker='d',ls='None',label='30Dor (F555W - F775W)',c='gray')
plt.plot(lowess_57[:,0],lowess_57[:,1],c='gray',zorder=0,ls=':')
plt.legend()
plt.ylabel('$A_\\lambda$/ E(F555W - F814W)',fontsize=14)
plt.xlabel('1/$\\lambda$ [$\\mu$m$^{-1}$]',fontsize=14)
plt.title(f'Color: F555W - F814W')
plt.tight_layout()
#plt.savefig('rv.curv_555min814.png',dpi=300)
#plt.savefig('rv.curv_zoom_off_555min814.png',dpi=300)

plt.figure()
#plt.errorbar(rvs_ir[:,2],rvs_ir[:,0],yerr=rvs_ir[:,1],label='N159',c=c_all,zorder=5,**lprops)
plt.errorbar(e_ir[:,2],e_ir[:,0],yerr=e_ir[:,1],label='N159E',ls=':',c=c_e,**lprops)
plt.errorbar(w_ir[:,2],w_ir[:,0],yerr=w_ir[:,1],label='N159W',ls=':',c=c_w,**lprops)
plt.errorbar(s_ir[:,2],s_ir[:,0],yerr=s_ir[:,1],label='N159S',ls=':',c=c_s,**lprops)#'''
plt.errorbar(off_ir[:,2],off_ir[:,0],yerr=off_ir[:,1],label='Off',ls=':',c=c_off,**lprops)
plt.gca().set_xscale('log')
plt.errorbar(r136_invlam,r136_r092,yerr=r136_r092_err,capsize=2,marker='d',ls='None',label='30Dor (F090W - F200W)',c='gray')
plt.plot(lowess_092[:,0],lowess_092[:,1],c='gray',zorder=0,ls=':')
plt.legend()
plt.ylabel('$A_\\lambda$/ E(F125W - F160W)',fontsize=14)
plt.xlabel('1/$\\lambda$ [$\\mu$m$^{-1}$]',fontsize=14)
plt.title(f'Color: F125W - F160W')
plt.tight_layout()
#plt.savefig('rv.curv_125min160.png',dpi=300)
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
plt.plot(full_targ_curvs,new_targ,label='30Dor (2023)',c='g',marker='d')
#plt.plot(old30_invlam,old30_r57,label='30Dor (2016)',c='orange',**targ_lprops,marker='d')
plt.plot(full_targ_curvs,full_targ_RMW,label='MW',c='firebrick',marker='s',ls='--')
plt.gca().set_xscale('log')
plt.legend()
plt.ylabel('$A_\\lambda$/ E(V - I)',fontsize=14)
plt.xlabel('1/$\\lambda$ [$\\mu$m$^{-1}$]',fontsize=14)
plt.title(f'Color: V - I')
plt.tight_layout()
#plt.savefig('rv.curv_VminI.png',dpi=300)


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
plt.ylim(0.5,2.8)
plt.axhline(y=1,c='k',ls='--')
#plt.savefig('rv.curv_ratio_VminI.png',dpi=300)


plt.figure()
#plt.plot(targ_curvs,rvs_targ/targ_RMW,label='N159 / MW',c=c_all,zorder=5,marker='o',ls='-')
plt.plot(targ_curvs,e_targ/targ_RMW,label='N159E / MW',c=c_e,zorder=5,marker='o',ls='-')
plt.plot(targ_curvs,w_targ/targ_RMW,label='N159W / MW',c=c_w,zorder=5,marker='o',ls='-')
plt.plot(targ_curvs,s_targ/targ_RMW,label='N159-S / MW',c=c_s,zorder=5,marker='o',ls='-')
plt.gca().set_xscale('log')
plt.legend()
plt.ylabel('Ratio $R_\\lambda$ / MW',fontsize=14)
plt.xlabel('1/$\\lambda$ [$\\mu$m$^{-1}$]',fontsize=14)
plt.title(f'Color: V - I')
plt.tight_layout()
plt.ylim(0.5,3)
plt.axhline(y=1,c='k',ls='--')
#plt.savefig('rv.curv_ratio_N159only_VminI.png',dpi=300)


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
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,  angle))#, **kwargs))


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

