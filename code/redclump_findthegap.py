import copy

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
import time

from findthegap.gapper import Gapper
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm

fuse = 'vis'
all_vi = load_phot(region='n159-all',fuse=fuse)
e_vi = load_phot(region='n159-e',fuse=fuse)
w_vi = load_phot(region='n159-w',fuse=fuse)
s_vi = load_phot(region='n159-s',fuse=fuse)
off_vi = load_phot(region='off-point',fuse=fuse)
region_dicts = {'n159e':e_vi,'n159w':w_vi,'n159s':s_vi,'all':all_vi,'off':off_vi}

### NOTE: RIGHT NOW, EXPECTED FOR 125 and 160 ARE BY EYE
exp_cols = {'555min814':0.925,'125min160':0.4}
exp_mags = {'mag_555':19.07,'mag_814':18.14,'mag_125':17.65,'mag_160':17.2}

col_bounds = {'555min814':[0.85,2.5],'125min160':[0.35,1]}
mag_bounds = {'mag_555':[18.5,22.75],'mag_814':[17.5,21],'mag_125':[16.5,19],'mag_160':[16.5,19]}


def bounds_rc(r_df,col_use='555min814',mag_use='mag_555'):

    xs = r_df[col_use].values.ravel()
    ys = r_df[mag_use].values.ravel()

    exp_col = exp_cols[col_use]
    exp_mag = exp_mags[mag_use]
    col_cut = col_bounds[col_use]
    mag_cut = mag_bounds[mag_use]
    rc_inds = (ys <= mag_cut[1]) & (ys >= mag_cut[0]) & (xs <= col_cut[1]) & (xs >= col_cut[0])
    rc_df = r_df[rc_inds].reset_index()

    return rc_df

region = 'off'
all_rc = bounds_rc(region_dicts[region])
data_nonorm = all_rc[['555min814','mag_555']].values


plt.figure(figsize=(8,7))
plt.scatter(data_nonorm[:,0],data_nonorm[:,1],c='grey',s=0.5)
plt.gca().invert_yaxis()
plt.title(f'{region}')



scaler = StandardScaler()
scaler.fit(data_nonorm)
data = scaler.transform(data_nonorm)

# Boundaries for the Gapper (if none are provided, this is the default mode)
bounds = np.array([[np.min(data[:,d]),np.max(data[:,d])] for d in range(data.shape[1])])

## Choice of bandwidth might be non trivial
## The gridding size impacts computation-time -- might have to be a function of bw
bw = .15
gridding_size = 80
gapper_base = Gapper(data, bw, bounds)

t = time.time()
grid_data, grid_density = gapper_base.compute_density_grid(gridding_size = gridding_size)
print((time.time() - t))
density_matr = grid_density.reshape((gridding_size, gridding_size))

grid_linspace = [ np.linspace(bounds[d][0], bounds[d][1], gridding_size) for d in range(data.shape[1]) ]
meshgrid = np.array(np.meshgrid(*grid_linspace, indexing='ij'))

plt.figure(figsize=(8,7))
ctf = plt.contourf(meshgrid[0], meshgrid[1], density_matr, 20, cmap=cm.cividis)
cb = plt.colorbar(label='Density estimate')
plt.gca().invert_yaxis()
plt.title(f'{region}')
plt.contour(off_mgrid[0], off_mgrid[1], off_densmatr, 20,cmap=cm.plasma,label='off')
plt.legend()


import copy
off_mgrid =copy.deepcopy(meshgrid)
off_densmatr = copy.deepcopy(density_matr)
#############


## Parameters:
## - distance_to_group: the code groups the resulting 'critical-points' after optimization if they're closer than
##                      some distance_to_group threshold
## Additional parameters: maxiter and gtol for the scipy.minimize call.
## This basically minimize the squared gradient of the density for each point on a (fine) meshgrid, then group the
## points found together. See code for details.
t = time.time()
critical_points = gapper_base.compute_all_critical_points(distance_to_group= .1*bw)
print(time.time() - t)


plt.figure(figsize=(8,7))
ctf = plt.contourf(meshgrid[0], meshgrid[1], density_matr, 20, cmap=cm.cividis)
cb = plt.colorbar(label='Density estimate')
plt.scatter(critical_points[:,0], critical_points[:,1], s=50, c='w', label='Critical points')
plt.gca().invert_yaxis()



H_eigval_crits = []
t = time.time()
for pts in critical_points:
    #The column eig_vec[:, i] is the normalized eigenvector corresponding to the eigenvalue eiv_val[i].
    #eigenvalue (and corresponding eig_vec) already sorted by ascending order (Following np.linalg.eigh function)
    eig_val_H, eig_vec_H, logd_apoint, g, H = gapper_base.get_g_H_eigvH(pts)
    H_eigval_crits.append(eig_val_H)
print(time.time() - t)


H_eigval_crits = np.array(H_eigval_crits)
max_eigval_H = np.max(H_eigval_crits, axis=1)
index_count_eigvalH = np.sum(H_eigval_crits < 0 , axis=1)

plt.figure(figsize=(8,7))
ctf = plt.contourf(meshgrid[0], meshgrid[1], density_matr, 20, cmap=cm.cividis)
critsc = plt.scatter(critical_points[:,0], critical_points[:,1], s=80, c=index_count_eigvalH,
            cmap = cm.PiYG.reversed(), edgecolor='k', vmin =0,
                     vmax=2)
cb = plt.colorbar(label='Index (count of eigenvalue < 0)')
plt.gca().invert_yaxis()


## Critical points with a maximum eigenvalue of the Hessian close to zero are of less interest if one is looking
## gaps or bumps in the density distribution


plt.figure(figsize=(8,7))
ctf = plt.contourf(meshgrid[0], meshgrid[1], density_matr, 20, cmap=cm.cividis)
critsc = plt.scatter(critical_points[:,0], critical_points[:,1], s=80, c=max_eigval_H,
            cmap = cm.RdBu.reversed(), edgecolor='k', vmin = -(np.max(np.abs(max_eigval_H))),
                     vmax=np.max(np.abs(max_eigval_H)))
cb = plt.colorbar(label='Maximum eigenvalue of Hessian')
plt.gca().invert_yaxis()


fig, axs = plt.subplots(1, 2, figsize=(9*2,7))
ctf = axs[0].contourf(meshgrid[0], meshgrid[1], density_matr, 20, cmap=cm.cividis)
critsc = axs[0].scatter(critical_points[:,0], critical_points[:,1], s=80, c= index_count_eigvalH,
                        cmap = cm.PiYG.reversed(), edgecolor='k', vmin =0,
                     vmax=2)
cb= fig.colorbar(critsc, ax=axs[0], label='Index (count of eigenvalue < 0)')

ctf = axs[1].contourf(meshgrid[0], meshgrid[1], density_matr, 20, cmap=cm.cividis)
critsc = axs[1].scatter(critical_points[:,0], critical_points[:,1], s=80, c=max_eigval_H,
            cmap = cm.RdBu.reversed(), edgecolor='k', vmin = -(np.max(np.abs(max_eigval_H))),
                     vmax=np.max(np.abs(max_eigval_H)))
cb = fig.colorbar(critsc, ax=axs[1], label='Maximum eigenvalue of Hessian')
[ax.invert_yaxis() for ax in axs]



N = 20

idx_best_crits  = np.argsort(max_eigval_H)[::-1][:N]

paths_best_crits = []
t = time.time()
for crit in critical_points[idx_best_crits]:
    path, feedbacks = gapper_base.compute_path_of_a_critpt(crit)
    paths_best_crits.append(path)

print(time.time() - t)



####################################

t = time.time()
eig_vals_H = np.array([np.linalg.eigh(gapper_base.get_Hessian(pt))[0] for pt in grid_data])
print(time.time() -t)

max_eig_vals_H_grid = np.max(eig_vals_H, axis=1).reshape((gridding_size, gridding_size))

plt.figure(figsize=(8,7))
plt.contourf(meshgrid[0], meshgrid[1], max_eig_vals_H_grid, 20, cmap=cm.RdBu.reversed(),
             vmin = -(np.max(np.abs(max_eig_vals_H_grid))), vmax=np.max(np.abs(max_eig_vals_H_grid)))
cb = plt.colorbar(label="Maximum eigenvalue of the Hessian $\mathbf{H}$")
plt.contour(meshgrid[0], meshgrid[1], density_matr, 20, cmap=cm.copper, alpha=.3)
critsc = plt.scatter(critical_points[idx_best_crits][:,0], critical_points[idx_best_crits][:,1],
                     s=80, c="None", edgecolor='k', label='{} best critical points'.format(N))
plt.legend()
plt.gca().invert_yaxis()

##################################


## If the gradient and Hessian have been computed before, they can be put as argument of that function to avoid
## recomputing them
t = time.time()
PiHPis_grid = []
eigval_PiHPi = []
for pt in grid_data:
    _pihpi = gapper_base.get_PiHPi(pt)
    _pihpi_eigval, _pihpi_eigvec = np.linalg.eigh(_pihpi)

    PiHPis_grid.append(_pihpi)
    eigval_PiHPi.append(_pihpi_eigval)

print(time.time() - t)


PiHPis_grid, eigval_PiHPi = np.array(PiHPis_grid), np.array(eigval_PiHPi)
max_eigval_PiHPi = np.max(eigval_PiHPi, axis=1)
max_eigval_PiHPi_resh = max_eigval_PiHPi.reshape((gridding_size, gridding_size))

plt.figure(figsize=(8,7))
pi = plt.contourf(meshgrid[0], meshgrid[1], max_eigval_PiHPi_resh, 20, cmap=cm.afmhot, extend='min')
plt.contour(meshgrid[0], meshgrid[1], density_matr, 20, cmap=cm.spring, alpha=.5)
critsc = plt.scatter(critical_points[idx_best_crits][:,0], critical_points[idx_best_crits][:,1],
                     s=80, c="None", edgecolor='w', label='{} best critical points'.format(N))
cb = plt.colorbar(pi, label="Maximum eigenvalue of $\Pi \mathbf{H} \Pi$")
plt.legend(fontsize=10)
plt.gca().invert_yaxis()

###############################

trace_PiHPi = np.trace(PiHPis_grid, axis1=1, axis2=2)
trace_PiHPi_resh = trace_PiHPi.reshape((gridding_size, gridding_size))

plt.figure(figsize=(8,7))
pi = plt.contourf(meshgrid[0], meshgrid[1], trace_PiHPi_resh, 20, cmap=cm.afmhot)
#plt.contour(meshgrid[0], meshgrid[1], density_matr, 20, cmap=cm.winter, alpha=.5)
critsc = plt.scatter(critical_points[idx_best_crits][:,0], critical_points[idx_best_crits][:,1],
                     s=80, c="None", edgecolor='k', label='{} best critical points'.format(N))
cb = plt.colorbar(pi, label="Trace of $\Pi \mathbf{H} \Pi$")
plt.legend(fontsize=10)
plt.gca().invert_yaxis()

#########################################################

t = time.time()

PiHPis_data = []
eigval_PiHPi_data = []

for pt in data:
    _pihpi = gapper_base.get_PiHPi(pt)
    _pihpi_eigval, _pihpi_eigvec = np.linalg.eigh(_pihpi)

    PiHPis_data.append(_pihpi)
    eigval_PiHPi_data.append(_pihpi_eigval)

print(time.time() - t)

max_eigval_PiHPi_data = np.max(eigval_PiHPi_data, axis=1)
threshold_eigvalPiHPi = np.percentile(eigval_PiHPi, 90)

plt.figure(figsize=(7,7))
mask_threshold_eigval = max_eigval_PiHPi_data > threshold_eigvalPiHPi
plt.scatter(data[mask_threshold_eigval][:,0], data[mask_threshold_eigval][:,1],
                     s=5, c="k")
plt.scatter(data[:,0], data[:,1],
                     s=0.5, c="grey",zorder=0)

plt.contour(meshgrid[0], meshgrid[1], density_matr, 20, cmap=cm.winter, alpha=.5)
plt.title("Points selected with threshold cut using max eigenvalue on grid")
plt.gca().invert_yaxis()

plt.figure(figsize=(7,7))
#plt.hexbin(data[:,0],data[:,1],
#           gridsize=40,mincnt=1,cmap=cm.cividis)
mask_threshold_eigval = max_eigval_PiHPi_data > threshold_eigvalPiHPi
plt.scatter(data[mask_threshold_eigval][:,0], data[mask_threshold_eigval][:,1],
                     s=5, c="r", alpha=.5)
plt.gca().invert_yaxis()



