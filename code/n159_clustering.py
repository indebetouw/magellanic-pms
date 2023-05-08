from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
from sklearn.neighbors import NearestNeighbors
from skimage import measure

alma_co = fits.open('/Users/toneill/N159/alma/12CO_combined.regrid.gtr0.2K.maximum.fits')

min_samples = 75

x_pms = full['ra_814'][pms_inds]
y_pms = full['dec_814'][pms_inds]

'''
neight = NearestNeighbors(n_neighbors=min_samples)
nbrs = neight.fit(pd.DataFrame([x_pms,y_pms]).T)
distances,inds = nbrs.kneighbors(pd.DataFrame([x_pms,y_pms]).T)
distances = np.sort(distances,axis=0)
#distances = distances[:,1]

# Choose approximate point where curve becomes smooth - currently
# by eye, could be improved
plt.figure()
plt.plot(distances[:,1])#,marker='v',markersize=3)
plt.xlabel('Source Number')
plt.ylabel('Nearest Neighbor Distance')
plt.title('Nearest Neighbor Distances of Sources')'''

# set up plotting colors for dbscan
colors = ['royalblue', 'tab:orange', 'orangered', 'tab:red', 'm', \
          'hotpink', 'tab:cyan', 'tab:purple', 'gold',
          'limegreen', 'deeppink', 'slateblue','maroon','cornflowerblue',
          'firebrick','navy',  'r', 'darkgrey']
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])

min_samples = 100
eps= 0.0065
run_dbscan(min_samples=30,eps=0.004)

def run_dbscan(min_samples=100,eps=0.005):
    m = DBSCAN(eps=eps,min_samples=min_samples)
    m.fit(pd.DataFrame([x_pms,y_pms]).T)
    clusters = m.labels_
    #print(np.unique(clusters))
    print(len(np.unique(clusters)))

    fig = plt.figure(figsize=(10,10))#figsize=(13.5, 6.5))
    ax = fig.add_subplot(111, projection=wcs.WCS(alma_co[0].header))
    s = ax.scatter(x_pms,y_pms,   c = vectorizer(clusters), s = 0.5,
                   transform=ax.get_transform('fk5'))
    ax.set_xlabel('RA', fontsize=12)
    ax.set_ylabel('Dec', fontsize=12, labelpad=0)
    ax.set_xlim(110, 335)
    ax.set_ylim(-97, 240)
    ax.contour(alma_co[0].data, levels=[5], colors=['k'], zorder=3,lws=[1])
    alma_cont = np.where(alma_co[0].data >= -100000, 0, 1)
    contours = measure.find_contours(alma_cont, 0.9)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], c='k', lw=1)  # ,ls=':')






