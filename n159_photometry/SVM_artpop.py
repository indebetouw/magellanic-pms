import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import os, copy, pickle
import cmasher as cmr
#from matplotlib.colors import BoundaryNorm, ListedColormap,DivergingNorm
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
# autoset catalog path based on user
if os.environ['USER'] =='toneill':
    catalogdir = '/Users/toneill/Box/MC/HST/'
else:
    catalogdir="../../MCBox/HST/"
from sklearn import preprocessing

from load_data import load_phot


region = 'n159-all'
fuse = 'vis.ir'
full = load_phot(region=region,fuse=fuse)

####################################################
cmap_use = cmr.get_sub_cmap('inferno', 0, 0.9)  # cmr.ember, 0.2, 0.8)

train = pd.read_csv('/Users/toneill/N159/isochrones/artpop_trim_train_df.csv')#[::10]


y = np.where(train['phase'] == 'PMS',1,0)
features_vict = ['F555Wmag', 'F814Wmag']#,'F125Wmag','F160Wmag']


X = train[features_vict].to_numpy()
scaler = preprocessing.MinMaxScaler()  # .fit(X)
X_train0, X_test0, y_train, y_test = train_test_split(X, y,    test_size=0.5)  # 70% training and 30% test
X_train = scaler.fit_transform(X_train0)  # scaler.transform(X_train0)
X_test = scaler.transform(X_test0)  # scaler.transform(X_test0)

#######################

# load dereddened HTTP catalog
#photdir = '/Users/toneill/N159/photometry/'

features = ['mag_555_dered', 'mag_814_dered']#,'AvNN']
feat_title = [features[i][2::] for i in range(len(features))]
#full = full.dropna(how='any', subset=features)
X_full = full[features].to_numpy()
X_scale = scaler.transform(X_full)


grid = True
if grid:

    C_range = [2.**n for n in np.linspace(1,10,8)]#np.linspace(1,10,10) #
    gamma_range =[2.**n for n in np.linspace(2,8,8)] # np.linspace(1,100,10)#
    param_grid = {'C':C_range, 'gamma': gamma_range}
    SM = svm.SVC(kernel='rbf')
    strat = False
    if strat:
        cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3)
        grid = GridSearchCV(SM, param_grid, n_jobs=8, cv=cv, refit=True, verbose=1)
    if not strat:
        grid = GridSearchCV(SM, param_grid, n_jobs=8, refit=True, verbose=1)
    grid.fit(X_train, y_train)
    print(grid.best_params_)
    clf = svm.SVC(kernel='rbf', probability=True, C=grid.best_params_['C'], gamma=grid.best_params_['gamma'])
if not grid:
    clf = svm.SVC(kernel='rbf', probability=True , gamma=10, C=20)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

full_prob = clf.predict_proba(X_scale)

fig = plt.figure(constrained_layout=True, figsize=(12, 8))#,sharex=True,sharey=True)
gs = GridSpec(1,3, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

####### plot training set
s1 = ax1.scatter(X_train0[:, 0] - X_train0[:, 1], X_train0[:, 0], c=y_train,
                 cmap=cmap_use, s=1, vmin=0, vmax=1)
ax1.set_title('Artpop Training set')
ax1.set_xlabel('F555W - F775W')
ax1.set_ylabel('F555W [mag]')
# ax1.invert_yaxis()
ax1.set_xlim(-2, 3.5)
ax1.set_ylim(28, 10)

# ig.colorbar(s1,ax=ax1,label='P(PMS)')
####### plot testing set
s2 = ax2.scatter(X_test0[:, 0] - X_test0[:, 1], X_test0[:, 0], c=y_prob[:, 1],
                 cmap=cmap_use, s=1, vmin=0, vmax=1)
ax2.set_title('Artpop Testing set')
ax2.set_xlabel('F555W - F814W [mag]')
ax2.set_ylabel('F555W [mag]')
ax2.text(0.65, 0.9,
         'Accuracy: %.1f' % (100 * metrics.accuracy_score(y_test, y_pred)) + '%',
         transform=ax2.transAxes)
ax2.set_xlim(ax1.set_xlim()[0], ax1.set_xlim()[1])
ax2.set_ylim(ax1.set_ylim()[0], ax1.set_ylim()[1])
# fig.colorbar(s2,ax=ax2,label='P(PMS)')

s3 = ax3.scatter(X_full[:, 0] - X_full[:, 1], X_full[:, 0], c=full_prob[:, 1],
                 cmap=cmap_use, s=0.4, vmin=0, vmax=1)
ax3.set_title('Trained SVM in N159')  # on Dereddened All')
ax3.set_xlabel('F555W - F814W [mag]')
ax3.invert_yaxis()
ax3.set_ylabel('F555W [mag]')
ax3.set_xlim(ax1.set_xlim()[0], ax1.set_xlim()[1])
ax3.set_ylim(ax1.set_ylim()[0], ax1.set_ylim()[1])

fig.colorbar(s3, ax=ax3, label='P(PMS)')

for ax in [ax1, ax2, ax3]:
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(direction='in', which='both', labelsize=9)


#plt.savefig('artpop_2filt_traintest.png', dpi=250)


#############################

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
s = ax.scatter(full['ra_814'], full['dec_814'],s=0.5,alpha=0.8)#, c=full_prob[:, 1],
               #cmap=cmap_use, s=0.5, vmin=0, vmax=1, alpha=0.8)
ax.invert_xaxis()
fig.colorbar(s, ax=ax, label='P(PMS)')
ax.set_title('Artpop training set')
#plt.savefig('artpop_spatial.png', dpi=250)




