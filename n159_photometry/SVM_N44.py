import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import os, copy, pickle
#from matplotlib.colors import BoundaryNorm, ListedColormap,DivergingNorm
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
# autoset catalog path based on user
if os.environ['USER'] =='toneill':
    catalogdir = '/Users/toneill/Box/MC/HST/'
else:
    catalogdir="../../MCBox/HST/"
from sklearn import preprocessing
import cmasher as cmr



traindir = '/Users/toneill/N159/TrainingSets/'

train_full = pd.read_csv(traindir + 'MYSST_TrainingSet_final.csv')

train = train_full[['f555w_mag','f814w_mag','Av','pms_mbp']]
y = np.where(train['pms_mbp'].values >= 0.85, 1, 0)


features_vict = ['f555w_mag','f814w_mag']#,'Av']
X = train[features_vict].to_numpy()
scaler = preprocessing.MinMaxScaler()
X_train0, X_test0, y_train, y_test = train_test_split(X, y,
                                                      test_size=0.3)  # 70% training and 30% test
X_train = scaler.fit_transform(X_train0)  # scaler.transform(X_train0)
X_test = scaler.transform(X_test0)  # scaler.transform(X_test0)

#######################

# load dereddened HTTP catalog
photdir = '/Users/toneill/N159/photometry/'

features = ['mag_555', 'mag_814']#,'AvNN']
feat_title = [features[i][2::] for i in range(len(features))]

full = pd.read_csv(photdir + 'n159-all_reduce.phot.cutblue.dered.csv')
X_full = full[features].to_numpy()
X_scale = scaler.transform(X_full)



grid = False

if grid:

    SM = svm.SVC(kernel='rbf')

    C_range = np.linspace(1,100,10) #[2.**n for n in np.linspace(8,14,5)]
    gamma_range = np.linspace(1,100,10)# [2.**n for n in np.linspace(6,10,5)]
    param_grid = {'C':C_range, 'gamma': gamma_range}

    strat = False
    if strat:
        # create  K-fold with stratification training/test balanced
        # repeat to reduce variance
        cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3)
        grid = GridSearchCV(SM, param_grid, n_jobs=8, cv=cv,
                            refit=True, verbose=1)

    if not strat:
        grid = GridSearchCV(SM, param_grid, n_jobs=8,
                            refit=True, verbose=1)

    grid.fit(X_train, y_train)
    print(grid.best_params_)

    ######### Run SVM using CVd hyperparams
    clf = svm.SVC(kernel='rbf', probability=True, C=grid.best_params_['C'], gamma=grid.best_params_['gamma'])

if not grid:
    clf = svm.SVC(kernel='rbf', probability=True, gamma=10, C=20)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

full_prob = clf.predict_proba(X_scale)

cmap_use = cmr.get_sub_cmap('inferno', 0, 0.9)  # cmr.ember, 0.2, 0.8)

###########################################
plot_sum = True
if plot_sum:
    fig = plt.figure(constrained_layout=True, figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])

    ####### plot training set
    s1 = ax1.scatter(X_train0[:, 0] - X_train0[:, 1], X_train0[:, 0], c=y_train,
                     cmap=cmap_use, s=1, vmin=0, vmax=1)
    ax1.set_title('N44 Training set')
    # ax1.set_xlabel('F555W - F775W')
    ax1.set_ylabel('F555W [mag]')
    # ax1.invert_yaxis()
    ax1.set_xlim(-0.5, 4)
    ax1.set_ylim(30, 13)

    # ig.colorbar(s1,ax=ax1,label='P(PMS)')
    ####### plot testing set
    s2 = ax2.scatter(X_test0[:, 0] - X_test0[:, 1], X_test0[:, 0], c=y_prob[:, 1],
                     cmap=cmap_use, s=1, vmin=0, vmax=1)
    ax2.set_title('N44 Testing set')
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
    ax3.set_xlim(-0.5, 4)
    ax3.set_ylim(27.5, 16)

    fig.colorbar(s3, ax=ax3, label='P(PMS)')

    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(direction='in', which='both', labelsize=9)

    #plt.savefig('N44_traintest.png', dpi=250)


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
s = ax.scatter(full['ra_814'], full['dec_814'], c=full_prob[:, 1],
               cmap=cmap_use, s=0.5, vmin=0, vmax=1, alpha=0.8)
ax.invert_xaxis()
fig.colorbar(s, ax=ax, label='P(PMS)')
ax.set_title('N44 training set')
plt.savefig('N44_spatial.png', dpi=250)
