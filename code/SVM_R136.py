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

train_full = pd.read_csv(catalogdir+'Ksoll2018_training_set.csv')


########################## DE REDDEN
# de Marchi 2016 extinction law
# https://academic.oup.com/mnras/article/455/4/4373/1264525
R_BV = [4.48, 3.03, 1.83, 1.22]  # at V,I,J,H
# R_BV = [4.48, 3.74, 1.83, 1.22] # at V,R,J,H
label = ['f555w', 'f775u', 'f110w', 'f160w']

tr_av0 = train_full['A_v']
train_full['m_f555w_dered'] = train_full['m_f555w'] - tr_av0 * R_BV[0] / R_BV[0]
train_full['m_f775w_dered'] = train_full['m_f775w'] - tr_av0 * R_BV[1] / R_BV[0]
train_full['m_f110w_dered'] = train_full['m_f110w'] - tr_av0 * R_BV[2] / R_BV[0]
train_full['m_f160w_dered'] = train_full['m_f160w'] - tr_av0 * R_BV[3] / R_BV[0]


###############################

train = copy.deepcopy(train_full)
train = train.dropna(how='any', subset=['m_f555w', 'm_f775w','m_f110w','m_f160w'])  # features)
for m in ['m_f555w', 'm_f775w','m_f110w','m_f160w']:
    train.drop(train[train[m] > 30].index, inplace=True)
train = train.reset_index(drop=False)

train['m_f814w_dered'] = -0.032 + 0.917*train['m_f775w_dered'] + \
                         0.247*train['m_f110w_dered'] - 0.106*train['m_f160w_dered'] -0.056*train['m_f555w_dered']
train['m_f125w_dered'] = 0.033 -0.120*train['m_f775w_dered'] + \
                         0.884*train['m_f110w_dered'] + 0.207*train['m_f160w_dered'] + 0.027*train['m_f555w_dered']



###########
# Split into training & testing sets to make SVM
y = np.where(train['pms_membership_prob'].values >= 0.85, 1, 0)

features_vict = ['m_f555w_dered', 'm_f814w_dered']

from sklearn import preprocessing

X = train[features_vict].to_numpy()
scaler = preprocessing.MinMaxScaler()  # .fit(X)
X_train0, X_test0, y_train, y_test = train_test_split(X, y,
                                                      test_size=0.3)  # 70% training and 30% test
X_train = scaler.fit_transform(X_train0)  # scaler.transform(X_train0)
X_test = scaler.transform(X_test0)  # scaler.transform(X_test0)


#######################

# load dereddened HTTP catalog
photdir = '/Users/toneill/N159/photometry/'
features = ['mag_555_dered', 'mag_814_dered']#,'AvNN']
feat_title = [features[i][2::] for i in range(len(features))]
full = pd.read_csv(photdir + 'n159-all_reduce.phot.cutblue.dered.csv')
#full = full.dropna(how='any', subset=features)

X_full = full[features].to_numpy()
X_scale = scaler.transform(X_full)


SM = svm.SVC(kernel='rbf')

grid = True

if grid:

    C_range = np.linspace(1,10,10) #[2.**n for n in np.linspace(8,14,5)]
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

    clf = svm.SVC(kernel='rbf', probability=True, C=grid.best_params_['C'], gamma=grid.best_params_['gamma'])
if not grid:
    clf = svm.SVC(kernel='rbf', probability=True , gamma=10, C=20)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

full_prob = clf.predict_proba(X_scale)

cmap_use = cmr.get_sub_cmap('inferno', 0, 0.9)  # cmr.ember, 0.2, 0.8)

fig = plt.figure(constrained_layout=True, figsize=(12, 8))
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[:, 1])

####### plot training set
s1 = ax1.scatter(X_train0[:, 0] - X_train0[:, 1], X_train0[:, 0], c=y_train,
                 cmap=cmap_use, s=1, vmin=0, vmax=1)
ax1.set_title('R136 Training set')
# ax1.set_xlabel('F555W - F775W')
ax1.set_ylabel('F555W [mag]')
# ax1.invert_yaxis()
ax1.set_xlim(-1, 3.5)
ax1.set_ylim(30, 13)

# ig.colorbar(s1,ax=ax1,label='P(PMS)')
####### plot testing set
s2 = ax2.scatter(X_test0[:, 0] - X_test0[:, 1], X_test0[:, 0], c=y_prob[:, 1],
                 cmap=cmap_use, s=1, vmin=0, vmax=1)
ax2.set_title('R136 Testing set')
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
ax3.set_ylim(27.5, 16)

fig.colorbar(s3, ax=ax3, label='P(PMS)')

for ax in [ax1, ax2, ax3]:
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(direction='in', which='both', labelsize=9)


plt.savefig('r136_traintest.png', dpi=250)


#############################

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
s = ax.scatter(full['ra_814'], full['dec_814'], c=full_prob[:, 1],
               cmap=cmap_use, s=0.5, vmin=0, vmax=1, alpha=0.8)
ax.invert_xaxis()
fig.colorbar(s, ax=ax, label='P(PMS)')
ax.set_title('R136 training set')
plt.savefig('R136_spatial.png', dpi=250)



