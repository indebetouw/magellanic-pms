#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 10:14:00 2021

@author: toneill
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

'''
#################################################################

evaluate_classifier: grab bag of ways to assess/visualize 
                    a fit classifier
                    
!!! currently need to have run build_SVM.py already in same session !!!

#################################################################
'''

######### 
# Calculate performance metrics
print()
print(classification_report(y_test, y_pred,
                            target_names=['Non-PMS', 'PMS']))
# Model Accuracy: how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# plot confusion matrix to see which labels
# likely to get confused
# e.g., how many "true" PMS stars labeled by SVM as Non-PMS
print(metrics.confusion_matrix(y_test, y_pred))

mat = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=['Non-PMS', 'PMS'],
            yticklabels=['Non-PMS', 'PMS'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.title(str(feat_title))

from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-PMS', 'PMS'])

cmap_use2 = cmr.get_sub_cmap('inferno_r', 0.2, 0.8)

cmap_use2 = cmr.get_sub_cmap('inferno_r', 0.2, 0.8)

fig, ax1 = plt.subplots(1, 1)
disp.plot( ax=ax1, xticks_rotation='horizontal', include_values=True, colorbar=False)
# ax1.xaxis.tick_top()
# ax1.xaxis.set_label_position('top')    #ax1.grid(False)
ax1.set_xlabel('Predicted Type', fontsize=14, labelpad=10)  # ,fontweight='bold')
ax1.set_ylabel('True Type', fontsize=14)  # ,fontweight='bold')
# ax1.set_title('Normalized Confusion Matrix')
fig.tight_layout()

# Model Precision: what fraction of predicted positive outcomes are correct?
# (ratio of true positives to total # of positives)
# print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what fraction of originally labeled PMS (true positives + false negatives)
# are correctly identified? (ratio of true positives to (true pos + false negative))
# print("Recall:",metrics.recall_score(y_test, y_pred))

# inspect support vectors
# clf.support_vectors_
# get indices of support vectors
# clf.support_
# get num of SVs for each class
# clf.n_support_

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.collections as mcoll


##################################################################

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments


def colorline(x, y, z=None, ax=None, cmap=plt.get_cmap('copper'),
              norm=plt.Normalize(0.0, 1.0), lw=8, alpha=1, zo=1):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    # pdb.set_trace()

    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidths=lw, alpha=alpha, zorder=zo)
    ax.add_collection(lc)

    return lc


### plot ROC (receiver operating characteristic) curve

fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])

# get the best threshold
J = tpr - fpr
ix = np.argmax(J)
best_thresh = thresholds[ix]
print('Best Threshold=%f' % (best_thresh))

plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')

### ROC AUC (area under curve) can compare classifier performances
# best possible score is 1, completely random classifier has ROC AUC = 0.5
# (percent of ROC plot underneath the curve)
ROC_AUC = roc_auc_score(y_test, y_pred)
print('ROC AUC : {:.4f}'.format(ROC_AUC))

fig, [ax2, ax1] = plt.subplots(2, 1, figsize=(4, 8))  # 12,5))
disp.plot(cmap='inferno', ax=ax1, xticks_rotation='horizontal', include_values=True)
# ax1.xaxis.tick_top()
# ax1.xaxis.set_label_position('top')    #ax1.grid(False)
ax1.set_xlabel('Predicted Type', fontsize=13, labelpad=10)  # ,fontweight='bold')
ax1.set_ylabel('True Type', fontsize=13, labelpad=-15)  # ,fontweight='bold')
# ax1.set_title('Normalized Confusion Matrix')
# Get the images on an axis
im = ax1.images
# Assume colorbar was plotted last one plotted last
cb = im[-1].colorbar
cb.remove()

ax2.plot(fpr, tpr, linewidth=3, c='palevioletred', label=f'SVM, AUC = {ROC_AUC:.3f}')
ax2.plot([0, 1], [0, 1], c='grey', ls='--', label='Random, AUC = 0.5')
# plt.title('ROC curve for '+str(feat_title)+', ROC AUC : {:.4f}'.format(ROC_AUC))
ax2.set_xlabel('False Positive Rate', fontsize=13)  # (1 - Specificity)')
ax2.set_ylabel('True Positive Rate', fontsize=13)  # (Sensitivity)')
ax2.legend(loc='lower right', fontsize=12)

for ax in [ax2]:
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(direction='in', which='both', labelsize=10)


#########################################

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay







prec, recall, thresh_prec = precision_recall_curve(y_test, y_prob[:, 1])  # , pos_label=clf.classes_[1])
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()

#fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(6, 8))
fig,ax1 = plt.subplots(1, 1, figsize=(9,7))#(6, 8))
ss = colorline(fpr, tpr, z=thresholds, cmap=cmap_use, ax=ax1,lw=7)
ax1.plot([0, 1], [0, 1], c='grey', ls='--', label='Random, AUC = 0.5',zorder=0)
ax1.set_xlabel('False Positive Rate', fontsize=16)  # (1 - Specificity)')
ax1.set_ylabel('True Positive Rate', fontsize=16,labelpad=20)  # (Sensitivity)')
ax1.text(0.65, 0.03, f'AUC = {ROC_AUC:.3f}', fontsize=20,
         bbox=dict(facecolor='none', edgecolor='k'))
ax1.set_aspect('equal')
ax1.xaxis.set_ticks_position('both')
ax1.yaxis.set_ticks_position('both')
ax1.tick_params(direction='in', which='both', labelsize=10)

cbar = fig.colorbar(ss,label='Threshold')
cbar.set_label('Threshold',fontsize=14)

plt.tight_layout()

plt.savefig('roc_auc.png',dpi=300)
#colorline(recall, prec, z=thresh_prec, cmap=cmap_use, ax=ax2)
#ax2.set_xlabel('Recall', fontsize=13)
#ax2.set_ylabel('Precision', fontsize=13)

#fig.tight_layout()
fig.subplots_adjust(left=0.01)
cbar_ax = fig.add_axes([0.89,0.115,0.03,0.86])
cbar = fig.colorbar(ss, cax=cbar_ax,label='Threshold')
cbar.ax.yaxis.set_tick_params(labelsize=8)


fig.subplots_adjust(top=0.9)
cbar_ax = fig.add_axes([0.115,0.92,0.84,0.025])
cbar = fig.colorbar(ss, cax=cbar_ax,orientation='horizontal')
cbar.ax.xaxis.set_ticks_position("top")
cbar.set_label(label='Threshold', fontsize=10)
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.set_tick_params(labelsize=8)




#############################

from sklearn.inspection import permutation_importance

# this gives relative importance of features in THIS PARTICULAR MODEL
# - not intrinsticallly over all models!!! 
# https://scikit-learn.org/stable/modules/permutation_importance.html

perm_importance = permutation_importance(clf, X_test, y_test, n_repeats=30)
# sorted_idx = perm_importance.importances_mean.argsort()

plt.figure()
plt.barh(features, perm_importance.importances_mean, facecolor='cornflowerblue')
plt.title("Relative Importance of Features in RBF SVM")
plt.xlabel('Permutation Importance')

#######################3 plot decision regions

from mlxtend.plotting import plot_decision_regions

# scatter_kwargs = {'s':0.5}#,'alpha':0.7}
# plot_decision_regions(X_train,y_train,clf=clf)#,scatter_kwargs=scatter_kwargs)
# plt.gca().invert_yaxis()

zero_inds, = np.where(train['pms_membership_prob'].values == 0)

scatter_highlight_kwargs = {'s': 3.5, 'label': 'P(PMS)=0', 'alpha': 0.7}
scatter_kwargs = {'s': 3, 'edgecolor': None, 'alpha': 0.4}

# Plotting decision regions with more than 2 features
fig, axarr = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
values = [1, 1.5, 2, 3]
width = 0.5
for value, ax in zip(values, axarr.flat):
    plot_decision_regions(X, y, clf=clf,
                          X_highlight=X[zero_inds],
                          filler_feature_values={2: value},
                          filler_feature_ranges={2: width},
                          legend=2, ax=ax, scatter_highlight_kwargs=scatter_highlight_kwargs,
                          scatter_kwargs=scatter_kwargs)
    ax.set_xlabel(feat_title[0])
    ax.set_ylabel(feat_title[1])
    ax.set_title('$A_v$ = %.1f' % value + ' $\pm$ %.1f' % width)

# Adding axes annotations
# fig.suptitle('SVM on make_blobs')
plt.show()
axarr[0][0].set_xlabel('')
axarr[0][1].set_xlabel('')
for i, j in [[0, 0], [0, 1], [1, 0], [1, 1]]:
    print(i, j)
    handles, labels = axarr[i][j].get_legend_handles_labels()
    axarr[i][j].legend(handles,
                       ['Non-PMS', 'PMS', 'P(PMS)=0'],
                       framealpha=0.3, scatterpoints=1)


def plot_svc_decision_function(model, ax=None, plot_support=True, color='k'):
    """
    Plot decision function for a SVC trained on only two features
    
    Based on: https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html
    """

    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors=color,
               levels=[-1, 0, 1], alpha=1,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return
