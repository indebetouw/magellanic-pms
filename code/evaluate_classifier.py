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
                        target_names=['Non-PMS','PMS']))
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# plot confusion matrix to see which labels
# likely to get confused
# e.g., how many "true" PMS stars labeled by SVM as Non-PMS
print(metrics.confusion_matrix(y_test,y_pred))

mat = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=['Non-PMS','PMS'],
            yticklabels=['Non-PMS','PMS'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.title(str(feat_title))   


from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred,normalize='true')


disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['Non-PMS','PMS'])  
fig, ax1 = plt.subplots(1,1)
disp.plot(cmap='Reds_r',ax=ax1,xticks_rotation='horizontal',include_values=True)
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top')    #ax1.grid(False)
ax1.set_xlabel('Predicted Type',fontweight='bold')
ax1.set_ylabel('True Type',fontweight='bold')
ax1.set_title('Normalized Confusion Matrix')


# Model Precision: what fraction of predicted positive outcomes are correct?
# (ratio of true positives to total # of positives)
#print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what fraction of originally labeled PMS (true positives + false negatives)
# are correctly identified? (ratio of true positives to (true pos + false negative))
#print("Recall:",metrics.recall_score(y_test, y_pred))

# inspect support vectors
#clf.support_vectors_
# get indices of support vectors
#clf.support_
# get num of SVs for each class
#clf.n_support_

### plot ROC (receiver operating characteristic) curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

### ROC AUC (area under curve) can compare classifier performances
# best possible score is 1, completely random classifier has ROC AUC = 0.5
# (percent of ROC plot underneath the curve)
ROC_AUC = roc_auc_score(y_test, y_pred)
print('ROC AUC : {:.4f}'.format(ROC_AUC))        

plt.figure()
plt.plot(fpr, tpr, linewidth=2,c='cornflowerblue')
plt.plot([0,1], [0,1], 'k--' )
plt.title('ROC curve for '+str(feat_title)+', ROC AUC : {:.4f}'.format(ROC_AUC))
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

#############################

from sklearn.inspection import permutation_importance
# this gives relative importance of features in THIS PARTICULAR MODEL
# - not intrinsticallly over all models!!! 
    #https://scikit-learn.org/stable/modules/permutation_importance.html

perm_importance = permutation_importance(clf, X_test, y_test,n_repeats=30)
#sorted_idx = perm_importance.importances_mean.argsort()

plt.figure()
plt.barh(features, perm_importance.importances_mean,facecolor='cornflowerblue')
plt.title("Relative Importance of Features in RBF SVM")  
plt.xlabel('Permutation Importance')


#######################3 plot decision regions

from mlxtend.plotting import plot_decision_regions
#scatter_kwargs = {'s':0.5}#,'alpha':0.7}
#plot_decision_regions(X_train,y_train,clf=clf)#,scatter_kwargs=scatter_kwargs)    
#plt.gca().invert_yaxis() 

zero_inds, = np.where(train['pms_membership_prob'].values ==0)

scatter_highlight_kwargs = {'s': 3.5, 'label': 'P(PMS)=0', 'alpha': 0.7}    
scatter_kwargs = {'s': 3, 'edgecolor': None, 'alpha': 0.4}

# Plotting decision regions with more than 2 features
fig, axarr = plt.subplots(2, 2, figsize=(10,8), sharex=True, sharey=True)
values = [1,1.5,2,3]
width = 0.5
for value, ax in zip(values, axarr.flat):
    plot_decision_regions(X, y, clf=clf,
                          X_highlight=X[zero_inds],
                          filler_feature_values={2: value},
                          filler_feature_ranges={2: width},
                          legend=2, ax=ax,scatter_highlight_kwargs=scatter_highlight_kwargs,
                  scatter_kwargs=scatter_kwargs)
    ax.set_xlabel(feat_title[0])
    ax.set_ylabel(feat_title[1])
    ax.set_title('$A_v$ = %.1f'%value+' $\pm$ %.1f'%width)

# Adding axes annotations
#fig.suptitle('SVM on make_blobs')
plt.show()  
axarr[0][0].set_xlabel('')
axarr[0][1].set_xlabel('')
for i,j in [[0,0],[0,1],[1,0],[1,1]]:
    print(i,j)
    handles, labels = axarr[i][j].get_legend_handles_labels()
    axarr[i][j].legend(handles, 
              ['Non-PMS','PMS','P(PMS)=0'], 
               framealpha=0.3, scatterpoints=1)



def plot_svc_decision_function(model, ax=None, plot_support=True):
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
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    return
