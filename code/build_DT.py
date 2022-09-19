#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 15:44:23 2021

@author: toneill
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import os, copy, pickle
#from make_2dKDE import twoD_kde
#from make_Hess import hess_bin
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from six import StringIO
from IPython.display import Image  
import pydotplus



from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
# autoset catalog path based on user
if os.environ['USER'] =='toneill':
    catalogdir = '/Users/toneill/Box/MC/HST/'
else:
    catalogdir="../../MCBox/HST/"

'''
#################################################################

build_DT: Script to create, execute, and test results of a
       decision tree classifier (DT)

#################################################################
'''

# whether to include only longer wavelengths in training
long = False
full = True
extin = False

if long:
    features = ['m_f775w','m_f110w','m_f160w']
if full:
    features = ['m_f555w','m_f775w','m_f110w','m_f160w']
if extin:
    features = ['m_f110w','m_f160w','A_v']

feat_title = [features[i][2::] for i in range(len(features))]
############# 
# Load and clean training set 
train_full = pd.read_csv(catalogdir+'Ksoll2018_training_set.csv')
train = copy.deepcopy(train_full)
# drop any entries with missing mag estimates - revisit later to be less strict?
train = train.dropna(how='any',subset=features)
for m in ['m_f110w','m_f160w']:
    train.drop(train[train[m]>30].index,inplace=True)
    
    
# Split into training & testing sets to make SVM
y = np.where(train['pms_membership_prob'].values >= 0.9, 1, 0)


X = train[features].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                    test_size=0.3) # 70% training and 30% test




    
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
    

path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
print(
    "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]
    )
)

##### FOR THIS EXAMPLE: REMOVE optimal 1 node tree since trivial
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()


    
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) 
    
    
fig = plt.figure(figsize=(25,20))
plot_tree(clf, feature_names=features,  
                   class_names=['Non-PMS','PMS'],
                   filled=True)    
    
    
    
    
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,
                feature_names = feat_title,
                class_names=['Non-PMS','PMS'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('DT_firstTest.png')
Image(graph.create_png())
    
