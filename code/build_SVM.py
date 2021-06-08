
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import os, copy, pickle
from make_2dKDE import twoD_kde
from make_Hess import hess_bin
# autoset catalog path based on user
if os.environ['USER'] =='toneill':
    catalogdir = '/Users/toneill/Box/MC/HST/'
else:
    catalogdir="../../MCBox/HST/"

'''
#################################################################

build_SVM: Script to create, execute, and test results of a
        Support Vector Machine (SVM)
        
To Dos include:
    - cross validation for cost/sigma kernel parameters
    - improve performance metrics analyzing & visualization
    - update so using de-reddened data

#################################################################
'''

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


if __name__ == '__main__': 
    
    ############# 
    # Load and clean training set 
    train_full = pd.read_csv(catalogdir+'Ksoll2018_training_set.csv')
    train = copy.deepcopy(train_full)
    # drop any entries with missing mag estimates - revisit later to be less strict?
    train = train.dropna(how='any',subset=['m_f555w','m_f775w','m_f110w','m_f160w'])
    for m in ['m_f110w','m_f160w']:
        train.drop(train[train[m]>30].index,inplace=True)
    
    ########### 
    # Replicate Fig 16 in Ksoll+ 2018 with entire training set
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(111)
    s=ax1.scatter(train['m_f555w']-train['m_f775w'],train['m_f555w'],
                c=train['pms_membership_prob'],
                cmap='RdYlBu_r',s=0.7)
    ax1.invert_yaxis()
    ax1.set_xlabel('F555W - F775W')
    ax1.set_ylabel('F555W')
    ax1.set_xlim(-1.3,3.4)
    ax1.set_ylim(26.8,18.7)
    fig.colorbar(s,label='P(PMS)')
    ax1.set_title('R136 Training Set')
    fig.tight_layout()
    
    ###########
    # Split into training & testing sets to make SVM
    y = np.where(train['pms_membership_prob'].values >= 0.85, 1, 0)
    X = train[['m_f555w','m_f775w','m_f110w','m_f160w']].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                        test_size=0.3) # 70% training and 30% test
     
    ######### 
    # Build SVM
    #       - kernel choices include linear, poly, and radial basis fxn
    #       - C is cost function (~error tolerance)
    clf = svm.SVC(kernel='rbf',probability=True)#C=1e6,
                                      
    # Train the model using the training sets
    clf.fit(X_train, y_train)
    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    
    ######### 
    # Calculate performance metrics
    print(metrics.confusion_matrix(y_test,y_pred))
    print(metrics.classification_report(y_test,y_pred))
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:",metrics.precision_score(y_test, y_pred))
    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:",metrics.recall_score(y_test, y_pred))

    # inspect support vectors
    #clf.support_vectors_
    # get indices of support vectors
    #clf.support_
    # get num of SVs for each class
    #clf.n_support_
    
    #########
    # Visualize results
        
    fig, [ax1,ax2] = plt.subplots(2,1,figsize=(6,8),sharex=True,sharey=True)
    # plot training set
    s1=ax1.scatter(X_train[:, 0]-X_train[:,1], X_train[:, 0], c=y_train, 
                cmap='RdYlBu_r',s=0.7)  
    ax1.set_title('R136 Training set')
    ax1.set_ylabel('F555W [mag]')
    fig.colorbar(s1,ax=ax1,label='P(PMS)')
    # plot testing set
    s2=ax2.scatter(X_test[:, 0]-X_test[:, 1], X_test[:, 0], c=y_prob[:,1], 
                cmap='RdYlBu_r',s=0.7)  
    ax2.set_title('R136 Testing set')
    ax2.set_xlabel('F555W - F775W [mag]')
    ax2.set_ylabel('F555W [mag]')
    ax2.text(0.65,0.9,
             'Accuracy: %.1f'%(100*metrics.accuracy_score(y_test, y_pred))+'%',
             transform=ax2.transAxes)
    ax2.invert_yaxis()
    fig.colorbar(s2,ax=ax2,label='P(PMS)')
        
    ########################################################################
    # Run SVM on entirety of 30Dor data
    
    # load and clean data
    fullcatname=catalogdir+"HTTP.2015_10_20.1.astro"
    cat_full=pickle.load(open(fullcatname+".pkl",'rb')).to_pandas()
    
    #k2018_pms = pd.read_csv(catalogdir+'Ksoll2018_HTTP_PMS_catalogue.csv')
    #k2018_ums = pd.read_csv(catalogdir+'Ksoll2018_HTTP_UMS_selection.csv')
    #cat_full = pd.concat([k2018_pms,k2018_ums], axis=0, ignore_index=True)
    
    full = copy.deepcopy(cat_full)
    # drop any entries with missing mag estimates - revisit later to be less strict?
    full = full.dropna(how='any',subset=['m_f555w','m_f775u','m_f110w','m_f160w'])
    for m in ['m_f110w','m_f160w','m_f555w','m_f775u']:
        full.drop(full[full[m]>30].index,inplace=True)  
    
    # predict PMS probabilities
    X_full = full[['m_f555w','m_f775u','m_f110w','m_f160w']].to_numpy()
    full_prob = clf.predict_proba(X_full)
    
    
    #########
    # Visualize results
    
    fig = plt.figure(constrained_layout=True,figsize=(12,8))
    gs = GridSpec(2,2, figure=fig)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[:,1])
    
    ####### plot training set
    s1=ax1.scatter(X_train[:, 0]-X_train[:,1], X_train[:, 0], c=y_train, 
                cmap='RdYlBu_r',s=0.7)  
    ax1.set_title('R136 Training set')
    #ax1.set_xlabel('F555W - F775W')
    ax1.set_ylabel('F555W [mag]')
    ax1.invert_yaxis()

    #ig.colorbar(s1,ax=ax1,label='P(PMS)')
    ####### plot testing set
    s2=ax2.scatter(X_test[:, 0]-X_test[:, 1], X_test[:, 0], c=y_prob[:,1], 
                cmap='RdYlBu_r',s=0.7)  
    ax2.set_title('R136 Testing set')
    ax2.set_xlabel('F555W - F775W [mag]')
    ax2.set_ylabel('F555W [mag]')
    ax2.text(0.65,0.9,
             'Accuracy: %.1f'%(100*metrics.accuracy_score(y_test, y_pred))+'%',
             transform=ax2.transAxes)
    ax2.invert_yaxis()
    ax2.set_xlim(ax1.set_xlim()[0],ax1.set_xlim()[1])
    ax2.set_ylim(ax1.set_ylim()[0],ax1.set_ylim()[1])
    #fig.colorbar(s2,ax=ax2,label='P(PMS)')
    
    s3=ax3.scatter(X_full[:, 0]-X_full[:, 1], X_full[:, 0], c=full_prob[:,1], 
                cmap='RdYlBu_r',s=0.7)  
    ax3.set_title('30Dor')
    ax3.set_xlabel('F555W - F775W [mag]')
    ax3.invert_yaxis()
    ax3.set_ylabel('F555W [mag]')
    fig.colorbar(s3,ax=ax3,label='P(PMS)')
    
    ########################################################################
    # Compare new SVM results to Ksoll2018 results
    
    
    
    plt.figure()
    plt.scatter(full['p_svm'],full_prob[:,1])
    
    
    
    
    
    
    
    
    
    
    
    
    
    