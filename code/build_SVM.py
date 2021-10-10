
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import os, copy, pickle
#from make_2dKDE import twoD_kde
#from make_Hess import hess_bin
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from astropy.table import Table
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
# autoset catalog path based on user
if os.environ['USER'] =='toneill':
    catalogdir = '/Users/toneill/Box/MC/HST/'
else:
    catalogdir="../../MCBox/HST/"
from kNN_extinction_map import kNN_regr_Av #kNN_extinction, 


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
    
    # whether to include only longer wavelengths in training
    long = False
    full = False
    extin = False
    dered = True
    dered_full=False
    simp=False
    
    if simp:
        features = ['m_f555w_dered','m_f775w_dered']
    if long:
        features = ['m_f775w','m_f110w','m_f160w']
    if full:
        features = ['m_f555w','m_f775w','m_f110w','m_f160w']#,'A_v']
    if extin:
        features = ['m_f110w','m_f160w','A_v']
    if dered:
        features = ['m_f555w_dered','m_f775w_dered','A_v']#'m_f110w_dered','m_f160w_dered']#,'A_v']
    if dered_full:
        features = ['m_f555w_dered','m_f775w_dered','m_f110w_dered','m_f160w_dered']
    
    feat_title = [features[i][2::] for i in range(len(features))]
    ############# 
    # Load and clean training set 
    
    ########################## DE REDDEN 
    # de Marchi 2016 extinction law
    # https://academic.oup.com/mnras/article/455/4/4373/1264525
    R_BV = [4.48, 3.03, 1.83, 1.22] # at V,I,J,H
    #R_BV = [4.48, 3.74, 1.83, 1.22] # at V,R,J,H
    label = ['f555w','f775u', 'f110w', 'f160w']
    
    train_full = pd.read_csv(catalogdir+'Ksoll2018_training_set.csv')
    trCoords = [[train_full['Ra'].values[i],train_full['Dec'].values[i]] for i in range(len(train_full))]
    #tr_av0 = kNN_regr_Av(UCoords,trCoords,UAV,eps,nnear,ncore=7)
    tr_av0 = train_full['A_v']
    train_full['AvNN'] = tr_av0
    ## note that values getting from NN Av here vary slightly from Ksoll training Avs
    train_full['m_f555w_dered'] = train_full['m_f555w'] - tr_av0*R_BV[0]/R_BV[0]
    train_full['m_f775w_dered'] = train_full['m_f775w'] - tr_av0*R_BV[1]/R_BV[0]
    train_full['m_f110w_dered'] = train_full['m_f110w'] - tr_av0*R_BV[2]/R_BV[0]
    train_full['m_f160w_dered'] = train_full['m_f160w'] - tr_av0*R_BV[3]/R_BV[0]    
    
    train = copy.deepcopy(train_full)
    # drop any entries with missing mag estimates - revisit later to be less strict?
    train = train.dropna(how='any',subset=features)
    for m in ['m_f110w','m_f160w','m_f555w','m_f775w']:
        train.drop(train[train[m]>30].index,inplace=True)
    train = train.reset_index(drop=False)
    
    #################
    # plot comparison of training sets with/without p=0
    '''
    plot1 = 1
    plot2 = 3
    feat1 = features[plot1]
    feat2 = features[plot2]    
    
    
    fig,[ax1,ax2] = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8,6))
    norm = DivergingNorm(vmin=0, vcenter=0.5,vmax=1) 

    s=ax1.scatter(train[feat1]-train[feat2],
                  train[feat1],
                c=train['pms_membership_prob'],
                cmap='RdYlBu_r',s=0.5,norm=norm)
    ax1.invert_yaxis()
    ax1.set_xlabel(feat1+' - '+feat2)
    ax1.set_ylabel(feat1)
    ax1.set_xlim(-1.3,3.4)
    #ax1.set_ylim(26.8,18.7)
    fig.colorbar(s,label='P(PMS)')
    ax1.set_title('All training entries with P=0')
    fig.tight_layout()
    
    #########################3    
    train.drop(train[train['pms_membership_prob']==0].index,inplace=True)
    train=train.reset_index(drop=True)
    ##################  

    s2=ax2.scatter(train[feat1]-train[feat2],
                  train[feat1],
                c=train['pms_membership_prob'],
                cmap='RdYlBu_r',s=0.5,norm=norm)
    ax2.set_xlabel(feat1+' - '+feat2)
    ax2.set_ylabel(feat1)
    ax2.set_title('All training entries without P=0')
    
    #pick_df = pickle.dump(train,open('trimmed_ksoll_training.p','wb')) 
    #train[['m_f555w', 'm_f775w', 'm_f110w', 'm_f160w', 'A_v',
    #       'pms_membership_prob']].to_csv('trimmed_ksoll_training.csv',index=False)
    
    
    
    train1 = copy.deepcopy(train)
    train.drop(train[train['pms_membership_prob']==0].index,inplace=True)
    train2=train.reset_index(drop=True)
    
    from glue import qglue
    qglue(train0=train1,train_n0=train2)
    '''
    
    
    
    ########### 
    # Replicate Fig 16 in Ksoll+ 2018 with entire training set
    '''
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(111)
    s=ax1.scatter(train['m_f555w_dered']-train['m_f775w_dered'],
                  train['m_f555w_dered'],
                c=train['pms_membership_prob'],
                cmap='RdYlBu_r',s=0.3)
    ax1.invert_yaxis()
    ax1.set_xlabel('F555W - F775W')
    ax1.set_ylabel('F555W')
    ax1.set_xlim(-1.3,3.4)
    ax1.set_ylim(26.8,18.7)
    fig.colorbar(s,label='P(PMS)')
    ax1.set_title('R136 Training Set')
    fig.tight_layout()
    '''
    
    
    
    
    #plt.style.use('ggplot')
    ###########
    # Split into training & testing sets to make SVM
    y = np.where(train['pms_membership_prob'].values >= 0.9, 1, 0)
    
    scale = False
    
    if scale:
        
        ### doesn't seem to impact accuracy much
        
        from sklearn import preprocessing   
        
        X = train[features].to_numpy()
        #scaler = preprocessing.StandardScaler().fit(X)
        scaler = preprocessing.MinMaxScaler().fit(X)  
        #scaler = preprocessing.RobustScaler().fit(X)
        #scaler = preprocessing.Normalizer().fit(X)
        
        X_train0, X_test0, y_train, y_test = train_test_split(X, y, 
                            test_size=0.3) # 70% training and 30% test
        
        X_train = scaler.transform(X_train0)
        X_test = scaler.transform(X_test0)
        
        
    if not scale:
        
        X = train[features].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.3) # 70% training and 30% test
        X_train0 = X_train
        X_test0 = X_test
    
    ###############################################################
    # Build SVM
    #       - kernel choices include linear, poly, sigmoid, and radial basis fxn (rbf)
    ######### Cross validate hyperparameters
    #       - C is cost function (~error tolerance)
    #       - gamma is
    #  - could also add kernel choice to this but would increase comp time
    
    # instantiate SVM with default hyperparams and no prob. calc
    # to reduce comp time
    SM = svm.SVC(kernel='rbf')
    param_grid = {'C':[2.**n for n in np.linspace(4,13,6)],
                  'gamma':[2.**n for n in np.linspace(-3,6,6)]}
         
    '''
    - NOTE: n_jobs controls how many cores to use,
        most of the time works but has a known bug where sometimes
        can result in a "broken pipe" error
    - if encounter, just try running it again and it normally resolves itself
    '''     
    strat = False
    if strat:
        # create  K-fold with stratification training/test balanced
        # repeat to reduce variance
        cv = RepeatedStratifiedKFold(n_splits=3,n_repeats=3)         
        grid = GridSearchCV(SM, param_grid, n_jobs=7, cv=cv,
                            refit=True,verbose=1) 
        
    if not strat:
        grid = GridSearchCV(SM, param_grid, n_jobs=8, 
                            refit=True,verbose=1) 
        
    grid.fit(X_train, y_train)
    print(grid.best_params_)    
        
    #### examine the best model
    #print()
    ## best score achieved during the GridSearchCV
    #print('GridSearch CV best score : {:.4f}\n\n'.format(grid.best_score_))
    # print parameters that give the best results
    #print('Parameters that give the best results :','\n\n', (grid.best_params_))
    # print estimator that was chosen by the GridSearch
    #print('\n\nEstimator that was chosen by the search :','\n\n', (grid.best_estimator_))
    
    ######### Run SVM using CVd hyperparams
    clf = svm.SVC(kernel='rbf',probability=True,C=grid.best_params_['C'], gamma=grid.best_params_['gamma'])
    
    # Train the model using the training sets
    clf.fit(X_train, y_train)
    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    
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

    
    ########################################################################
    # Run SVM on entirety of 30Dor data
    
    # load and clean data
    #fullcatname=catalogdir+"HTTP.2015_10_20.1.astro"
    #cat_full=pickle.load(open(fullcatname+".pkl",'rb')).to_pandas()
    
    #k2018_pms = pd.read_csv(catalogdir+'Ksoll2018_HTTP_PMS_catalogue.csv')
    #k2018_ums = pd.read_csv(catalogdir+'Ksoll2018_HTTP_UMS_selection.csv')
    #cat_full = pd.concat([k2018_pms,k2018_ums], axis=0, ignore_index=True)
    
    '''full = copy.deepcopy(cat_full).reset_index(drop=False)
    # drop any entries with missing mag estimates - revisit later to be less strict?
    full = full.dropna(how='any',subset=['m_f555w','m_f775u'])#,'m_f110w','m_f160w'])
    for m in ['m_f110w','m_f160w','m_f555w','m_f775u']:
        full.drop(full[full[m]>30].index,inplace=True) 
    for f in ['f_f555w','f_f775u']:
        full.drop(full[full[f]>2].index,inplace=True)  ''' 
        
    # load dereddened HTTP catalog
    full = pd.read_csv(catalogdir+'trim_HTTP.2015_10_20.1.csv')
    for m in ['m_f110w','m_f160w','m_f555w','m_f775u']:
        full.drop(full[full[m]>30].index,inplace=True) 
    
    if dered:
        features2 = ['m_f555w_dered', 'm_f775u_dered', 'AvNN']#'m_f110w_dered', 'm_f160w_dered']
        X_full = full[features2].to_numpy()
    
    if dered_full:
        features2 = ['m_f555w_dered', 'm_f775u_dered', 'm_f110w_dered', 'm_f160w_dered']#'m_f110w_dered', 'm_f160w_dered']
        X_full = full[features2].to_numpy()
        
    else:
        X_full = full[['m_f555w','m_f775u','m_f110w','m_f160w']].to_numpy()
    
    if scale:
        #scaler = preprocessing.StandardScaler().fit(X)
        scaler = preprocessing.MinMaxScaler().fit(X)  
        #scaler = preprocessing.RobustScaler().fit(X_full)
        #scaler = preprocessing.Normalizer().fit(X)
        X_scale = scaler.transform(X_full)        
        
    if not scale:
        
        X_scale = X_full
    
    full_prob = clf.predict_proba(X_scale)
    
    
    
    ########################################################################
    # Compare new SVM results to Ksoll2018 results
    
    from matplotlib.colors import DivergingNorm
    
    #cmap = LinearSegmentedColormap.from_list('seismic',['mediumblue','lightyellow','red'])
    
    norm = DivergingNorm(vmin=0, vcenter=0.5,vmax=1) #lsrk
    
    
    fig,[ax2,ax1] = plt.subplots(1,2,figsize=(12,8),sharex=True,sharey=True)
    
    s1=ax1.scatter(X_full[:, 0]-X_full[:, 1], X_full[:, 0], c=full_prob[:,1], 
                cmap='RdYlBu_r',s=0.3)  
    ax1.set_title('New SVM')
    ax1.set_xlabel('F555W - F775W [mag]')
    ax1.set_xlim(-1,3.5)
    ax1.set_ylim(28,12)
    ax1.set_ylabel('F555W [mag]')
    fig.colorbar(s1,ax=ax1,label='P(PMS)')
    
    ax2.scatter(X_full[:, 0]-X_full[:, 1], X_full[:, 0], c='#333399',s=0.1)  
    #s2 = ax2.scatter(pcolor0_dered,pmag0_dered,s=0.1,c=pms['p_svm'],
    #                 cmap='RdYlBu_r',norm=norm)
    ax2.set_title('Ksoll SVM')
    #fig.colorbar(s2,ax=ax2,label='P(PMS)')
    fig.tight_layout()
    ax2.set_ylabel('F555W [mag]')
    ax2.set_xlabel('F555W - F775W [mag]')
        
    
    
    
    #########
    # Visualize results
    
    fig = plt.figure(constrained_layout=True,figsize=(12,8))
    gs = GridSpec(2,2, figure=fig)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[:,1])
    
    ####### plot training set
    s1=ax1.scatter(X_train0[:, 0]-X_train0[:,1], X_train0[:, 0], c=y_train, 
                cmap='RdYlBu_r',s=0.7)  
    ax1.set_title('R136 Training set')
    #ax1.set_xlabel('F555W - F775W')
    ax1.set_ylabel('F555W [mag]')
    ax1.invert_yaxis()

    #ig.colorbar(s1,ax=ax1,label='P(PMS)')
    ####### plot testing set
    s2=ax2.scatter(X_test0[:, 0]-X_test0[:, 1], X_test0[:, 0], c=y_prob[:,1], 
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
                cmap='RdYlBu_r',s=0.1)  
    ax3.set_title('SVM on Dereddened 30Dor')
    ax3.set_xlabel('F555W - F775W [mag]')
    ax3.invert_yaxis()
    ax3.set_ylabel('F555W [mag]')
    ax3.set_xlim(-1,3.5)
    ax3.set_ylim(28,12)

    fig.colorbar(s3,ax=ax3,label='P(PMS)')
    
    
    #########
    # Visualize results
    
    fig = plt.figure(constrained_layout=True,figsize=(12,8))
    gs = GridSpec(1,2, figure=fig)
    ax2 = fig.add_subplot(gs[0])
    ax3 = fig.add_subplot(gs[1])
    
    #ig.colorbar(s1,ax=ax1,label='P(PMS)')
    ####### plot testing set
    s2=ax2.scatter(train['m_f555w']-train['m_f775w'], train['m_f555w'], 
                   c=train['pms_membership_prob'], 
                cmap='RdYlBu_r',s=0.7)  
    ax2.set_title('R136 Testing set')
    ax2.set_xlabel('F555W - F775W [mag]')
    ax2.set_ylabel('F555W [mag]')
    ax2.set_xlim(-1,3.5)
    ax2.set_ylim(28,12)
    ax2.text(0.65,0.9,
             'Accuracy: %.1f'%(100*metrics.accuracy_score(y_test, y_pred))+'%',
             transform=ax2.transAxes)
    ax2.set_title('Full training and test set')
    #fig.colorbar(s2,ax=ax2,label='P(PMS)')
    
    s3=ax3.scatter(X_full[:, 0]-X_full[:, 1], X_full[:, 0], c=full_prob[:,1], 
                cmap='RdYlBu_r',s=0.7)  
    ax3.set_title('30Dor')
    ax3.set_xlabel('F555W - F775W [mag]')
    ax3.invert_yaxis()
    ax3.set_ylabel('F555W [mag]')
    ax3.set_xlim(-1,3.5)
    ax3.set_ylim(28,12)

    fig.colorbar(s3,ax=ax3,label='P(PMS)')
    
    

    
    
    ##### thnk working properly now
    
    pms_ids = pms['ID'].values-1
    
    ksoll_probs = np.array([])
    new_probs = np.array([])
    for i in range(len(full)):
        corr_pms_idx = np.where(pms_ids == full['index'].values[i])
        print(i,corr_pms_idx)
        if corr_pms_idx[0] >= 0:
            ksoll_probs = np.append(ksoll_probs,pms['p_svm'].values[corr_pms_idx])
            new_probs = np.append(new_probs,full_prob[:,1][i])
    
    
    plt.figure()
    plt.scatter(ksoll_probs,new_probs,s=0.1,c='k')
    plt.xlabel('Ksoll P(PMS)')
    plt.ylabel('New SVM P(PMS)')
    plt.title('Comparing new to Ksoll18 P(PMS)')
    
    
    
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
    

    
    

    
    
    #############################################
    # Plot CMD of results
    
    # make custom cmap for binary pms training probs 
    cmap = ListedColormap(['darkblue', 'crimson'])
    bounds=[0,0.9,1]
    norm = BoundaryNorm(bounds, cmap.N)    
    # choosing what mags to plot & label
    plot1 = 0
    plot2 = 1
    
    ## plot
    fig, [ax1,ax2] = plt.subplots(2,1,figsize=(6,8),sharex=True,sharey=True)
    ax1.set_title('R136 with '+str(feat_title))
    # plot training set
    s1=ax1.scatter(X_train0[:, plot1]-X_train0[:,plot2], X_train0[:, plot1], 
                   c=y_train, 
                   norm=norm,
                cmap=cmap,s=0.7)  
    #ax1.set_title('R136 Training set')
    ax1.set_ylabel(features[plot1][2::],fontsize=12)
    ax1.text(0.05,0.9,'Training Set',
             transform=ax1.transAxes,fontsize=12)
        
    #ax1.set_ylabel(features[plot1][2::]+' - '+features[plot2][2::])
    fig.colorbar(s1,ax=ax1,label='P(PMS)')
    # plot testing set
    s2=ax2.scatter(X_test0[:, plot1]-X_test0[:, plot2], X_test0[:, plot1], 
                   c=y_prob[:,1], 
                cmap='RdYlBu_r',s=0.7)  
    #ax2.set_title('R136 Testing set')
    ax2.set_xlabel(features[plot1][2::]+' - '+features[plot2][2::],fontsize=12)
    ax2.set_ylabel(features[plot1][2::],fontsize=12)
    ax2.text(0.05,0.9,'Testing Set',
             transform=ax2.transAxes,fontsize=12)
    ax2.text(0.05,0.8,
             'Accuracy: %.1f'%(100*metrics.accuracy_score(y_test, y_pred))+'%',
             transform=ax2.transAxes)
    ax2.invert_yaxis()
    fig.colorbar(s2,ax=ax2,label='P(PMS)')
    
    fig.tight_layout()
    
    
    
    
    
    
    