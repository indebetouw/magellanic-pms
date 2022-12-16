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

'''
#################################################################

build_SVM: Script to create, execute, and test results of a
        Support Vector Machine (SVM)

#################################################################
'''


if __name__ == '__main__': 
    
    
    scale = True
    
    # whether to include only longer wavelengths in training
    long = False
    full = False
    extin = False
    dered = False
    dered_full=True
    simp=False
    
    if simp:
        features = ['mag_555_dered','mag_814_dered']
    #if long:
    #    features = ['m_f775w','m_f110w','m_f160w']
    if full:
        features = ['mag_555_dered','mag_814_dered']#,'m_f110w','m_f160w']#,'A_v']
    #if extin:
    #    features = ['m_f110w','m_f160w','A_v']
    if dered:
        features = ['mag_555_dered','mag_814_dered']#,'A_v']#'m_f110w_dered','m_f160w_dered']#,'A_v']
    if dered_full:
        features =  ['mag_555_dered','mag_814_dered']#,'A_v']#,'m_f110w_dered','m_f160w_dered']
    
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
    tr_av0 = train_full['A_v']
    #train_full['AvNN'] = tr_av0
    ## note that values getting from NN Av here vary slightly from Ksoll training Avs
    train_full['m_f555w_dered'] = train_full['m_f555w'] - tr_av0*R_BV[0]/R_BV[0]
    train_full['m_f775w_dered'] = train_full['m_f775w'] - tr_av0*R_BV[1]/R_BV[0]
    train_full['m_f110w_dered'] = train_full['m_f110w'] - tr_av0*R_BV[2]/R_BV[0]
    train_full['m_f160w_dered'] = train_full['m_f160w'] - tr_av0*R_BV[3]/R_BV[0]    
    
    train = copy.deepcopy(train_full)
    # drop any entries with missing mag estimates - revisit later to be less strict?
    train = train.dropna(how='any',subset=['m_f555w_dered','m_f775w_dered'])#features)
    for m in ['m_f555w','m_f775w']:#'m_f110w','m_f160w',
        train.drop(train[train[m]>30].index,inplace=True)
    train = train.reset_index(drop=False)
    

    ###########
    # Split into training & testing sets to make SVM
    y = np.where(train['pms_membership_prob'].values >= 0.85, 1, 0)
    
    if scale:

        features_vict = ['m_f555w_dered','m_f775w_dered']
        ### doesn't seem to impact accuracy much
        
        from sklearn import preprocessing   
        
        X = train[features_vict].to_numpy()
        #scaler = preprocessing.StandardScaler().fit(X)
        scaler = preprocessing.MinMaxScaler()#.fit(X)  
        #scaler = preprocessing.RobustScaler().fit(X)
        #scaler = preprocessing.Normalizer().fit(X)
        
        X_train0, X_test0, y_train, y_test = train_test_split(X, y, 
                            test_size=0.3) # 70% training and 30% test
        
        X_train = scaler.fit_transform(X_train0) #scaler.transform(X_train0)
        X_test = scaler.transform(X_test0) # scaler.transform(X_test0)
        
        
    if not scale:
        
        X = train[features].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.3) # 70% training and 30% test
        X_train0 = X_train
        X_test0 = X_test

    ########################################################################
    # Run SVM on entirety of 30Dor data

    # k2018_pms = pd.read_csv(catalogdir+'Ksoll2018_HTTP_PMS_catalogue.csv')
    # k2018_ums = pd.read_csv(catalogdir+'Ksoll2018_HTTP_UMS_selection.csv')

    # load dereddened HTTP catalog
    photdir = '/Users/toneill/N159/photometry/'

    full = pd.read_csv(photdir + 'n159-all_reduce.phot.cutblue.dered.csv')
    '''for m in [features]:#'m_f110w','m_f160w','m_f555w',' m_f775u']:
        full.drop(full[full[m]>30].index,inplace=True)'''

    X_full = full[features].to_numpy()
    '''features2 = copy.deepcopy(features)
    if 'm_f775w_dered' in features2:
        features2[features2.index('m_f775w_dered')] = 'm_f775u_dered'
    if 'm_f775w' in features2:
        features2[features2.index('m_f775w')] = 'm_f775u'    
    if dered == True:
        features2[2] = 'AvNN'
    X_full = full[features2].to_numpy()'''

    if scale:
        X_scale = scaler.transform(X_full)
    if not scale:
        X_scale = X_full

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

    grid = False

    if grid:

        C_range = np.logspace(-2, 10, 13)
        gamma_range = np.logspace(-9, 3, 13)

        param_grid = {'C':list(np.linspace(9.2,10,10)), 'gamma':np.linspace(19.5,21,10)}

                         # list(np.logspace(-1,0.2, 15))}#,
           # 'C':[2.**n for n in np.linspace(9,14,10)],
                    # 'gamma':[2.**n for n in np.linspace(7,9,8)]}

        strat = True
        if strat:
            # create  K-fold with stratification training/test balanced
            # repeat to reduce variance
            cv = RepeatedStratifiedKFold(n_splits=3,n_repeats=3)
            grid = GridSearchCV(SM, param_grid, n_jobs=8, cv=cv,
                                refit=True,verbose=1)

        if not strat:
            grid = GridSearchCV(SM, param_grid, n_jobs=8,
                                refit=True,verbose=1)

        grid.fit(X_train, y_train)
        print(grid.best_params_)

        ######### Run SVM using CVd hyperparams
        clf = svm.SVC(kernel='rbf',probability=True,C=grid.best_params_['C'], gamma=grid.best_params_['gamma'])

    if not grid:
        clf = svm.SVC(kernel='rbf',probability=True,gamma=10,C=20)

    # Train the model using the training sets
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    full_prob = clf.predict_proba(X_scale)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    import cmasher as cmr

    cmap_use = cmr.get_sub_cmap('inferno',0,0.9)#cmr.ember, 0.2, 0.8)
    #summary_plot(cmap_use)

    ###############################################

    plt.figure(figsize=(6,8))
    plt.scatter(X_scale[:,0],X_scale[:,1],c=full_prob[:,1],cmap=cmap_use,vmin=0,vmax=1,s=0.25,alpha=0.5)
    plot_svc_decision_function(clf)
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.xlabel('F555W [mag]')
    plt.ylabel('F814W [mag]')


    ##############################################################################################

    fig, [ax1,ax2] = plt.subplots(2,1,figsize=(6,9),sharex=True,sharey=True)

    ####### plot training set
    s1 = ax1.scatter(X_train0[:, 0] - X_train0[:, 1], X_train0[:, 0], c=y_train,
                     cmap=cmap_use, s=6, vmin=0, vmax=1)
    ax1.set_title('R136 Training set')
    # ax1.set_xlabel('F555W - F775W')
    ax1.set_ylabel('F555W [mag]',fontsize=12)
    ax1.invert_yaxis()
    #ax1.set_xlim(-1, 2.5)
    #ax1.set_ylim(26, 16)

    ####### plot testing set
    s2 = ax2.scatter(X_test0[:, 0] - X_test0[:, 1], X_test0[:, 0], c=y_prob[:, 1],
                     cmap=cmap_use, s=5, vmin=0, vmax=1)
    ax2.set_title('R136 Testing set')
    ax2.set_xlabel('F555W - F775W [mag]',fontsize=12)
    ax2.set_ylabel('F555W [mag]',fontsize=12)
    ax2.text(0.7, 0.9,
             'Accuracy: %.1f' % (100 * metrics.accuracy_score(y_test, y_pred)) + '%',
             transform=ax2.transAxes,bbox=dict(facecolor='none', edgecolor='k'))

    for ax in [ax1, ax2]:
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(direction='in', which='both', labelsize=9)

    fig.subplots_adjust(right=0.85,bottom=0.06,top=0.96,left=0.1,hspace=0.13)
    cbar_ax = fig.add_axes([0.87, 0.06, 0.05, 0.9])
    cbar = fig.colorbar(s2, cax=cbar_ax)
    cbar.set_label(label='P(PMS)', fontsize=12, labelpad=-2)
    #cbar.ax.tick_params(labelsize=7.5)

    plt.savefig('traintest.png', dpi=300)

    ##########################################################


    fig, [ax1,ax2] = plt.subplots(1,2,figsize=(12,8))#,sharex=True,sharey=True)

    ax1.scatter(X_scale[:,0],X_scale[:,1], c= full_prob[:,1],
                     cmap=cmap_use, s=4, vmin=0, vmax=1)
    plot_svc_decision_function(clf,ax=ax1,color='royalblue')
    ax1.invert_yaxis()
    ax1.invert_xaxis()
    ax1.set_xlabel('Scaled F555W',fontsize=12)
    ax1.set_ylabel('Scaled F814W',fontsize=12)

    #ax2.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
    #            cmap=cmap_use, s=0.5, vmin=0, vmax=1)

    s3 = ax2.scatter(X_full[:, 0] - X_full[:, 1], X_full[:, 0], c=full_prob[:, 1],
                     cmap=cmap_use, s=3, vmin=0, vmax=1)
    ax2.set_xlabel('F555W - F814W [mag]',fontsize=12)
    ax2.invert_yaxis()
    ax2.set_ylabel('F555W [mag]',fontsize=12,labelpad=7)
    ax2.set_xlim(-1, 4.2)
    ax2.set_ylim(26, 16)

    for ax in [ax1, ax2]:
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(direction='in', which='both', labelsize=10)

    fig.subplots_adjust(right=0.92,bottom=0.06,top=0.97,left=0.05,wspace=0.13)
    cbar_ax = fig.add_axes([0.93,0.06,0.02,0.91])
    cbar = fig.colorbar(s2, cax=cbar_ax)
    cbar.set_label(label='P(PMS)', fontsize=12, labelpad=-2)

    plt.savefig('fit_pms_cmds.png',dpi=300)







    #### examine the best model
    # print()
    ## best score achieved during the GridSearchCV
    # print('GridSearch CV best score : {:.4f}\n\n'.format(grid.best_score_))
    # print parameters that give the best results
    # print('Parameters that give the best results :','\n\n', (grid.best_params_))
    # print estimator that was chosen by the GridSearch
    # print('\n\nEstimator that was chosen by the search :','\n\n', (grid.best_estimator_))



    def summary_plot(cmap_use):
    
        fig = plt.figure(constrained_layout=True,figsize=(12,8))
        gs = GridSpec(2,2, figure=fig)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[:,1])

        ####### plot training set
        s1=ax1.scatter(X_train0[:, 0]-X_train0[:,1], X_train0[:, 0], c=y_train,
                    cmap=cmap_use,s=1,vmin=0,vmax=1)
        ax1.set_title('R136 Training set')
        #ax1.set_xlabel('F555W - F775W')
        ax1.set_ylabel('F555W [mag]')
        #ax1.invert_yaxis()
        ax1.set_xlim(-1,2.5)
        ax1.set_ylim(26,16)

        #ig.colorbar(s1,ax=ax1,label='P(PMS)')
        ####### plot testing set
        s2=ax2.scatter(X_test0[:, 0]-X_test0[:, 1], X_test0[:, 0], c=y_prob[:,1],
                    cmap=cmap_use,s=1,vmin=0,vmax=1)
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
                    cmap=cmap_use,s=0.4,vmin=0,vmax=1)
        ax3.set_title('Trained SVM')# on Dereddened All')
        ax3.set_xlabel('F555W - F814W [mag]')
        ax3.invert_yaxis()
        ax3.set_ylabel('F555W [mag]')
        ax3.set_xlim(-1,3.5)
        ax3.set_ylim(26,16)

        fig.colorbar(s3,ax=ax3,label='P(PMS)')

        for ax in [ax1,ax2,ax3]:
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.tick_params(direction='in', which='both', labelsize=9)

    ##############################################

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    s = ax.scatter(full['ra_814'],full['dec_814'], c=full_prob[:,1],
                cmap=cmap_use,s=0.5,vmin=0,vmax=1,alpha=0.8)
    ax.invert_xaxis()
    fig.colorbar(s,ax=ax,label='P(PMS)')

    from astropy.io import fits
    from astropy.wcs import wcs
    import cmasher as cmr

    alma_co = fits.open('/Users/toneill/N159/alma/12CO_combined.regrid.gtr0.2K.maximum.fits')
    #region = 'N159E'

    refdir = photdir+'ref_files_WCS/'
    hst_n159e = fits.open(refdir+'n159e_f814w_drc_sci.chip0.fits')
    hst_n159w = fits.open(refdir+'n159w_f814w_drc_sci.chip0.fits')
    hst_n159s = fits.open(refdir+'n159s_f814w_drc_sci.chip0.fits')



    pms_inds = full_prob[:,1] >= 0.95

    from skimage import measure

    fig = plt.figure(figsize=(13.5,6.5))
    ax = fig.add_subplot(121, projection=wcs.WCS(alma_co[0].header))
    ax2 = fig.add_subplot(122, projection=wcs.WCS(alma_co[0].header))
    #ax.imshow(alma_co[0].data,cmap='Greys_r',zorder=0,alpha=0)
    s = ax.scatter(full['ra_814'],full['dec_814'], c=full_prob[:,1],
                cmap=cmap_use,s=0.5,vmin=0,vmax=1,alpha=0.95,
                   transform=ax.get_transform('fk5'))
    ax.set_xlabel('RA',fontsize=12)
    ax.set_ylabel('Dec',fontsize=12,labelpad=0)
    ax.set_xlim(110,335)
    ax.set_ylim(-97,240)

    alma_cont = np.where(alma_co[0].data >= -100000, 0, 1)
    contours = measure.find_contours(alma_cont, 0.9)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], c='k',   lw=1)  # ,ls=':')

    ax2.contour(alma_co[0].data, levels=[5], colors=['grey'],  zorder=3)
    #ax2.imshow(alma_co[0].data,vmax=20,cmap=cmr.freeze,zorder=0)
    ax2.scatter(full['ra_814'][pms_inds],full['dec_814'][pms_inds],c=full_prob[:,1][pms_inds],
               cmap=cmap_use,s=2,vmin=0,vmax=1,alpha=0.5,transform=ax2.get_transform('fk5'),zorder=2)
    ax2.set_xlabel('RA',fontsize=12)
    ax2.set_ylabel(' ',fontsize=1)
    ax2.set_xlim(110,335)
    ax2.set_ylim(-97,240)
    for contour in contours:
        ax2.plot(contour[:, 1], contour[:, 0], c='k',     lw=1)  # ,ls=':')

    for ax_i in [ax, ax2]:
        ax_i.xaxis.set_ticks_position('both')
        ax_i.yaxis.set_ticks_position('both')
        ax_i.tick_params(direction='in', which='both', labelsize=9)

    fig.subplots_adjust(right=0.92,bottom=0.08,top=0.97,left=0.07,wspace=0.17)
    cbar_ax = fig.add_axes([0.94,0.08,0.02,0.89])
    cbar = fig.colorbar(s, cax=cbar_ax)
    cbar.set_label(label='P(PMS)', fontsize=12, labelpad=-2)

    plt.savefig('spatial_pms.png', dpi=300)

    #################################################

    full['ppms'] = full_prob[:,1]
    full.to_csv(photdir + 'n159-all_reduce.phot.cutblue.dered.ppms.csv',index=False)


    fig = plt.figure(figsize=(13.5,6.5))
    ax = fig.add_subplot(121, projection=wcs.WCS(alma_co[0].header))
    #ax2 = fig.add_subplot(122, projection=wcs.WCS(alma_co[0].header))

    ax.imshow(hst_n159e[0].data,transform=ax.get_transform(wcs.WCS(hst_n159e[0].header)),vmax=1,cmap='Greys_r')
    ax.imshow(hst_n159w[0].data,transform=ax.get_transform(wcs.WCS(hst_n159w[0].header)),vmax=2,alpha=1,zorder=0,cmap='Greys_r')
    ax.scatter(full['ra_814'][pms_inds],full['dec_814'][pms_inds],c=full_prob[:,1][pms_inds],
               cmap=cmap_use,s=2,vmin=0,vmax=1,alpha=0.5,transform=ax.get_transform('fk5'),zorder=2)

    ax.imshow(hst_n159s[0].data,transform=ax.get_transform(wcs.WCS(hst_n159s[0].header)),vmax=0.5)
    ax.contour(alma_co[0].data, levels=[5], colors=['r'],  zorder=3)


    for fitsim in [hst_n159e, hst_n159w,hst_n159s]:
        hst_cont = np.where(fitsim[0].data >= 0, 0, 1)
        contours = measure.find_contours(hst_cont, 0.9)
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], c='k',
                     lw=0.5)

    ######################################

    import seaborn as sns

    nonpms_inds = (full_prob[:,1] < 0.1)

    sns.kdeplot(full['ra_814'][pms_inds],full['dec_814'][pms_inds],color='#f89c04')
    sns.kdeplot(full['ra_814'][nonpms_inds],full['dec_814'][nonpms_inds],color='k')
    plt.gca().invert_xaxis()
    plt.xlabel('RA')
    plt.ylabel('Dec')


    #########################################


    from scipy import stats

    binned = stats.binned_statistic_2d(x=full['ra_814'].values,
                                     y=full['dec_814'].values, values=full_prob[:,1],
                                     statistic='mean',bins=50)

    xedges = binned.x_edge
    yedges = binned.y_edge

    # map of target field
    plt.figure(figsize=(8,6))
    plt.imshow(binned.statistic.T, extent=[xedges[0], xedges[-1], \
                                               yedges[0], yedges[-1]],
                    cmap='RdYlBu_r', vmin=0,vmax=1)
    plt.colorbar()

    #fig.tight_layout()

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(121, projection=wcs.WCS(alma_co[0].header))
    ax.contour(alma_co[0].data, levels=[5],colors=['c'],zorder=6)
    twoD_kde(full['ra_814'][pms_inds],full['dec_814'][pms_inds],ax=ax,fig=fig,discrete=False,cmap_kde=cmr.ember)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')
    ax.set_title('P(PMS) $\geq$ 0.85')

    ax2 = fig.add_subplot(122, projection=wcs.WCS(alma_co[0].header))
    twoD_kde(full['ra_814'][~pms_inds],full['dec_814'][~pms_inds],ax=ax2,fig=fig,discrete=False,cmap_kde=cmr.ember)
    plt.gca().invert_xaxis()
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')
    ax.set_title('P(PMS) $<$ 0.9')








    #####
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    s = ax.scatter(full['ra_814'][pms_inds],full['dec_814'][pms_inds], c=full_prob[:,1][pms_inds],
                cmap='RdYlBu_r',s=10,vmin=0,vmax=1,alpha=0.5)
    fig.colorbar(s,ax=ax,label='P(PMS)')

    '''
    # To send to glue:

    from glue import qglue

    full['p_svm'] = full_prob[:,1]    
    full['VminI'] = full['m_f555w'] - full['m_f775u']
    full['VminI_dered'] = full['m_f555w_dered'] - full['m_f775u_dered']    

    gsesh = qglue(full=full)
    '''

    ########################################################################
    # Visualize results
    ########################################################################
    #cmap_use = 'RdYlBu_r'
    #cmap_use = 'inferno'

    n = 0.2
    cmap_use = cmr.get_sub_cmap('twilight',n,1-n)
    cmap_use = cmr.get_sub_cmap(cmr.fusion,n,1-n)

    #cmap_use = cmr.get_sub_cmap('Reds_r', 0.0, 0.5)

    ######################
    #cmap_use = cmr.get_sub_cmap(cmr.voltage, 0, 0.5)
    #cmap_use = cmr.get_sub_cmap(cmr.amethyst, 0.1, 0.6)

    #####################
    # Plot training/test/and results

    '''cmap_use = cmr.get_sub_cmap(cmr.bubblegum, 0, 0.8)
       # cmr.bubblegum

    summary_plot(cmr.get_sub_cmap(cmr.voltage, 0, 0.6))


    summary_plot(cmr.get_sub_cmap(cmr.torch, 0.1,0.7))

    summary_plot(cmr.get_sub_cmap(cmr.redshift, 0.05, 0.95))

    summary_plot(cmr.get_sub_cmap(cmr.guppy_r, 0.15, 0.85))'''

    #cmap_use = cmr.get_sub_cmap(cmr.guppy_r, 0.15, 0.85)

    #####################
    # Compare new SVM results to Ksoll2018 results
    # need to have run rev_30dor_dered.py to have pcolor0_dered, pmag0_dered
    # of ksoll pms data - TO DO: fix this to work without having done that

    # norm = DivergingNorm(vmin=0, vcenter=0.5,vmax=1) #lsrk
    '''
    fig,[ax2,ax1] = plt.subplots(1,2,figsize=(12,8),sharex=True,sharey=True)

    s1=ax1.scatter(X_full[:, 0]-X_full[:, 1], X_full[:, 0], c=full_prob[:,1], 
                cmap='RdYlBu_r',s=0.1,norm=norm)  
    ax1.set_title('New SVM')
    ax1.set_xlabel('F555W - F775W [mag]')
    ax1.set_xlim(-1,3.5)
    ax1.set_ylim(28,12)
    ax1.set_ylabel('F555W [mag]')
    fig.colorbar(s1,ax=ax1,label='P(PMS)')

    #ax2.scatter(X_full[:, 0]-X_full[:, 1], X_full[:, 0], c='#333399',s=0.1)  
    #s2 = ax2.scatter(pcolor0_dered,pmag0_dered,s=0.1,c=pms['p_svm'],
    #                 cmap='RdYlBu_r',norm=norm)
    ax2.set_title('Ksoll SVM')
    #fig.colorbar(s2,ax=ax2,label='P(PMS)')
    fig.tight_layout()
    ax2.set_ylabel('F555W [mag]')
    ax2.set_xlabel('F555W - F775W [mag]')'''

    '''
    #########
    # Plot combined train + test vs results
    
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
    
    
    #############################################
    # Plot training vs test only
    
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
    '''
    
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
                cmap=cmap_use,vmin=0,vmax=1,s=0.3)
    ax1.invert_yaxis()
    ax1.set_xlabel('F555W - F775W')
    ax1.set_ylabel('F555W')
    ax1.set_xlim(-1.3,3.4)
    ax1.set_ylim(26.8,18.7)
    fig.colorbar(s,label='P(PMS)')
    ax1.set_title('R136 Training Set')
    fig.tight_layout()
    '''
       

    '''
    
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
    plt.title('Comparing new to Ksoll18 P(PMS)')'''
     
    
    
    
    #plt.style.use('ggplot')    
    
    
    
    