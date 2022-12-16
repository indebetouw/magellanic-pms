import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import os, copy, pickle
if os.environ['USER'] =='toneill':
    catalogdir = '/Users/toneill/Box/MC/HST/'
else:
    catalogdir="../../MCBox/HST/"

def plot_svc_decision_function(model,data, orig_probs,ax=None, plot_support=True):
    """
    Plot decision function for a SVC trained on only two features (eg RA/dec)
    Based on: https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html
    """
    allX1 = data[:,0]
    allX2 = data[:,1]
    xlim = [np.min(allX1),np.max(allX1)]
    ylim= [np.min(allX2),np.max(allX2)]
    predClass = model.predict(data)

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    ### plot
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 6),sharex=True,sharey=True)
    ax1.scatter(allX1, allX2, c=orig_probs, cmap='RdYlBu', s=3, alpha=0.8)
    ax1.set_title('Original class')
    ax2.scatter(allX1, allX2, c=predClass, cmap='RdYlBu', s=3, alpha=0.8)
    ax2.set_title('Predicted class')

    # plot decision boundary and margins
    for ax in [ax1,ax2]:
        ax.contour(X, Y, P, colors='k',
                   levels=[-1, 0, 1], alpha=1,
                   linestyles=['--', '-', '--'],label='Decision boundary')
        # plot support vectors
        if plot_support:
            plt.scatter(model.support_vectors_[:, 0],
                       model.support_vectors_[:, 1],
                       s=300, linewidth=2, facecolors='none',label='Support Vectors')
    fig.tight_layout()
    return

if __name__ == '__main__':

    ################### first bit n159 specific
    #### here using non-dereddened for this example
    features = ['m_f555w', 'm_f775w']
    train_full = pd.read_csv(catalogdir + 'Ksoll2018_training_set.csv')
    train = copy.deepcopy(train_full)
    train = train.dropna(how='any', subset=features)
    for m in ['m_f110w', 'm_f160w', 'm_f555w', 'm_f775w']:
        train.drop(train[train[m] > 30].index, inplace=True)
    train = train.reset_index(drop=False)

    ################# below is general use once have training set
    X = train[features].to_numpy()
    y = np.where(train['pms_membership_prob'].values >= 0.9, 1, 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3)  # 70% training and 30% test
    X_train0 = X_train
    X_test0 = X_test

    ###############################################################
    # Build SVM - kernel choices include linear, poly, sigmoid, and radial basis fxn (rbf)
    clf = svm.SVC(kernel='linear',probability=True)
    # Train the model using the training sets
    clf.fit(X_train, y_train)
    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    ####################### plot decision boundaries
    plot_svc_decision_function(clf,X_train,y_train,plot_support=True)

