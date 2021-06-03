
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics
#from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

'''
#################################################################

build_SVM: Script to create, execute, and test results of a
        Support Vector Machine (SVM)
        
In progress: To Dos include:
    - incorp actual HTTP training set
    - improve plotting fxns 
    - cross validation for cost/etc params
    - explore probability calculations
    - generalize/rethink structure of script
        
#################################################################
'''

def plot_svc_decision_function(model, ax=None, plot_support=True):
    """
    Plot the decision function for a 2D SVC
    
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
    
    ######### use astroML to pull similar phot. data
    # while waiting for training set
    # https://github.com/astroML/astroML
    from astroML.datasets import fetch_rrlyrae_combined
    # get data
    X, y = fetch_rrlyrae_combined()
    
    ########
    #split into training & testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                        test_size=0.3,random_state=109) # 70% training and 30% test
     
    ######### Build SVM
    #Create a svm Classifier (kernel choices include linear, poly, and rbf)
    clf = svm.SVC(kernel='rbf',C=1e6) # radial basis function kernel
                                      # C is cost function (basically error tolerance)

    #Train the model using the training sets
    clf.fit(X_train, y_train)
    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    
    ######### plot results
    '''
    # Plotting decision regions
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    z_min, z_max = X_test[:, 2].min() - 1, X_test[:, 2].max() + 1
    q_min, q_max = X_test[:, 3].min() - 1, X_test[:, 3].max() + 1
    
    
    xx, yy, zz, qq = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1),
                         np.arange(z_min, z_max, 0.1),
                         np.arange(q_min, q_max, 0.1))    
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel(),zz.ravel(),qq.ravel()])
    Z = Z.reshape(xx.shape)    
    '''    
        
    fig, ax = plt.subplots(1,1)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, s=3, cmap='bwr')  
    #ax.contourf(xx[:,:,0,0],yy[:,:,0,0],Z[:,:,0,0],alpha=0.4)
    ax.set_xlabel('u - g')
    ax.set_ylabel('g - r')
    ax.set_title('RR Lyrae')
    
    ######### metrics
    # inspect support vectors
    clf.support_vectors_
    # get indices of support vectors
    clf.support_
    # get num of SVs for each class
    clf.n_support_
    
    #### Test how well SVM works
    print(metrics.confusion_matrix(y_test,y_pred))
    print(metrics.classification_report(y_test,y_pred))
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:",metrics.precision_score(y_test, y_pred))
    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:",metrics.recall_score(y_test, y_pred))