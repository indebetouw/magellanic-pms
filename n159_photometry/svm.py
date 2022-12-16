







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
