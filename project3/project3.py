from sklearn.tests.test_cross_validation import X_sparse

__author__ = 'Bahram'

from Reading_data import *
from scipy import stats
from sklearn.decomposition import PCA
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.utils import shuffle
from scipy.sparse import coo_matrix
import numpy as np
from pylab import *
from sklearn.mixture import GMM
from sklearn import cross_validation
import sklearn.metrics as mcs
from graphs import *


def hierarchical_cluster(remove_doc_index=None ):
    '''

    :param remove_doc_index: The index of a doc instance to be removed .It is used for removing potential outlier
    :return:
    '''
    from Reading_data import *
    # Normalize data
    X = stats.zscore(X)
    # shuffle data
    # X_sparse = coo_matrix(X)
    # X, X_sparse, y = shuffle(X, X_sparse, y)
    if (remove_doc_index is not None):
        X = np.delete(X, (remove_doc_index), axis=0)
        y=np.delete(y,(remove_doc_index),None)
        y=y.T

    # Perform hierarchical/agglomerative clustering on data matrix

    methods = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']
    metrics = ['euclidean', 'cityblock', 'cosine']
    Method = 'single'
    Metric = 'euclidean'

    Maxclust = 2
    Z = linkage(X, method=Method, metric=Metric)
    # Compute and display clusters by thresholding the dendrogram
    cls = fcluster(Z, criterion='maxclust', t=Maxclust)
    F_Measure=mcs.f1_score(y, cls, average='weighted')
    figure(1)
    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)
    #If data is more than 2-dimensional it should be first projected onto the first two principal components
    clusterplot(X_r, cls.reshape(cls.shape[0], 1), y=y, x_label="PCA1", y_label="PCA2")

    # Display dendrogram
    max_display_levels = 20
    figure(2)
    xlabel("doc instance")
    ylabel('Distance')
    title('F Measure: {0}'.format(round(F_Measure,2)))
    dendrogram(Z, truncate_mode='level', p=max_display_levels)
    show()


def gmm_inner_loop(nr, X_cv, K=11, CV=10):
    # Range of K's to try
    KRange = range(1, K)
    T = len(KRange)

    covar_type = 'diag'  # you can try out 'diag' as well
    reps = 3  # number of fits with different initalizations, best result will be kept

    # Allocate variables
    CVE = np.zeros((CV, T))
    N, M = X_cv.shape
    # K-fold crossvalidation
    CV = cross_validation.KFold(N, CV, shuffle=True)
    # For each crossvalidation fold
    fold = 0
    for train_index, test_index in CV:
        # extract training and test set for current CV fold
        X_train = X_cv[train_index]
        X_test = X_cv[test_index]
        for i, K in enumerate(KRange):
            print('Inner fold {0} Fitting model K={1}\n'.format(fold + 1, i))
            # Fit Gaussian mixture model to X_train
            gmm = GMM(n_components=K, covariance_type=covar_type, n_init=reps, params='wmc').fit(X_train)
            # compute negative log likelihood of X_test
            # CVE[t] += -gmm.score(X_test).sum()
            CVE[fold, i] = -gmm.score(X_test).sum()
        fold += 1
    # taking the mean of the CVE
    mean_cve_vs_k = np.mean(CVE, 0)
    best_K = KRange[np.argmin(mean_cve_vs_k)]
    plot_inner(nr, KRange, mean_cve_vs_k)
    show()
    # return best_K, mean_cve_vs_k



    # Plot results
    """
    figure(1); hold(True)
    plot(KRange, BIC)
    plot(KRange, AIC)
    plot(KRange, 2*CVE)
    legend(['BIC', 'AIC', 'Crossvalidation'])
    xlabel('K')
    show()
    """


def plot_inner(FigNr, XValues, YValues):
    figure(FigNr);
    hold(True)
    plot(XValues, YValues)
    legend(['GMM Crossvalidation'])
    xlabel('K')
    ylabel('CVE')


def gmm_cluster():
    CV_OUTER = 10
    CV_INNER = 2;
    KRange = 11;
    CV = cross_validation.KFold(N, CV_OUTER, shuffle=True)
    i = 1;
    for train_index, test_index in CV:
        X_train = X[train_index]
        X_test = X[test_index]
        gmm_inner_loop(i, X_train, KRange, CV_INNER)
        break
        # i+=1


def fmeasure_vs_mm():
    from Reading_data import *

    methods = ['single', 'complete', 'average', 'weighted', 'median', 'ward']
    metrics = ['euclidean']
    plot_fscore_vs_method_metrics_hc(X, methods, metrics);


hierarchical_cluster()
# gmm_cluster()

#fmeasure_vs_mm();