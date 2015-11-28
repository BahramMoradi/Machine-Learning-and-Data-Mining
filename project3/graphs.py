from pylab import *
from sklearn.cluster import k_means
import sklearn.metrics.cluster as cluster_metrics
from scipy import stats
from sklearn.decomposition import PCA
#from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.utils import shuffle
from scipy.sparse import coo_matrix
import numpy as np
from pylab import *
from sklearn.mixture import GMM
from sklearn import cross_validation
import sklearn.metrics as mcs
# Maximum number of clusters:
def clusterplot(fold,X, clusterid, centroids='None', y='None', covars='None',plotTitle=None,x_Label=None,y_label=None,dir_to_save=None):
    '''
    CLUSTERPLOT Plots a clustering of a data set as well as the true class
    labels. If data is more than 2-dimensional it should be first projected
    onto the first two principal components. Data objects are plotted as a dot
    with a circle around. The color of the dot indicates the true class,
    and the cicle indicates the cluster index. Optionally, the centroids are
    plotted as filled-star markers, and ellipsoids corresponding to covariance
    matrices (e.g. for gaussian mixture models).

    Usage:
    clusterplot(X, clusterid)
    clusterplot(X, clusterid, centroids=c_matrix, y=y_matrix)
    clusterplot(X, clusterid, centroids=c_matrix, y=y_matrix, covars=c_tensor)

    Input:
    X           N-by-M data matrix (N data objects with M attributes)
    clusterid   N-by-1 vector of cluster indices
    centroids   K-by-M matrix of cluster centroids (optional)
    y           N-by-1 vector of true class labels (optional)
    covars      M-by-M-by-K tensor of covariance matrices (optional)
    '''

    X = np.asarray(X)
    cls = np.asarray(clusterid)
    if y=='None':
        y = np.zeros((X.shape[0],1))
    else:
        y = np.asarray(y)
    if centroids!='None':
        centroids = np.asarray(centroids)
    #K = np.size(np.unique(cls))
    K=np.max(cls)+1
    C = np.size(np.unique(y))
    ncolors = np.max([C,K])
    fig=figure(figsize=(14,9))
    # plot data points color-coded by class, cluster markers and centroids
    fig.hold(True)
    colors = [0]*ncolors
    for color in range(ncolors):
        colors[color] = cm.jet.__call__(color*255/(ncolors-1))[:3]
    for i,cs in enumerate(np.unique(y)):
        plot(X[(y==cs).ravel(),0], X[(y==cs).ravel(),1], 'o', markeredgecolor='k', markerfacecolor=colors[i],markersize=6, zorder=2)
    for i,cr in enumerate(np.unique(cls)):
        plot(X[(cls==cr).ravel(),0], X[(cls==cr).ravel(),1], 'o', markersize=12, markeredgecolor=colors[i], markerfacecolor='None', markeredgewidth=3, zorder=1)
    if centroids!='None':
        for cd in range(centroids.shape[0]):
            plot(centroids[cd,0], centroids[cd,1], '*', markersize=22, markeredgecolor='k', markerfacecolor=colors[cd], markeredgewidth=2, zorder=3)
    # plot cluster shapes:
    if covars!='None':
        for cd in range(centroids.shape[0]):
            x1, x2 = gauss_2d(centroids[cd],covars[cd,:,:])
            plot(x1,x2,'-', color=colors[cd], linewidth=3, zorder=5)

    if x_Label is not None:
        xlabel(x_Label)
    if y_label is not None:
        ylabel(y_label)

    # create legend
    legend_items = np.unique(y).tolist()+np.unique(cls).tolist()+np.unique(cls).tolist()
    for i in range(len(legend_items)):
        if i<C: legend_items[i] = 'Class: {0}'.format(legend_items[i]);
        elif i<C+K: legend_items[i] = 'Cluster: {0}'.format(legend_items[i]);
        else: legend_items[i] = 'Centroid: {0}'.format(legend_items[i]);
    legend(legend_items, numpoints=1, markerscale=.75, prop={'size': 9},bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if dir_to_save is not None:
        savefig(dir_to_save + '\\gmmCluster{0}.eps'.format(fold), format='eps', dpi=1000,bbox_inches='tight')
        savefig(dir_to_save + '\\gmmCluster{0}.png'.format(fold), format='png',bbox_inches='tight')
    #fig.hold(False)

def gauss_2d(centroid, ccov, std=2, points=100):
    ''' Returns two vectors representing slice through gaussian, cut at given standard deviation. '''
    mean = np.c_[centroid];
    tt = np.c_[np.linspace(0, 2 * np.pi, points)]
    x = np.cos(tt);
    y = np.sin(tt);
    ap = np.concatenate((x, y), axis=1).T
    d, v = np.linalg.eig(ccov);
    d = std * np.sqrt(np.diag(d))
    bp = np.dot(v, np.dot(d, ap)) + np.tile(mean, (1, ap.shape[1]))
    return bp[0, :], bp[1, :]

def find_cluster_label(cls,y_true):
    '''
    find the number
    :param cls:
    :param y_true:
    :return:
    '''
    cluster_nr=len(set(cls))
    cc=np.zeros((cluster_nr,2))
    #print "clusters : ",cls
    #print len(cls)
    #print len(y_true)

    for cluster,clss in zip(cls,y_true):
        cc[cluster,clss]+=1
    return cc;
    #print cc
    #print cc.sum()
def clusterval(y, clusterid):
    '''
    CLUSTERVAL Estimate cluster validity using Entropy, Purity, Rand Statistic,
    and Jaccard coefficient.

    Usage:
      Entropy, Purity, Rand, Jaccard = clusterval(y, clusterid);

    Input:
       y         N-by-1 vector of class labels
       clusterid N-by-1 vector of cluster indices

    Output:
      Entropy    Entropy measure.
      Purity     Purity measure.
      Rand       Rand index.
      Jaccard    Jaccard coefficient.
    '''
    y = np.asarray(y).ravel();
    clusterid = np.asarray(clusterid).ravel()
    C = np.unique(y).size;
    K = np.unique(clusterid).size;
    N = y.shape[0]
    EPS = 2.22e-16

    p_ij = np.zeros((K, C))  # probability that member of i'th cluster belongs to j'th class
    m_i = np.zeros((K, 1))  # total number of objects in i'th cluster
    for k in range(K):
        m_i[k] = (clusterid == k).sum()
        yk = y[clusterid == k]
        for c in range(C):
            m_ij = (yk == c).sum()  # number of objects of j'th class in i'th cluster
            if m_i[k]!=0:
                p_ij[k, c] = m_ij.astype(float) / m_i[k]
    entropy = ( (1 - (p_ij * np.log2(p_ij + EPS)).sum(axis=1)) * m_i.T ).sum() / (N * K)
    purity = ( p_ij.max(axis=1) ).sum() / K

    f00 = 0;
    f01 = 0;
    f10 = 0;
    f11 = 0
    for i in range(N):
        for j in range(i):
            if y[i] != y[j] and clusterid[i] != clusterid[j]:
                f00 += 1;  # different class, different cluster
            elif y[i] == y[j] and clusterid[i] == clusterid[j]:
                f11 += 1;  # same class, same cluster
            elif y[i] == y[j] and clusterid[i] != clusterid[j]:
                f10 += 1;  # same class, different cluster
            else:
                f01 += 1;  # different class, same cluster
    rand = np.float(f00 + f11) / (f00 + f01 + f10 + f11)
    jaccard = np.float(f11) / (f01 + f10 + f11)

    return entropy, purity, rand, jaccard


def plot_eval(K, y, cls):
    '''

    :param K: number of cluster
    :return:
    '''
    # Allocate variables:
    Entropy = np.zeros((K, 1))
    Purity = np.zeros((K, 1))
    Rand = np.zeros((K, 1))
    Jaccard = np.zeros((K, 1))
    OtherMetrics = np.zeros((K, 5))

    for k in range(K):
        # run K-means clustering:
        # cls = Pycluster.kcluster(X,k+1)[0]
        #centroids, cls, inertia = k_means(X,k+1)
        # compute cluster validities:
        Entropy[k], Purity[k], Rand[k], Jaccard[k] = clusterval(y, cls)
        # compute other metrics, implemented in sklearn.metrics package
        OtherMetrics[k, 0] = cluster_metrics.supervised.completeness_score(y.A.ravel(), cls)
        OtherMetrics[k, 1] = cluster_metrics.supervised.homogeneity_score(y.A.ravel(), cls)
        OtherMetrics[k, 2] = cluster_metrics.supervised.mutual_info_score(y.A.ravel(), cls)
        OtherMetrics[k, 3] = cluster_metrics.supervised.v_measure_score(y.A.ravel(), cls)
        OtherMetrics[k, 4] = cluster_metrics.supervised.adjusted_rand_score(y.A.ravel(), cls)




    # Plot results:

    figure(1)
    title('Cluster validity')
    hold(True)
    plot(np.arange(K) + 1, -Entropy)
    plot(np.arange(K) + 1, Purity)
    plot(np.arange(K) + 1, Rand)
    plot(np.arange(K) + 1, Jaccard)
    ylim(-2, 1.1)
    legend(['Negative Entropy', 'Purity', 'Rand', 'Jaccard'], loc=4)

    figure(2)
    title('Cluster validity - other metrics')
    hold(True)
    plot(np.arange(K) + 1, OtherMetrics)
    legend(['completeness score', 'homogeneity score', 'mutual info score', 'v-measure score', 'adjusted rand score'],
           loc=4)

    show()


def plot_fscore_vs_method_metrics_hc(X, methods, metrics):
    from Reading_data import *

    labels = list()
    fm = list()
    X = stats.zscore(X)
    for i in methods:
        for j in metrics:
            print "Method:{0} Metrics: {1}".format(i, j)
            labels.append(i)
            Maxclust = 2
            Z = linkage(X, method=i, metric=j)
            # Compute and display clusters by thresholding the dendrogram
            cls = fcluster(Z, criterion='maxclust', t=Maxclust)
            measure = mcs.f1_score(y, cls, average='weighted')
            fm.append(measure)

    figure()
    plot(np.arange(len(labels)), fm)
    xticks(np.arange(len(labels)), labels, rotation='vertical')
    show()