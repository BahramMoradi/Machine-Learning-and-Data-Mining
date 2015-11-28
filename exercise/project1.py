__author__ = 'Bahram'
from sklearn.decomposition import PCA
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from pandas import DataFrame
import xlrd
import matplotlib
from pandas.tools.plotting import radviz, parallel_coordinates, andrews_curves
from scipy import stats

matplotlib.style.use('ggplot')

'''
Different method for calculating PCA
'''


def line():
    print "=" * 80


def readData():
    # attributes
    cols = open("spam_data.csv", "rb").readlines()[0].split(',', -1)
    print "Attributes"
    print cols
    line()
    mat = np.loadtxt(open("spam_data.csv", "rb"), delimiter=",", skiprows=1)
    mat = np.matrix(mat)
    # remove class label (last column)
    mat = mat[:, :57]
    # transformation of last three column
    # for i in range(54, 57):
    # mx = mat[:, i].max()
    # mat[:, i] = mat[:, i] * 100 / mx
    return mat


def pca_raw_cov(X):
    '''
    PCA Based on the Covariance Matrix of the Raw Data
    :param X:
    :return:
    '''
    # Compute the covariance matrix
    cov_mat = np.cov(X.T)

    # Eigendecomposition of the covariance matrix
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

    # Make a list of (eigenvalue, eigenvector) tuples
    # and sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs_cov = [(np.abs(eig_val_cov[i]), eig_vec_cov[:, i]) for i in range(len(eig_val_cov))]
    eig_pairs_cov.sort()
    eig_pairs_cov.reverse()

    # Construct the transformation matrix W from the eigenvalues that correspond to
    # the k largest eigenvalues (here: k = 2)
    matrix_w_cov = np.hstack((eig_pairs_cov[0][1].reshape(57, 1), eig_pairs_cov[1][1].reshape(57, 1)))

    # Transform the data using matrix W
    X_raw_transf = matrix_w_cov.T.dot(X.T).T

    # Plot the data
    plt.scatter(X_raw_transf[:, 0], X_raw_transf[:, 1])
    plt.title('PCA based on the covariance matrix of the raw data')
    plt.show()


def pca_standardize_cov(X):
    '''
    PCA Based on the Covariance Matrix of the Standardized Data
    :param X:
    :return:
    '''

    # Standardize data
    X_std = StandardScaler().fit_transform(X)

    # Compute the covariance matrix
    cov_mat = np.cov(X_std.T)

    # Eigendecomposition of the covariance matrix
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

    # Make a list of (eigenvalue, eigenvector) tuples
    # and sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs_cov = [(np.abs(eig_val_cov[i]), eig_vec_cov[:, i]) for i in range(len(eig_val_cov))]
    eig_pairs_cov.sort()
    eig_pairs_cov.reverse()

    # Construct the transformation matrix W from the eigenvalues that correspond to
    # the k largest eigenvalues (here: k = 2)
    matrix_w_cov = np.hstack((eig_pairs_cov[0][1].reshape(57, 1), eig_pairs_cov[1][1].reshape(57, 1)))

    # Transform the data using matrix W
    X_std_transf = matrix_w_cov.T.dot(X_std.T).T

    # Plot the data
    plt.scatter(X_std_transf[:, 0], X_std_transf[:, 1])
    plt.title('PCA based on the covariance matrix after standardizing the data')
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.show()


def pca_cor(X):
    # Standardize data
    X_std = StandardScaler().fit_transform(X)

    # Compute the correlation matrix
    cor_mat = np.corrcoef(X.T)

    # Eigendecomposition of the correlation matrix
    eig_val_cor, eig_vec_cor = np.linalg.eig(cor_mat)

    # Make a list of (eigenvalue, eigenvector) tuples
    # and sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs_cor = [(np.abs(eig_val_cor[i]), eig_vec_cor[:, i]) for i in range(len(eig_val_cor))]
    eig_pairs_cor.sort()
    eig_pairs_cor.reverse()

    # Construct the transformation matrix W from the eigenvalues that correspond to
    # the k largest eigenvalues (here: k = 2)
    matrix_w_cor = np.hstack((eig_pairs_cor[0][1].reshape(57, 1), eig_pairs_cor[1][1].reshape(57, 1)))

    # Transform the data using matrix W
    X_transf = matrix_w_cor.T.dot(X_std.T).T

    # Plot the data
    plt.scatter(X_transf[:, 0], X_transf[:, 1])
    plt.title('PCA based on the correlation matrix of the raw data')
    plt.show()


def scikit_pca(X):
    # Standardize
    X_std = StandardScaler().fit_transform(X)

    # PCA
    sklearn_pca = PCA(n_components=2)
    X_transf = sklearn_pca.fit_transform(X_std)

    # Plot the data
    plt.scatter(X_transf[:, 0], X_transf[:, 1])
    plt.title('PCA via scikit-learn (using SVD)')
    plt.show()


# scikit_pca(readData())
def sort_matrix():
    mat = np.matrix([
        [1, 2, 3],
        [2, 3, 4],
        [1, 4, 9],
        [7, 9, 2]
    ])

    mat = np.array(mat)
    from operator import itemgetter

    mt = sorted(mat, key=itemgetter(2))


def read_csv():
    sheet = xlrd.open_workbook('spam_data.xls').sheet_by_index(0)
    header = sheet.row_values(0, 1, 59)
    classLabel = sheet.col_values(0, 1, 4602)
    X = np.mat(np.empty((4601, 57)))
    for i, col_id in enumerate(range(1, 58)):
        X[:, i] = np.mat(sheet.col_values(col_id, 1, 4602)).T

    header = header[:len(header) - 1]
    df = pd.DataFrame(data=X[:4, :4], columns=header[:4])
    # print df.describe()
    line()
    #print df.corr()
    line()
    #print df.cov()
    line()
    #print df[['word_freq_make:','word_freq_address:']].corr()


def plot_viz():
    Attr_nr=57
    xtik=[i for i in range(Attr_nr)]
    sheet = xlrd.open_workbook('spam_data.xls').sheet_by_index(0)
    header = sheet.row_values(0, 1, 59)
    classLabel = sheet.col_values(0, 1, 4602)
    X = np.mat(np.empty((4601, 57)))
    for i, col_id in enumerate(range(1, 58)):
        X[:, i] = np.mat(sheet.col_values(col_id, 1, 4602)).T
    header = header[:len(header) - 1]
    x_std= stats.zscore(X, ddof=1)
    df = pd.DataFrame(data=x_std[:,:Attr_nr], columns=header[:Attr_nr])
    df['Name'] = classLabel
    plt.figure()
    #radviz(df, 'Name')
    parallel_coordinates(df, 'Name', color=['blue','red'])
    trimed_header=[h.replace("word_freq_","",-1).replace(':','',-1) for h in header[:Attr_nr]]
    plt.xticks(xtik,trimed_header, rotation='vertical')
    #andrews_curves(df, 'Name', colormap='winter')
    #plt.show()
    #normalizeing
    #df_norm = (df - df.mean()) / (df.max() - df.min())
    #df_norm.plot(kind='box')
    plt.savefig('C://Users//Bahram//Desktop//E2015//pic//att20.eps', format='eps', dpi=1000,bbox_inches='tight')
    plt.savefig('C://Users//Bahram//Desktop//E2015//pic//att20.png', format='png', dpi=1000,bbox_inches='tight')
    plt.show()
plot_viz()



