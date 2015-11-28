# -*- coding: utf8 -*-

import pandas as pd
from pylab import *
from sklearn import cross_validation, tree
import numpy as np
import xlrd
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
"""
xls_file = pd.ExcelFile("spam_data.xls")
df = xls_file.parse('Sheet1')
classLabel=df['class Label']
className=set(classLabel)
attributeNames=df.columns
y=df['spam_bool']
X=df.iloc[:,1:58]   #selection all rows and attribute columns
N,M=X.shape
"""

 # Load xls sheet with data
doc = xlrd.open_workbook('spam_data.xls').sheet_by_index(0)

    #last column of the attributes considered
cA = 58 #58 55
    #number of the attributes considered
nA = 57 #57 54

    # Extract attribute names
attributeNames = doc.row_values(0,1,cA)

    # Extract class names to python list,
    # then encode with integers (dict)
classLabels = doc.col_values(0,1,4602)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,range(2)))

    # Extract vector y, convert to NumPy matrix and transpose
    # This is the posicion of the ClassName (Water,...) in the excel table
y = np.mat([classDict[value] for value in classLabels]).T

    # Preallocate memory, then extract excel data to matrix X
X = np.mat(np.empty((4601,nA)))
for i, col_id in enumerate(range(1,cA)):
    X[:,i] = np.mat(doc.col_values(col_id,1,4602)).T

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)



def model_comlexity_and_error_rate(complexity_from,complexity_to,test_proportion):
# Tree complexity parameter - constraint on maximum depth
    tc = np.arange(complexity_from,complexity_to, 1)

# Simple holdout-set crossvalidation
    #test_proportion = 0.5
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=test_proportion)

    # Initialize variables
    Error_train = np.empty((len(tc),1))
    Error_test = np.empty((len(tc),1))

    for i, t in enumerate(tc):
        # Fit decision tree classifier, Gini split criterion, different pruning levels
        dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
        dtc = dtc.fit(X_train,y_train.ravel())

        # Evaluate classifier's misclassification rate over train/test data
        y_est_test = dtc.predict(X_test)
        y_est_train = dtc.predict(X_train)
        misclass_rate_test = sum(np.abs(np.mat(y_est_test).T - y_test)) / float(len(y_est_test))
        misclass_rate_train = sum(np.abs(np.mat(y_est_train).T - y_train)) / float(len(y_est_train))
        Error_test[i], Error_train[i] = misclass_rate_test, misclass_rate_train

    f = figure(); f.hold(True)
    plot(tc, Error_train)
    plot(tc, Error_test)
    xlabel('Model complexity (max tree depth)')
    ylabel('Error (misclassification rate)')
    legend(['Error_train','Error_test'])
    show()



def cross_validation_method(range_from=2, range_to=21,K=3):
    # Tree complexity parameter - constraint on maximum depth
    tc = np.arange(range_from,range_to, 1)

    # K-fold crossvalidation
    #K = 10
    CV = cross_validation.KFold(N,K,shuffle=True)

    # Initialize variable
    Error_train = np.empty((len(tc),K))
    Error_test = np.empty((len(tc),K))

    k=0
    for train_index, test_index in CV:
        print('Computing CV fold: {0}/{1}..'.format(k+1,K))

        # extract training and test set for current CV fold
        X_train, y_train = X[train_index,:].A, y[train_index,:].A
        X_test, y_test = X[test_index,:].A, y[test_index,:].A

        for i, t in enumerate(tc):
            # Fit decision tree classifier, Gini split criterion, different pruning levels
            dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
            dtc = dtc.fit(X_train,y_train.ravel())
            y_est_test = dtc.predict(X_test)
            y_est_train = dtc.predict(X_train)
            # Evaluate misclassification rate over train/test data (in this CV fold)
            misclass_rate_test = sum(np.abs(np.mat(y_est_test).T - y_test)) / float(len(y_est_test))
            misclass_rate_train = sum(np.abs(np.mat(y_est_train).T - y_train)) / float(len(y_est_train))
            Error_test[i,k], Error_train[i,k] = misclass_rate_test, misclass_rate_train
        k+=1

    """
    f = figure(); f.hold(True)
    boxplot(Error_test.T)
    xlabel('Model complexity (max tree depth)')
    ylabel('Test error across CV folds, K={0})'.format(K))

    f = figure(); f.hold(True)
    plot(tc, Error_train.mean(1))
    plot(tc, Error_test.mean(1))
    xlabel('Model complexity (max tree depth)')
    ylabel('Error (misclassification rate, CV K={0})'.format(K))
    legend(['Error_train','Error_test'])
    d=Error_train.mean(1)
    print d
    print d.min()
    show()
    """
    plt.subplot(1, 2, 1)
    plt.plot(tc,Error_train.mean(1))
    plot(tc, Error_test.mean(1))

    ls=list(Error_test.mean(1))
    min_index=ls.index(Error_test.mean(1).min())
    min_x=tc[min_index]
    min_y=Error_test.mean(1).min()
    plt.plot(min_x,min_y,'ko-',color = 'black')
    plt.annotate('Min test error rate ({0} , {1})'.format(min_x,min_y), xy=(min_x, min_y), xytext=(min_x, min_y+0.02),
            arrowprops=dict(facecolor='red', shrink=0.05))

    plt.xlabel('Model complexity (max tree depth)')
    plt.ylabel('Error (misclassification rate, CV K={0})'.format(K))
    plt.legend(['Error_train','Error_test'])

    plt.subplot(1, 2, 2)
    plt.boxplot(Error_test.T)
    plt.xlabel('Model complexity (max tree depth)')
    plt.ylabel('Test error across CV folds, K={0})'.format(K))
    plt.show()


def leave_one_out_validation(range_from=2,range_to=20):
    tc = np.arange(range_from,range_to, 1)

    # leave one out crossvalidation
    loo = cross_validation.LeaveOneOut(n=N)
    Error_train = np.empty((len(tc),N))
    Error_test = np.empty((len(tc),N))

    k=0
    for train_index, test_index in loo:
        print('Computing CV fold: {0}/{1}..'.format(k+1,N))

        # extract training and test set for current CV fold
        X_train, y_train = X[train_index,:].A, y[train_index,:].A
        X_test, y_test = X[test_index,:].A, y[test_index,:].A

        for i, t in enumerate(tc):
            # Fit decision tree classifier, Gini split criterion, different pruning levels
            dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
            dtc = dtc.fit(X_train,y_train.ravel())
            y_est_test = dtc.predict(X_test)
            y_est_train = dtc.predict(X_train)
            # Evaluate misclassification rate over train/test data (in this CV fold)
            misclass_rate_test = sum(np.abs(np.mat(y_est_test).T - y_test)) / float(len(y_est_test))
            misclass_rate_train = sum(np.abs(np.mat(y_est_train).T - y_train)) / float(len(y_est_train))
            Error_test[i,k], Error_train[i,k] = misclass_rate_test, misclass_rate_train
        k+=1


    f = figure(); f.hold(True)
    boxplot(Error_test.T)
    xlabel('Model complexity (max tree depth)')
    ylabel('Test error across CV folds, K={0})'.format(N))

    f = figure(); f.hold(True)
    plot(tc, Error_train.mean(1))
    plot(tc, Error_test.mean(1))
    xlabel('Model complexity (max tree depth)')
    ylabel('Error (misclassification rate, CV K={0})'.format(N))
    legend(['Error_train','Error_test'])

    show()


def feature_selection_with_scikit():
    """
    1-VarianceThreshold is a simple baseline approach to feature selection. It removes all features whose variance doesn’t
     meet some threshold. By default, it removes all zero-variance features, i.e. features that have the same value in
     all samples.
    2-Univariate feature selection works by selecting the best features based on univariate statistical tests.
     It can be seen as a preprocessing step to an estimator
    """
    p=0.8
    selector = VarianceThreshold(threshold=(p * (1 - p)))
    c=selector.fit_transform(X)
    print  "Number of the attribute before: ",X.shape[1]
    print "number of the attribute after:",c.shape[1]

    # selecting k best attribute instead of chi2, f_classif can also be used
    skb=SelectKBest(chi2, k=10)
    X_new=skb.fit_transform(X, y)
    attr=np.where(skb._get_support_mask(),attributeNames,'-1')

    print "Best attribute choosen with SelectKBest: "
    i=1
    for att in attr:
        if att!='-1':
            print i, ": ",att
            i+=1

    #using  ExtraTreesClassifier
    print "Using feature importance..."
    etc=ExtraTreesClassifier()
    etc.fit(X,y).transform(X)
    print etc.feature_importances_
    print etc.max_features
    print etc.max_depth

    print "Recursive feature selection : "
    from sklearn.svm import SVC
    import sklearn.linear_model as lm
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.feature_selection import RFECV
    # Create the RFE object and compute a cross-validated score.
    estim=lm.LinearRegression()
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=estim, step=1, cv=StratifiedKFold(y, 2),
                  scoring='accuracy')
    rfecv.fit(X, y)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()


#feature_selection_with_scikit()

cross_validation_method(range_from=2,range_to=21,K=20)
#model_comlexity_and_error_rate(2,21,0.5)
#leave_one_out_validation(2,20)   # this method takes to much time
import sklearn.feature_selection as fs
