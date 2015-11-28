__author__ = 'Bahram'
from Reading_data import*
from pylab import *
import numpy as np
from sklearn import cross_validation
import scipy.linalg as linalg
from scipy import stats
from sklearn.naive_bayes import MultinomialNB
import sklearn.linear_model as lm
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

def fit_and_compare(K1=10):
    K=10
    #tree_prf=np.empty(K)
    #logistic_prf=np.empty(K)
    #KNN_prf=np.empty(K)
    #nb_prf=np.empty(K)
    NB_INDEX=0;DT_INDEX=1;LR_INDEX=2;KNN_INDEX=3;
    Accuracy_mat=np.empty((K,4))
    Error_mat=np.empty((K,4))

    #======optimized parameter
    TREE_DEPTH=15
    SAMPLE_SPLITE=20
    LOGISTIC_LAMBDA=10
    KNN_K=5
    KNN_DIST=1

    OUTER_CV = cross_validation.KFold(N, K1)
    k=0
    for train_index, test_index in OUTER_CV:
        print('\nOuter Crossvalidation fold: {0}/{1}'.format(k + 1, K))


        # extract training and test set for current CV fold
        X_train = X[train_index, :]
        y_train = y[train_index, :]
        X_test = X[test_index, :]
        y_test = y[test_index, :]

    #NB
        nb_classifier = MultinomialNB(alpha=1.0, fit_prior=True)
        nb_classifier.fit(X_train, y_train)
        y_est_prob = nb_classifier.predict_proba(X_test)
        y_est = np.argmax(y_est_prob,1)
        # a.argmax ???? what this does
        #nb_prf[k] = np.sum(y_est.ravel()!=y_test.ravel(),dtype=float)/y_test.shape[0] # ????
        accuracy,error_rate= computet_accoracy_and_error(y_est,y_test)
        Accuracy_mat[k,NB_INDEX]=accuracy
        Error_mat[k,NB_INDEX]=error_rate

    #tree classifire
        dtc = tree.DecisionTreeClassifier(criterion='gini',max_depth=TREE_DEPTH, min_samples_split=SAMPLE_SPLITE)
        dtc = dtc.fit(X_train,y_train)
        y_est=dtc.predict(X_test)
        accuracy,error_rate=computet_accoracy_and_error(y_est,y_test)
        Accuracy_mat[k,DT_INDEX]=accuracy
        Error_mat[k,DT_INDEX]=error_rate

    #Logistic
        model=lm.logistic.LogisticRegression(C=LOGISTIC_LAMBDA)
        model.fit(X_train,y_train)
        y_est=model.predict(X_test)
        accuracy ,error_rate=computet_accoracy_and_error(y_est,y_test)
        Accuracy_mat[k,LR_INDEX]=accuracy
        print error_rate
        Error_mat[k,LR_INDEX]=error_rate

     # Distance metric (corresponds to 2nd norm, euclidean distance).
    # You can set dist=1 to obtain manhattan distance (cityblock distance).
        knclassifier = KNeighborsClassifier(n_neighbors=KNN_K, p=KNN_DIST);
        knclassifier.fit(X_train, y_train);
        y_est = knclassifier.predict(X_test);
        accuracy,error_rate= computet_accoracy_and_error(y_est,y_test)
        Accuracy_mat[k,KNN_INDEX]=accuracy
        Error_mat[k,KNN_INDEX]=error_rate

        k+=1

    total_accuracy=np.mean(Accuracy_mat, 0)

    figure();
    subplot(2,1,1)
    plot(np.arange(1,11),Accuracy_mat[:,NB_INDEX],'g.-')
    plot(np.arange(1,11),Accuracy_mat[:,DT_INDEX],'b.-') #decision three
    plot(np.arange(1,11),Accuracy_mat[:,LR_INDEX],'r.-')
    plot(np.arange(1,11),Accuracy_mat[:,KNN_INDEX],'y.-')
    ylabel("Accuracy")
    xlabel("CV-Fold")
    legend(['MultinomialNB','Decision Tree','LogisticRegression','K-Nearest Neighbors'], loc=8)
    title('Accuracy of Classifiers (10-fold Cross validation ), ')
    #legend(['MultinomialNB','Decision Tree','K-Nearest Neighbors','LogisticRegression'])
    subplot(2,1,2)
    #plot(Error_mat[:,NB_INDEX],'g.-')
    #plot(Error_mat[:,DT_INDEX],'b.-')
    #plot(Error_mat[:,LR_INDEX],'r.-')
    #plot(Error_mat[:,KNN_INDEX],'y.-')
    # title('Classifiers Error rate')
    #legend(['MultinomialNB','Decision Tree','LogisticRegression','K-Nearest Neighbors'])
    # f2=figure()
    n = 4
    ind = np.arange(n)    # the x locations for the groups
    width = 0.80       # the width of the bars: can also be len(x) sequence
    p1 = plt.bar(ind, total_accuracy, width, color='b')
    plt.ylabel('Accuracy')
    plt.xlabel('Classifiers')
    #plt.title('Accuracy in 10-fold cross validation')
    plt.xticks(ind + width/2., ('NB', 'DT', 'LR', 'KNN'))
    plt.yticks(np.arange(0, 101, 10))
    #plt.legend((p1[0], p2[0]), ('Men', 'Women'))
    show()










def computet_accoracy_and_error(y_est,y_test):
    cm = confusion_matrix(y_test.A.ravel(), y_est)
    accuracy = float(100*cm.diagonal().sum())/float(cm.sum()); error_rate = 100-accuracy;
    return round(accuracy,2),round(error_rate,2)


fit_and_compare()

""""
      #=NB
        nb_classifier = MultinomialNB(alpha=1.0, fit_prior=True)
        nb_classifier.fit(X_train, np.ravel(y_train))
        y_est_prob = nb_classifier.predict_proba(X_test)
        y_est = np.argmax(y_est_prob,1) # a.argmax ???? what this does
        #nb_prf[k] = np.sum(y_est.ravel()!=y_test.ravel(),dtype=float)/y_test.shape[0] # ????
        accuracy,error_rate= computet_accoracy_and_error(y_est,y_test)
        pref_mat[k,nb]=accuracy
     #Logistic
        model=lm.logistic.LogisticRegression(C=LOGISTIC_LAMBDA)
        model.fit(X_train,y_train)
        y_est_test=model.predict(X_test)
        cm=confusion_matrix(y_test,y_est_test)
        #plot_probablity(model,y_train)
        accuracy ,error_rate=computet_accoracy_and_error(y_est,y_test)
        pref_mat[k,lr]=accuracy
    #knn
    # Distance metric (corresponds to 2nd norm, euclidean distance).
    # You can set dist=1 to obtain manhattan distance (cityblock distance).
    dist=computet_accoracy_and_error(y_est,y_test)
    knclassifier = KNeighborsClassifier(n_neighbors=KNN_K, p=KNN_DIST);
    knclassifier.fit(X_train, y_train);
    y_est = knclassifier.predict(X_test);
    cm = confusion_matrix(y_test.A.ravel(), y_est)
    accuracy,error_rate= computet_accoracy_and_error(y_est,y_test)
    pref_mat[k,knn]=accuracy
   """