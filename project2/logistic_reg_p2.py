__author__ = 'Bahram'
from Reading_data import *
import sklearn.linear_model as lm
from sklearn import cross_validation
from pylab import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score,confusion_matrix
import os
import time
#Ma=[];MA=[]
#Ba=[];BA=[]

def logisticR(k1=4, k2=10, lambdas_range=[-5, 1]):

    tvst_folder=create_new_dir('tvst_')
    conf_folder=create_new_dir('confu_')

    K = 5
    cv2 = 10
    #CV = cross_validation.KFold(N, K)
    lambdas = np.power(10., range(-7, 8))
    Error_train = np.empty((K, 1))
    Error_test = np.empty((K, 1))
    chosen_lambdas =np.empty((K, 1))
    Error_test_inner=np.empty((K,1))

    CV = cross_validation.KFold(N, K, shuffle=True)
    k = 0
    for train_index, test_index in CV:
        # extract training and test set for current CV fold
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        opt_lambda, train_error, test_error,log_loss_train_vs_lambda,log_loss_test_vs_lambda = inner_evaluation(X_train, y_train, lambdas, cv2)
        #plot_log_loss(opt_lambda,lambdas,log_loss_train_vs_lambda,log_loss_test_vs_lambda)

        model=lm.logistic.LogisticRegression(C=opt_lambda)
        model.fit(X_train,y_train)
        y_est_test=model.predict(X_test)
        cm=confusion_matrix(y_test,y_est_test)
        #plot_probablity(model,y_train)
        accuracy = 100.0*float(cm.diagonal().sum())/float(cm.sum())
        error_rate = 100-float(accuracy);
        Error_test[k]=error_rate
        # calculation for train
        y_est_train=model.predict(X_train)
        train_cm=confusion_matrix(y_train,y_est_train)
        accuracy_train = 100.0*float(cm.diagonal().sum())/float(cm.sum())
        error_rate_train = 100-float(accuracy);
        Error_train[k]=error_rate_train
        chosen_lambdas[k]=opt_lambda



        Error_test_inner[k]=np.min(test_error)
        plot_optimize_parameter(tvst_folder,k,opt_lambda, lambdas, train_error, test_error)
        plot_confusion_matrix(k,cm,conf_folder,"Lambda {0}, Accuracy : {1:.2f}%, Error rate : {2:.2f}%, {3}. fold".format(opt_lambda,accuracy,error_rate,k+1))

        k+=1
    #plot_result(chosen_lambdas,Error_train,Error_test)
    print "inner loop result"
    print Error_test_inner
    print "performance in the outer loop"
    print chosen_lambdas
    print Error_test


def inner_evaluation(X, y, lamdas, kfold):
    # test
    #using_mode= np.empty((kfold, len(lamdas)))
    opt_inner= np.empty((kfold,1))
    #test
    miss_class_train_error = np.empty((kfold, len(lamdas)))
    miss_class_test_error = np.empty((kfold, len(lamdas)))
    log_loss_train = np.empty((kfold, len(lamdas)))
    log_loss_test = np.empty((kfold, len(lamdas)))
    CV = cross_validation.KFold(X.shape[0], kfold, shuffle=True)
    k = 0
    for train_index, test_index in CV:
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        for i in range(0, len(lamdas)):
            model = lm.logistic.LogisticRegression(C=lamdas[i])
            model = model.fit(X_train, y_train)
            y_test_est = model.predict(X_test)
            lg_loss_test = log_loss(y_test, y_test_est)
            test_error_rate = sum(np.abs(np.mat(y_test_est).T - y_test)) / float(len(y_test_est))
            miss_class_test_error[k, i] = test_error_rate
            log_loss_test[k,i]=lg_loss_test

            y_train_est = model.predict(X_train)
            lg_loss_train = log_loss(y_train, y_train_est)
            train_error_rate = sum(np.abs(np.mat(y_train_est).T - y_train)) / float(len(y_train_est))
            miss_class_train_error[k, i] = train_error_rate
            log_loss_train[k,i]=lg_loss_train
            #test
            #using_mode[k,i]=calc_error_rate(y_test,y_test_est)
            #test
        #opt_inner[k]=lamdas[np.argmin(using_mode[k,:],0)]
        k += 1
    chosen_lambda = lamdas[np.argmin(np.mean(miss_class_test_error, 0))]
    miss_class_train_err_vs_lambda = np.mean(miss_class_train_error, 0)
    miss_class_test_err_vs_lambda = np.mean(miss_class_test_error, 0)
    log_loss_test_vs_lambda= np.mean(log_loss_test, 0)
    log_loss_train_vs_lambda= np.mean(log_loss_train, 0)

    #test
    #from scipy import stats
    #m=int(stats.mode(opt_inner )[0])
    #print "using model BM : ",chosen_lambda
    #print "Using model M : ",m
    #Ma.append(m)
    #Ba.append(chosen_lambda)

    #test

    return chosen_lambda, miss_class_train_err_vs_lambda, miss_class_test_err_vs_lambda,log_loss_train_vs_lambda,log_loss_test_vs_lambda


def plot_confusion_matrix(nf,cm,save_to_folder,tit='Confusion matrix', cmap=plt.cm.Blues):
    f = figure();
    f.hold(True)
    imshow(cm, interpolation='nearest', cmap=cmap)
    title(tit)
    colorbar()
    tick_marks = np.arange(C)
    xticks(tick_marks, classNames, rotation=45)
    yticks(tick_marks, classNames)
    tight_layout()
    ylabel('True Class')
    xlabel('Predicted Class')
    savefig(save_to_folder+'/confusion{0}.eps'.format(nf+1), format='eps', dpi=1000)
    savefig(save_to_folder+'/confusion{0}.png'.format(nf+1), format='png')


def plot_probablity(model, x):
    # no-spam /spam  0/1
    y_est_no_spam_prob = model.predict_proba(X)[:, 0]
    f = figure();
    f.hold(True)
    class0_ids = nonzero(y == 0)[0].tolist()[0]
    plot(class0_ids, y_est_no_spam_prob[class0_ids], '.g')
    class1_ids = nonzero(y == 1)[0].tolist()[0]
    plot(class1_ids, y_est_no_spam_prob[class1_ids], '.r')
    xlabel('Data object (Mail sample)');
    ylabel('Predicted prob. of class no-spam');
    legend(['no-spam', 'spam'])
    ylim(-0.5, 1.5)



def plot_optimize_parameter(save_to_folder,nf,opt_lambda, lambdas, miss_class_train, miss_class_test):
    min_miss_class=np.min(miss_class_test)
    min_class_lambda=lambdas[np.argmin(miss_class_test)]

    f = figure();
    f.hold(True)
    title('Optimal lambda: {0}, Accuracy: {1:.2f}%, Error rate: {2:.2f} %'.format(opt_lambda,100.0-100.0*min_miss_class,100.0*min_miss_class))
    xscale('log')
    plot(lambdas, 100*(miss_class_train.T), 'b.-', lambdas, 100*(miss_class_test.T), 'r.-')
    plot(min_class_lambda,100*( min_miss_class), marker='o',c='black')
    xlabel('Regularization factor')
    ylabel('Misclassification (crossvalidation)%')
    legend(['Misclassification training error', 'Misclassification test error'])
    savefig(save_to_folder+'/miss_class{0}.eps'.format(nf+1), format='eps', dpi=1000,bbox_inches='tight')
    savefig(save_to_folder+'/miss_class{0}.png'.format(nf+1), format='png',bbox_inches='tight')

def plot_log_loss(opt_lambda,lambdas,log_loss_train,log_loss_test):
    f = figure();
    f.hold(True)
    title('Optimal lambda = {0}'.format(opt_lambda))
    loglog(lambdas, log_loss_train.T, 'b.-', lambdas, log_loss_test.T, 'r.-')
    xlabel('Regularization factor')
    ylabel('Log loss (crossvalidation)')
    legend(['Log loss training error', 'Log loss test error'])

def plot_result(lambdas,train_error,test_error):
    min_train=np.min(train_error)
    min_train_lambda=lambdas[np.argmin(train_error)]
    min_test=np.min(test_error)
    min_test_lambda=lambdas[np.argmin(test_error)]
    lam1,train_error=combind_sort(lambdas,train_error)
    lam2,test_error=combind_sort(lambdas,test_error)
    f = figure();
    f.hold(True)
    title('Validation result')
    xscale('log')
    plot(lam1, train_error, 'b.-', lam2, test_error, 'r.-')
    plot(min_train_lambda, min_train, marker='o',c='black')
   # xscale('log')
    plot(min_test_lambda, min_test, marker='o',c='black')
    #xscale('log')
    xlabel('Regularization factor')
    ylabel('Error rate (crossvalidation)')
    legend(['Training error', 'Test error'])
    savefig('C:/Users/Bahram/Desktop/fig/validation.eps', format='eps', dpi=1000,bbox_inches='tight')
    savefig('C:/Users/Bahram/Desktop/fig/validation.png', format='png',bbox_inches='tight')



def combind_sort(ls1,ls2):
    import operator
    new_ls1=list()
    new_ls2=list()
    ls=list()
    for x,y in zip(ls1,ls2):
        ls.append((x,y))
    sols=sorted(ls,key=operator.itemgetter(0))
    for itm in sols:
        new_ls1.append(itm[0])
        new_ls2.append(itm[1])
    return new_ls1,new_ls2

def create_new_dir(prefix):
    dir = prefix+str(long(time.time()))
    new_dir = 'C:/Users/Bahram/Google Drev/Itroduction to data mining/project2/fig/lr/' +dir
    os.mkdir(new_dir)
    return new_dir


def calc_error_rate(y_test,y_est):
    cm=confusion_matrix(y_test,y_est)
    error_rate=sum(np.abs(np.mat(y_est).T - y_test)) / float(len(y_test))
    return error_rate


def compute_rates(y_test,y_est):
    print "Computing rates...."
    from sklearn.metrics import recall_score
    cm=confusion_matrix(y_test,y_est,classNames)
    tp=cm[0,0]
    fp=cm[0,1]
    fn=cm[1,0]
    tn=cm[1,1]

    #The ability of the classifier to find all the positive samples
    #rc=recall_score(y_test, y_est, average='macro')
    rc=float(tp)/float(tp+fn)  #recall
    accuracy = float(100*cm.diagonal().sum())/float(cm.sum()); error_rate = 100-accuracy
    print "="*40
    print 'Accuracy : {0} %'.format(round(accuracy,2))
    print 'Error rate :{0} %'.format(round(error_rate,2))
    print 'Recall : {0}%'.format(round(rc,2))






logisticR()
show()




