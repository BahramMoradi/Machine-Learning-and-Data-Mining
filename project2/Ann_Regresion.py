from pylab import *
from scipy.io import loadmat
import neurolab as nl
from sklearn import cross_validation
import scipy.linalg as linalg
from scipy import stats
from project2plots import *
# loading data from spam file
import time
from sklearn.utils import shuffle


# Look at ex7_3_6
start_clock = time.clock()
start_wall = time.time()


def outer_cross():
    from Reading_data import *
    folder_prefix=['Outer_','Inner_']
    # Normalize data
    X = stats.zscore(X) #shuffling X
    x_index = [24, 25, 31, 33]
    y_index = 54
    doc_nr = 1000
    y = X[:doc_nr, y_index]  # attribute 54 is used for prediction purpose
    X = X[:doc_nr, x_index]  # these  attribute are used as input
    N, M = X.shape
    print "Max value in all attribute :", np.max(X)
    print "Min value in all attribute ", np.min(X)
    print "X shape : ",X.shape
    # ============ parameter for ann =====================
    n_hidden_units =np.arange(2,13)  # number of hidden units
    n_train = 2  # number of networks trained in each k-fold
    learning_goal = 100  # stop criterion 1 (train mse to be reached)
    max_epochs = 65  # stop criterion 2 (max epochs in training)
    show_error_freq = 10  # frequency of training status updates'
    # =========================================================
    # summary index
    BEST_NEURONS_NR = 0
    NET_TRAIN_ERROR = 1
    BEST_TRAIN_ERROR = 2
    Y_TEST = 3
    Y_TEST_EST = 4
    Y_TRAIN = 5
    Y_TRAIN_EST = 6
    MEAN_TEST_ERR_VS_UNITS=7
    MEAN_TRAIN_ERR_VS_UNIT=8
    # ==================================================================

    OUTER_K = 3
    INNER_K = 5
    best_hidden_units = np.zeros(OUTER_K)
    Train_errors = np.zeros(OUTER_K)
    Test_errors = np.zeros(OUTER_K)
     #creating to folder for diagrams
    folder_one=create_new_dir(folder_prefix[0])
    folder_two=create_new_dir(folder_prefix[1])
    #===================================================================
    summary_dict = {}  # {k:(best_neurons_nr,net_train_error,best_train_error,y_test,y_test_est,y_train,y_train_est,mean_test_err_vs_unit,mean_train_err_vs_unit)}
    f = 0;
    OUTER_CV = cross_validation.KFold(N, OUTER_K, shuffle=True)
    for train_index, test_index in OUTER_CV:
        print('\nOuter Crossvalidation fold: {0}/{1}'.format(f + 1, OUTER_K))

        # extract training and test set for current CV fold
        X_train = X[train_index, :]
        y_train = y[train_index, :]
        X_test = X[test_index, :]
        y_test = y[test_index, :]
        best_neurons_nr, mean_test_err_vs_unit, mean_train_err_vs_unit, mean_best_train_err_vs_unit = inner_cross(
            X_train, y_train, INNER_K, n_hidden_units, n_train, learning_goal, max_epochs, show_error_freq)

        bestnet, best_train_error, net_train_errors = find_best_network(X_train, y_train, n_train, best_neurons_nr,
                                                                        learning_goal, max_epochs, show_error_freq)
        y_test_est = bestnet.sim(X_test)
        y_train_est = bestnet.sim(X_train)

        summary_dict[f] = (
        best_neurons_nr, net_train_errors, best_train_error, y_test, y_test_est, y_train, y_train_est,
        mean_test_err_vs_unit, mean_train_err_vs_unit)
        #for index in x_index:
           # new_index=x_index.index(index)
           # plot_featue_vs_residual(attributeNames[index],X_train[:,new_index],index,abs(y_test_est-y_test),folder_one)
        f += 1

        # after the work is done then visualize it
        #create folder for saving files

    for key in summary_dict:
        value = summary_dict[key]
        plot_result(key , value[BEST_NEURONS_NR], value[Y_TEST], value[Y_TEST_EST],folder_one)
        plot_error_vs_units(key , value[MEAN_TRAIN_ERR_VS_UNIT],value[MEAN_TEST_ERR_VS_UNITS],n_hidden_units,folder_two)

        # plot_error_vs_units(f + 1, mean_train_err_vs_unit, mean_test_err_vs_unit, mean_best_train_err_vs_unit,n_hidden_units)
        # bars(mean_test_err_vs_unit,mean_train_err_vs_unit,n_hidden_units)


def inner_cross(X, y, K, n_hidden_units=[2, 3, 4, 5], n_train=2, learning_goal=100, max_epochs=100,
                show_error_freq=5):
    N, M = X.shape
    CV = cross_validation.KFold(N, K, shuffle=True)


    # Variable for classification error
    Best_error = np.empty((K, len(n_hidden_units)))
    Train_error = np.empty((K, len(n_hidden_units)))
    Test_error = np.empty((K, len(n_hidden_units)))

    errors = np.zeros(K)
    error_hist = np.zeros((max_epochs, K))
    k = 0
    for train_index, test_index in CV:
        print "=" * 40
        print('\nInner Crossvalidation fold: {0}/{1}'.format(k + 1, K))
        print "=" * 40

        # extract training and test set for current CV fold
        X_train = X[train_index, :]
        y_train = y[train_index, :]
        X_test = X[test_index, :]
        y_test = y[test_index, :]
        for j in range(0, len(n_hidden_units)):
            bestnet, best_train_error, net_train_errors = find_best_network(X_train, y_train, n_train,
                                                                            n_hidden_units[j], learning_goal,
                                                                            max_epochs,
                                                                            show_error_freq)
            print('Best train error: {0}...'.format(best_train_error))
            y_est = bestnet.sim(X_test)
            test_est_error = np.power(y_est - y_test, 2).sum().astype(float) / y_test.shape[0]
            Test_error[k, j] = test_est_error
            y_est_train = bestnet.sim(X_train)
            train_est_error = np.power(y_est_train - y_train, 2).sum().astype(float) / y_train.shape[0]
            Train_error[k, j] = train_est_error
            Best_error[k, j] = best_train_error
            test_print("Best train error [{0},{1}]={2}".format(k, j, best_train_error))
            test_print("Test error [{0},{1}]={2}".format(k, j, test_est_error))
            test_print("Train error [{0},{1}]={2}".format(k, j, train_est_error))

        k += 1
    best_nr_unit = n_hidden_units[np.argmin(np.mean(Test_error, 0))]
    mean_test_err_vs_unit = np.mean(Test_error, 0)
    mean_train_err_vs_unit = np.mean(Train_error, 0)
    mean_best_train_err_vs_unit = np.mean(Best_error, 0)
    return best_nr_unit, mean_test_err_vs_unit, mean_train_err_vs_unit, mean_best_train_err_vs_unit


def find_best_network(X_train, y_train, n_train, n_hidden_neurons, learning_goal, max_epochs, show_error_freq):
    N, M = X_train.shape
    networks = list()
    best_train_errors = list()
    Train_errors = list()
    for i in range(n_train):
        print('Training network {0}/{1} for hidden unit nr:{2}...'.format(i + 1, n_train, n_hidden_neurons))
        # Create randomly initialized network with 2 layers
        ann = nl.net.newff([[-3, 3]] * M, [n_hidden_neurons, 1], [nl.trans.TanSig(), nl.trans.PureLin()])
        # train network
        train_error = ann.train(X_train, y_train, goal=learning_goal, epochs=max_epochs, show=show_error_freq)
        networks.append(ann)
        Train_errors.append(train_error)
        best_train_errors.append(train_error[-1])
    return networks[np.argmin(best_train_errors)], np.min(best_train_errors), Train_errors[np.argmin(best_train_errors)]


"""

 # Print the average least squares error
 print('Mean-square error: {0}'.format(mean(errors)))

 figure();
 subplot(2, 1, 1);
 bar(range(0, K), errors);
 title('Mean-square errors');
 subplot(2, 1, 2);
 plot(error_hist);
 title('Training error as function of BP iterations');
 figure();
 subplot(2, 1, 1);
 plot(y_est);
 plot(y_test.A);
 title('Last CV-fold: est_y vs. test_y');
 subplot(2, 1, 2);
 plot((y_est - y_test).A);
 title('Last CV-fold: prediction error (est_y-test_y)');
 show()
 show()
 """


def plot_mean_square(units, errors):
    f = figure();
    f.hold(True)
    bar(units, errors);
    title('Mean-square errors');


def test_print(what):
    print"=" * 40
    print what
    print "=" * 40




outer_cross()
end_clock = time.clock()
elapsed_clock = end_clock - start_clock
end_wall = time.time()
elapsed_wall = end_wall - start_wall
print "Processing time: ", elapsed_clock
print "Processing time (wall)", elapsed_wall
print "Processing time: {0}:{1}".format(int(elapsed_wall / 60), int(elapsed_wall % 60))

show()