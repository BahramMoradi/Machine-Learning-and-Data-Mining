__author__ = 'Bahram'

from pylab import *
import time
import numpy as np
import os


def bars(test_error, train_error, hidden_units):
    N = len(hidden_units)
    ind = np.arange(N)  # the x locations for the groups
    margin = 0.050
    width = (1 - 2. * margin) / N

    fig = plt.figure()
    plt.hold(True)
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, test_error.flatten(), width, color='r')
    rects2 = ax.bar(ind + width, train_error.flatten(), width, color='b')
    ax.set_ylabel('Mean Square error')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(hidden_units)
    ax.legend((rects1[0], rects2[0]), ('Test Error', 'Training Error'))
    autolabel(ax, rects1)
    autolabel(ax, rects2)


def autolabel(ax, rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * float(h), '{0:.3}'.format(float(h)),
                ha='center', va='bottom')


def plot_error_vs_units(fold, train_error, test_error, neurons_range,dir_to_save):
    min_unit = neurons_range[np.argmin(test_error)]
    min_error = np.min(test_error)
    f = figure();
    f.hold(True)
    title('Optimal nr. of neurons : {0}'.format(min_unit))
    plot(neurons_range, train_error.T, 'b.-', neurons_range, test_error.T, 'r.-')
    plot(min_unit, min_error, marker='o', c='black')
    xticks(neurons_range)
    xlabel('Number of neurons ')
    ylabel('Mean-square error(crossvalidation)')
    legend(['Mean-square training error', 'Mean-square test error'])
    savefig(dir_to_save + '/error_vs_neurons_{0}.eps'.format(fold + 1), format='eps', dpi=1000,bbox_inches='tight')
    savefig(dir_to_save + '/error_vs_neurons_{0}.png'.format(fold + 1), format='png',bbox_inches='tight')


def plot_result(k, neurons, y_est, y_test,dir_to_save):
    mean_square_error = np.power(y_est - y_test, 2).sum().astype(float) / y_test.shape[0]
    figure()
    hold(True)
    plot((y_est - y_test).A)
    #title('/est_y-test_y');
    title('CV-Fold:{0}, Neurons: {1}, Mean Square Error: {2:.3},  est_y-test_y'.format(k + 1, neurons,
                                                                                 mean_square_error));
    savefig(dir_to_save + '/yest_vs_ytest{0}.eps'.format(k + 1), format='eps', dpi=1000,bbox_inches='tight')
    savefig(dir_to_save + '/yest_vs_ytest_{0}.png'.format(k + 1), format='png',bbox_inches='tight')

def plot_featue_vs_residual(feature_name,feature_col,feature_idx,res,dir_to_save):
    figure()
    hold(True)
    plot(feature_col.A, res.A, '.r')
    xlabel(feature_name); ylabel('Residual')
    savefig(dir_to_save+'/feature_vs_res_{0}.eps'.format(feature_idx), format='eps', dpi=1000)
    savefig(dir_to_save+'/feature_vs_res_{0}.png'.format(feature_idx), format='png')

def create_new_dir(prefix):
    dir = prefix+str(long(time.time()))
    new_dir = 'C:/Users/Bahram/Google Drev/Itroduction to data mining/project2/fig/ann/' +dir
    os.mkdir(new_dir)
    return new_dir

