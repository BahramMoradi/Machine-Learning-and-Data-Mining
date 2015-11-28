__author__ = 'Bahram'
from Reading_data import*
from sklearn import cross_validation

OUTER_K=5
INNER_K=10
Error_train = np.empty((OUTER_K,1))
Error_test = np.empty((OUTER_K,1))
Error_train_inner = np.empty((OUTER_K,1))
Error_test_inner = np.empty((OUTER_K,1))

OUTER_CV=cross_validation.KFold(N,OUTER_K,shuffle=True)

k=0
for train_index, test_index in OUTER_CV:
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    INNER_CV=cross_validation.KFold(N,INNER_K,shuffle=True)








