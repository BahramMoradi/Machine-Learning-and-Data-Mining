# Load xls sheet with data
import xlrd, numpy as np
import scipy
from pylab import *
from Reading_data import *
from scipy import stats

X = stats.zscore(X, ddof=1)

co = np.corrcoef(X.T)
for i in range(co.shape[0]):
    for j in range(co.shape[0]):
        val = co[i][j]
        if val < 1 and val > 0.80:
            print i," ",j
            print "({0}-{1})={2}".format(attributeNames[i],attributeNames[j],val)


print "="*100
print scipy.stats.pearsonr(X[:,31],X[:,33])
print "="*100
print X.shape
color=['blue','red']
P=0
figure(figsize=(20, 18))
hold(True)
for i in [31,33,39]:
    for j in [33,39,31]:
        subplot(3,3,P)
        plt.scatter(X[:,i],X[:,j])
        xlabel(attributeNames[i])
        ylabel(attributeNames[j])
        P+=1
show()

"""
colors = ['blue', 'red']
n=10
figure(figsize=(12, 10))
hold(True)
for m1 in range(M):
    for m2 in range(M):
        subplot(n, n, m1 * M + m2 + 1)
        for c in range(C):
            class_mask = (y == c).A.ravel()
            plot(array(X[class_mask, m2]), array(X[class_mask, m1]), '.')
            if m1 == M - 1:
                xlabel(attributeNames[m2])
            else:
                xticks([])
            if m2 == 0:
                ylabel(attributeNames[m1])
            else:
                yticks([])
                # ylim(0,X.max()*1.1)
                #xlim(0,X.max()*1.1)
legend(classNames)

show()
"""





