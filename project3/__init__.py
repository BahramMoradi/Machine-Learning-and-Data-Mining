__author__ = 'Bahram'
"""
import numpy as np

mt=np.zeros((10,1))
for i in range(0,10):
    r=np.random.ranf()
    mt[i]=100*r
print mt
print "="*20
print "index of min: ",np.argmin(mt)
print mt[np.argmin(mt)]


print "="*40
KRange = range(1,11)
print KRange
print KRange[3]
"""

import time
import datetime
from pylab import *
import numpy as np
t=time.time()
st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
"""
a=['abslut-best','best-ever','complete-ever','distance-ever']
b=['e','f','g']
result=[1,22,10,18]

methods = ['single', 'complete', 'average', 'weighted']
metrics = ['euclidean', 'cityblock', 'chebychev']
combind=[]
for mth in methods:
    for met in metrics:
        combind.append((mth,met))
figure(figsize=(14,9))
hold(True)
plot(range(len(combind)),range(len(combind)))
xlabel("meth-metr")
xticks(range(len(combind)),combind,rotation=20)
ylabel("purity")
show()
"""
figure()
plot(2,2,'o',color='red')
show()








