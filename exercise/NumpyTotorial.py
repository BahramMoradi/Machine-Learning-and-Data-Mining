__author__ = 'Bahram'

import numpy as np
x=np.arange(0,10,1)
x=np.zeros(shape=(4,4))
print x


x=np.zeros(shape=(10,1))
print x

x=[1,2,3,4,5,6,7,8,9]


x = np.arange(9.).reshape(3, 3)
t=x[np.where( x > 5 )]
print t

x = np.arange(9.).reshape(3, 3)
t=np.where(x<4,x,0)
print np.count_nonzero(t)
print t
#====================================
print "================ nonzero()================="
x = np.arange(9.).reshape(9, 1)
print x[:,-1].nonzero()
print x[:,-1].nonzero()[0]
#=================================
print "=============== testing where =============="
sf=np.arange(0,10,1)
for i in range(4):
    print np.where(sf==i)[0]
