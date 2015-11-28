__author__ = 'Bahram'
import numpy as np
def line():
    print "="*40

M=np.mat([[2,4],
            [1,3],
            [0,0],
            [0,0]])
print "Matrix M"
print M
line()

MT=M.transpose()
print "Transpose of M"
print MT
line()

MMT=M*MT
w,v=np.linalg.eigh(MMT)
print "M* MT"
print MMT
print "Egenvector and egenvalue"
print w
print v
line()
MTM=MT*M
print "MT*M"
m,nv=np.linalg.eigh(MTM)
print m
print nv



