__author__ = 'Bahram'
import numpy as np

mat=np.matrix([[1,2,3,4],
               [4,3,2,1]])
print np.corrcoef(mat)
#help(np.cov)