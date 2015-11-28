__author__ = 'Bahram'
import numpy as np
import pandas as pan
import csv

def new_func():
    cols=open("spam_data.csv","rb").readlines()[0].split(',',-1)
    print cols
    mat=np.loadtxt(open("spam_data.csv","rb"),delimiter=",",skiprows=1)
    mat=np.matrix(mat)
    mat=mat[:,:57] # remove class label
    for i in range(54,57):
        max=mat[:,i].max()
        mat[:,i]=mat[:,i]*100/max
    for i in range(54,57):
        print "Max in col {0} :".format(i),mat[:,i].max()
    cor=np.corrcoef(mat)

new_func()



