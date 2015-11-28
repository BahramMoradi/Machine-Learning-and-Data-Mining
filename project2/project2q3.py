__author__ = 'Bahram'
import numpy as np
import xlrd

# Load xls sheet with data
doc = xlrd.open_workbook('extra_data_point.xlsx').sheet_by_index(0)

#last column of the attributes considered
cA = 58 #58 55
#number of the attributes considered
nA = 57 #57 54

# Extract attribute names
attributeNames = doc.row_values(0,1,cA)

# Extract class names to python list,
# then encode with integers (dict)
classLabels = doc.col_values(0,1,2)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,range(2)))

# Extract vector y, convert to NumPy matrix and transpose
# This is the posicion of the ClassName (Water,...) in the excel table
y = np.mat([classDict[value] for value in classLabels]).T

# Preallocate memory, then extract excel data to matrix X
X = np.mat(np.empty((1,nA)))
for i, col_id in enumerate(range(1,cA)):
    X[:,i] = np.mat(doc.col_values(col_id,1,2)).T

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)
print X
