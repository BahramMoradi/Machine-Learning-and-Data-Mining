__author__ = 'Bahram'
import xlrd
import numpy as np

# open workbook
workbook = xlrd.open_workbook('../02450Toolbox_Python/Data/nanonose.xls')
sheet_names = workbook.sheet_names();
print sheet_names
# open sheet
sheet = workbook.sheet_by_index(0)
attributes = sorted(sheet.row_values(0, 3, 11))  # row 0 , from col 3 to col 10
class_names = set(sheet.col_values(0, 2, 92))
print attributes
print class_names
class_dict = dict(zip(class_names, range(len(attributes))))
y = np.mat([class_dict[value] for value in class_names]).T
X=np.mat(np.empty((90,8)))
for i, col_id in enumerate(range(3,11)):
    X[:,i] = np.mat(sheet.col_values(col_id,2,92)).T

# scatter plot
import matplotlib.pyplot as plt
plt.scatter(X[:,0],X[:,1])
plt.show()


