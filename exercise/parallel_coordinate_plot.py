__author__ = 'Bahram'
from sklearn.decomposition import PCA
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from pandas import DataFrame
import xlrd
from pandas.tools.plotting import radviz, parallel_coordinates, andrews_curves
from scipy import stats
def plot_viz():
    Attr_nr=57  #how many attribute it should plot?
    xtik=[i for i in range(Attr_nr)]
    sheet = xlrd.open_workbook('spam_data.xls').sheet_by_index(0)
    header = sheet.row_values(0, 1, 59)
    classLabel = sheet.col_values(0, 1, 4602)
    X = np.mat(np.empty((4601, 57)))
    for i, col_id in enumerate(range(1, 58)):
        X[:, i] = np.mat(sheet.col_values(col_id, 1, 4602)).T
    header = header[:len(header) - 1]
    x_std= stats.zscore(X, ddof=1)
    df = pd.DataFrame(data=x_std[:,:Attr_nr], columns=header[:Attr_nr])
    df['Name'] = classLabel
    plt.figure()
    #radviz(df, 'Name')
    parallel_coordinates(df, 'Name', color=['blue','red'])
    trimed_header=[h.replace("word_freq_","",-1).replace(':','',-1) for h in header[:Attr_nr]]
    plt.xticks(xtik,trimed_header, rotation='vertical')
    #andrews_curves(df, 'Name', colormap='winter')
    #plt.show()
    #normalizeing
    #df_norm = (df - df.mean()) / (df.max() - df.min())
    #df_norm.plot(kind='box')
    plt.savefig('C://Users//Bahram//Desktop//E2015//pic//att20.eps', format='eps', dpi=1000,bbox_inches='tight')
    plt.savefig('C://Users//Bahram//Desktop//E2015//pic//att20.png', format='png', dpi=1000,bbox_inches='tight')
    plt.show()
plot_viz()