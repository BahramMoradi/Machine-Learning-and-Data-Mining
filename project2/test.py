__author__ = 'Bahram'
import numpy as np
import matplotlib.pyplot as plt

N = 3
ind = np.arange(N)  # the x locations for the groups
#width = 0.05       # the width of the bars
margin = 0.045
width = (1.-2.*margin)/N
fig = plt.figure()
ax = fig.add_subplot(111)

yvals = [4, 9, 2]
rects1 = ax.bar(ind, yvals, width, color='r')
zvals = [1,2,3]
rects2 = ax.bar(ind+width, zvals, width, color='g')


ax.set_ylabel('Scores')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('4', '5', '6') )
ax.legend( (rects1[0], rects2[0]), ('y', 'z') )

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '{0:.3}'.format(float(h)),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)


plt.show()