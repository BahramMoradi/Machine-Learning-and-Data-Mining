__author__ = 'Bahram'
from pylab import *
#import all from reading data
from sklearn.metrics import confusion_matrix
C  # number of classes
nf # number of fold
classNames # name of classes nospam and spam (from reading data )
confusionMX=confusion_matrix(y_test,y_est_test)    #y_test: the true y from samples, y_est_test : the  predicted y( class)
def plot_confusion_matrix(nf,confusionMX,tit='Confusion matrix',classNames,C, cmap=plt.cm.Blues):
    f = figure();
    f.hold(True)
    imshow(confusionMX, interpolation='nearest', cmap=cmap)
    title(tit)
    colorbar()
    tick_marks = np.arange(C)
    xticks(tick_marks, classNames, rotation=45)
    yticks(tick_marks, classNames)
    tight_layout()
    ylabel('True label')
    xlabel('Predicted label')
    savefig('C:/Users/Bahram/Desktop/fig/confusion{0}.png'.format(nf), format='eps', dpi=1000)
    savefig('C:/Users/Bahram/Desktop/fig/confusion{0}.png'.format(nf), format='png')
