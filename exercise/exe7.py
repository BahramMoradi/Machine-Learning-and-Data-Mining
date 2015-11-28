__author__ = 'Bahram'
from Project2 import Data
from sklearn.cross_validation import train_test_split
from pylab import *
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation

data=Data()
X=data.X
y=data.y
attributeNames=data.attributeNames
classNames=data.classNames
N=data.N
M=data.M
C=len(classNames)



def exe711(k):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)
    # Plot the training data points (color-coded) and test data points.
    f=figure();
    f.hold(True);
    styles = ['.b', '.r']
    for c in range(C):
        class_mask = (y_train==c).ravel()
        plot(X_train[class_mask,0], X_train[class_mask,1], styles[c])


    # K-nearest neighbors
    K=k

    # Distance metric (corresponds to 2nd norm, euclidean distance).
    # You can set dist=1 to obtain manhattan distance (cityblock distance).
    dist=2

    # Fit classifier and classify the test points
    knclassifier = KNeighborsClassifier(n_neighbors=K, p=dist);
    knclassifier.fit(X_train, y_train);
    y_est = knclassifier.predict(X_test);


    # Plot the classfication results
    styles = ['ob', 'or']
    for c in range(C):
        class_mask = (y_est==c)
        plot(X_test[class_mask,0], X_test[class_mask,1], styles[c], markersize=10)
        plot(X_test[class_mask,0], X_test[class_mask,1], 'kx', markersize=8)
    title('Synthetic data classification - KNN, K : {0}'.format(k));

    # Compute and plot confusion matrix
    cm = confusion_matrix(y_test.ravel(), y_est);
    accuracy = 100*cm.diagonal().sum()/cm.sum(); error_rate = 100-accuracy;
    figure(2);
    imshow(cm, cmap='binary', interpolation='None');
    colorbar()
    xticks(range(C)); yticks(range(C));
    xlabel('Predicted class'); ylabel('Actual class');
    title('Confusion matrix (Accuracy: {0}%, Error Rate: {1}%, K :{2})'.format(accuracy, error_rate,k));

    show()

def ex712():
        # Maximum number of neighbors
    L=40
    N=10
    CV = cross_validation.KFold(n=len(y), n_folds=N, shuffle=True,
                               random_state=None)
    errors = np.zeros((N,L))
    i=0
    for train_index, test_index in CV:
        print('Crossvalidation fold: {0}/{1}'.format(i+1,N))

        # extract training and test set for current CV fold
        X_train = X[train_index,:]
        y_train = y[train_index,:]
        X_test = X[test_index,:]
        y_test = y[test_index,:]

        # Fit classifier and classify the test points (consider 1 to 40 neighbors)
        for l in range(1,L+1):
            knclassifier = KNeighborsClassifier(n_neighbors=l);
            knclassifier.fit(X_train, ravel(y_train));
            y_est = knclassifier.predict(X_test);
            errors[i,l-1] = np.sum(y_est[0]!=y_test[0,0])

        i+=1

    # Plot the classification error rate
    figure()
    plot(100*sum(errors,0)/N)
    xlabel('Number of neighbors')
    ylabel('Classification error rate (%)')
    show()

def ex713():
    L=40
    # Cross-validation not necessary. Instead, compute matrix of nearest neighbor
    # distances between each pair of data points ..
    knclassifier = KNeighborsClassifier(n_neighbors=L+1).fit(X, ravel(y))
    neighbors = knclassifier.kneighbors(X)
    # .. and extract matrix where each row contains class labels of subsequent neighbours
    # (sorted by distance)
    ndist, nid = neighbors[0], neighbors[1]
    print len(ndist)
    print len(nid)
    print "="*20
    nclass = y[nid].flatten().reshape(N,L+1)

    # Use the above matrix to compute the class labels of majority of neighbors
    # (for each number of neighbors l), and estimate the test errors.
    errors = np.zeros(L)
    nclass_count = np.zeros((N,C))
    for l in range(1,L+1):
        for c in range(C):
            nclass_count[:,c] = sum(nclass[:,1:l+1]==c,1).A.ravel()
        y_est = np.argmax(nclass_count,1);
        errors[l-1] = (y_est!=y.A.ravel()).sum()


    # Plot the classification error rate
    figure(1)
    plot(100*errors/N)
    xlabel('Number of neighbors')
    ylabel('Classification error rate (%)')

    figure(2)
    imshow(nclass, cmap='binary', interpolation='None'); xlabel("k'th neighbor"); ylabel('data point'); title("Neighbors class matrix");

    show()


ex712()
#ex713()