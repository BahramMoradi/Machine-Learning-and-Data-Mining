# requires Reading data

from Reading_data import *

from pylab import *
import scipy.linalg as linalg

# Subtract mean value from data
#Y = X - np.ones((N,1))*X.mean(0)
Y = (X - np.ones((N,1))*X.mean(0))/X.std(0)

# PCA by computing SVD of Y
U,S,V = linalg.svd(Y,full_matrices=False)
V = mat(V).T

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

# Compute cumulative variance explained by principal components
rhoCum = np.zeros((len(rho),1))
for i in range(0,len(rho)):
    rhoCum[i]=rhoCum[i-1]+rho[i]

# Plot variance explained
figure()
plot(range(1,len(rho)/2+1),rho[0:len(rho)/2],'o-')
title('Variance explained by principal components');
xlabel('Principal component');
ylabel('Variance explained');
show()


# Plot variance explained
figure()
plot(range(1,len(rhoCum)/2+1),rhoCum[0:len(rhoCum)/2],'o-')
title('Cumulative variance explained by principal components');
xlabel('Principal component');
ylabel('Variance explained');
show()

print ("Word freq 85 is the highest impact in the principal component")