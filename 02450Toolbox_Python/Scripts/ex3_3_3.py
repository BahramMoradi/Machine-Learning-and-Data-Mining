# exercise 3.3.3

from pylab import *
import numpy as np
from scipy import stats

# Number of samples
N = 4000

# Mean
mu = 17

# Standard deviation
s = 2

# Number of bins in histogram
nbins =200

# Generate samples from the Normal distribution
X = np.mat(np.random.normal(mu,s,N)).T 
# or equally:
X = np.mat(np.random.randn(N)).T * s + mu

# Plot the histogram
f = figure()
f.hold()
title('Normal distribution N = {0}, mu = {1}, s = {2} and bins ={3}'.format(N,mu,s,nbins))
hist(X, bins=nbins, normed=True)

# Over the histogram, plot the theoretical probability distribution function:
x = linspace(X.min(), X.max(), 1000)
pdf = stats.norm.pdf(x,loc=17,scale=2)
plot(x,pdf,'.',color='red')

# Compute empirical mean and standard deviation
mu_ = X.mean()
s_ = X.std(ddof=1)

print "Theoretical mean: ", mu
print "Theoretical std.dev.: ", s
print "Empirical mean: ", mu_
print "Empirical std.dev.: ", s_

show()
