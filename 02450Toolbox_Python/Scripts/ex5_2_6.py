# exercise 5.2.6
from pylab import *
import sklearn.linear_model as lm

# requires data from exercise 5.1.4
from ex5_1_5 import *

# Fit logistic regression model
model = lm.logistic.LogisticRegression()
model = model.fit(X, y.A.ravel())

# Classify wine as White/Red (0/1) and assess probabilities
y_est = model.predict(X)
print y_est
print "="*40
y_est_white_prob = model.predict_proba(X)[:, 0] 

# Define a new data object (new type of wine), as in exercise 5.1.7
x = np.array([6.9, 1.09, .06, 2.1, .0061, 12, 31, .99, 3.5, .44, 12])
# Evaluate athe probability of x being a white wine (class=0) 
x_class = model.predict_proba(x)[0,0]

# Evaluate classifier's misclassification rate over entire training data
misclass_rate = sum(np.abs(np.mat(y_est).T - y)) / float(len(y_est))

# Display classification results
print('\nProbability of given sample being a white wine: {0:.4f}'.format(x_class))
print('\nOverall misclassification rate: {0:.3f}'.format(misclass_rate))

f = figure(); f.hold(True)
class0_ids = nonzero(y==0)[0].tolist()[0]
plot(class0_ids, y_est_white_prob[class0_ids], '.y')
class1_ids = nonzero(y==1)[0].tolist()[0]
plot(class1_ids, y_est_white_prob[class1_ids], '.r')
xlabel('Data object (wine sample)'); ylabel('Predicted prob. of class White');
legend(['White', 'Red'])
ylim(-0.5,1.5)

show()

