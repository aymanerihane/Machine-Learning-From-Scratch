import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import random

x,y = make_regression(n_samples=100,n_features=1,noise=10)
plt.scatter(x,y)
plt.show()


y = y.reshape((100,1))
# X = np.hstack([x,np.ones((100,1))])
X = np.hstack((x,np.ones(x.shape)))
print(X.shape)
print(X[:10])

theta =  np.random.randn(2,1)


def model(X,theta) :
    return X.dot(theta)

def cost_function(X,y,theta):
    m = len(y)
    return (1/(2*m)) * np.sum((model(X,theta) - y)**2)

def gradient(X,y,theta):
    m = len(y)
    return (1/m) * X.T.dot(model(X,theta) -y)

def gradient_descent(X,y,theta,learning_rate,iterations):
    
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        theta = theta - learning_rate * gradient(X,y,theta)
        cost_history[i] = cost_function(X,y,theta)

    return theta, cost_history

#evaluation with R-Squared
def r2_score (y,y_pred) :
    u=np.sum((y-y_pred)**2)
    v=np.sum((y-np.mean(y))**2)

    return 1 - (u/v)


theta_final ,cost_history = gradient_descent(X,y,theta,learning_rate=0.014,iterations=1000)

plt.scatter(x,y)
plt.plot(x,model(X,theta_final),c='r')
plt.show()

plt.plot(range(1000),cost_history)
plt.xlabel("Iterations")
plt.ylabel("cost")
plt.show()

y_pred= model(X,theta_final)

print(r2_score(y,y_pred))



#######################################
#       polynomial model
#######################################

#f(x)= ax^2 + bx + c
y= y + abs(y/2)
plt.scatter(x,y)
plt.show()

X2 = np.hstack((x**2,X))
theta = np.random.randn(3, 1) 
theta_final2 ,cost_history2 = gradient_descent(X2,y,theta,learning_rate=0.01,iterations=1000)

plt.scatter(x,y)
plt.plot(x,model(X2,theta_final2),c='r')
plt.show()

plt.plot(range(1000),cost_history2)
plt.xlabel("Iterations")
plt.ylabel("cost")
plt.show()

y_pred2= model(X2,theta_final2)

print(r2_score(y,y_pred2))


#######################################
#       normal expression model
#######################################