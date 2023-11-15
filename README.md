# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and Load the dataset.

2. Define X and Y array and Define a function for costFunction,cost and gradient.

3. Define a function to plot the decision boundary.

4. Define a function to predict the Regression value.
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: NAVEEN KUMAR B
RegisterNumber:212222230091


import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
*/
```

## Output:
![image](https://github.com/mrnaviz/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123350791/e2dbec53-b369-4ef3-84ba-35d9bc9fbecb)

## Array value of y:
![image](https://github.com/mrnaviz/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123350791/e04b743e-24a7-498f-87fa-1435d9a5ebe5)

## Score graph:
![image](https://github.com/mrnaviz/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123350791/a395a11e-3d94-4fb9-a818-e17e4bbdd615)

## Sigmoid function graph:
![image](https://github.com/mrnaviz/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123350791/2171f372-a450-4185-84db-8d951e68ee42)

## X train grad value:
![image](https://github.com/mrnaviz/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123350791/2d69c84d-28fc-4c2d-8571-7992121d1e8a)

## Y train grad value:
![image](https://github.com/mrnaviz/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123350791/72e04aff-cdf5-4bdb-872f-eb12459068ff)

## Regression value:
![image](https://github.com/mrnaviz/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123350791/b4286419-915e-42e0-bc46-72311ab1ac6c)

## decision boundary graph:
![image](https://github.com/mrnaviz/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123350791/c6636608-7449-4c6a-90d9-35b7d5b79359)

## Probability value:
![image](https://github.com/mrnaviz/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123350791/7cb5efeb-9fd6-4d07-8707-57febdeff983)

## Prediction value of mean:
![image](https://github.com/mrnaviz/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123350791/4c4d4627-e6ef-4b16-829c-9b4373e1d646)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

