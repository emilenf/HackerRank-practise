import numpy as np

data_shape = input()
f, n = map(int, data_shape.split())

X = np.array([[float(x) for x in input().split()] for i in range(n)])

y = X[:, -1]
X = X[:, :-1]

def hypothesis(theta, X, n):
    h = np.ones((X.shape[0],1))
    theta = theta.reshape(1,n+1)
    for i in range(0,X.shape[0]):
        h[i] = float(np.matmul(theta, X[i]))
    h = h.reshape(X.shape[0])
    return h

def batch_gradient_descent(theta, alpha, num_iters, h, X, y, n):
    cost = np.ones(num_iters)
    for i in range(0,num_iters):
        theta[0] = theta[0] - (alpha/X.shape[0]) * sum(h - y)
        for j in range(1,n+1):
            theta[j] = theta[j] - (alpha/X.shape[0]) * sum((h-y) * X.transpose()[j])
        h = hypothesis(theta, X, n)
        cost[i] = (1/X.shape[0]) * 0.5 * sum(np.square(h - y))
    theta = theta.reshape(1,n+1)
    return theta, cost

def linear_regression(X, y, alpha, num_iters):
    n = X.shape[1]
    one_column = np.ones((X.shape[0],1))
    X = np.concatenate((one_column, X), axis = 1)
    theta = np.zeros(n+1)
    h = hypothesis(theta, X, n)
    theta, cost = batch_gradient_descent(theta,alpha,num_iters,h,X,y,n)
    return theta, cost

theta, cost = linear_regression(X, y, 0.01, 5000)

t = int(input())
test = np.array([[float(x) for x in input().split()] for i in range(t)])

theta_0 = theta[0][0]
theta_v = theta[0][1:]

predictions = theta_0 + np.matmul(theta_v, test.T)

for pred in predictions:
     print(pred)
