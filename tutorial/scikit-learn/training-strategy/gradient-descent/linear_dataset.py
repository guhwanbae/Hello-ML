import numpy as np
import matplotlib.pyplot as plt

def random_linear_dataset(slope=3, intercept=4, n=100):
    w = np.array([intercept, slope]).reshape((2, 1))
    x0 = np.ones((n, 1))
    x1 = np.random.rand(n, 1) * 2
    X = np.concatenate((x0, x1), axis=1)
    y = np.dot(X, w) + np.random.randn(n, 1)
    return X, y

def plot_dataset(X, y):
    plt.scatter(X[:,1], y, c='r', marker='o')

def plot_line(theta):
    intercept, slope = theta[0], theta[1]
    plt.plot([0, 2], [intercept, 2*slope+intercept], color='b')
