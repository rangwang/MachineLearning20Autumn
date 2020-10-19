import numpy as np
from matplotlib import pyplot as plt

def generateData(mu, sigma, size, label):
    n = mu.size
    Y = np.ones([size, 1]) * label
    X = np.random.multivariate_normal(mu, sigma, size)
    Data = np.c_[X, Y]
    return Data

# mu = np.array([5,100])
# sigma = np.eye(2)
# X = generateData(mu, sigma, 50, 1)
# print(X)
# plt.scatter(X[:, 0], X[:, 1])
# plt.show()