import numpy as np
from matplotlib import pyplot as plt
# from lab2 import addConstant
from gradientDescent import gradientDescent
from predict import predict
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def addConstant(X):
    m = X.shape[0]
    I = np.ones([m, 1])
    X = np.c_[I, X]
    return X

data = np.loadtxt('examScore.txt', delimiter=',')

m = data.shape[0]

X = data[:, :2]
# X_mean = X.mean(axis=0)
# X_range = X.ptp(axis=0)

X = addConstant(X)
Y = data[:, 2:]
Theta_0 = np.zeros([3, 1])

data1 = np.zeros([0, 2])
data2 = np.zeros([0, 2])

# print(X[1, 1:].shape)
for i in range(0, m):
    if data[i, 2] == 0:
        data1 = np.r_[data1, X[i, 1:].reshape(1,2)]
    else:
        data2 = np.r_[data2, X[i, 1:].reshape(1,2)]

# print(data1)

# X_scale = (X - X_mean) / X_range
# Y_scale = Y.copy()
Theta = gradientDescent(X, Y, 0.01, Theta_0, 1e-7, 0)
X_plot = [min(X[:, 1]) - 1, max(X[:, 1]) + 1]
Y_plot = (-Theta[0] - Theta[1] * X_plot) / Theta[2]

p = predict(Theta, X)
print("准确率为：%f"%(np.mean(p == Y) * 100))

plt.scatter(data1[:, 0], data1[:, 1])
plt.scatter(data2[:, 0], data2[:, 1])
plt.plot(X_plot, Y_plot)

plt.show()