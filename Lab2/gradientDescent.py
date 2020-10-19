import numpy as np
from logisticRegressionReg import logisticRegressionReg

def gradientDescent(X, Y, alpha, Theta_0, delta, _lambda):
    m = np.size(X, 0)
    J_0, grad = logisticRegressionReg(X, Y, Theta_0, _lambda)
    i = 0
    # Theta_j = Theta_0
    while True:
        Theta_i = Theta_0 - alpha * grad
        J, grad = logisticRegressionReg(X, Y, Theta_i, _lambda)
        if np.abs(J_0 - J) < delta:
            print("梯度下降次数为%d次"%(i))
            break
        else:
            i = i + 1
            if J > J_0:
                alpha = alpha / 2
            J_0 = J
            Theta_0 = Theta_i
    return Theta_0