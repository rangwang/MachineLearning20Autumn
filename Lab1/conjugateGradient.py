import numpy as np

def conjugateGradient(x_data, y_data, p_lambda, delta):
    A = x_data.T @ x_data + p_lambda * np.identity(np.size(x_data, 1))
    b = x_data.T @ y_data
    i = 0
    theta = np.zeros(np.size(x_data, 1)).reshape(np.size(x_data, 1), 1)
    r = b - A @ theta
    p = r
    while True:
        i = i + 1
        alpha = np.sum(r * p) / np.sum(A @ p * p)
        theta = theta + alpha * p
        r = b - A @ theta
        if(r.T @ r < delta):
            print("共轭梯度下降次数为：%d"%(i))
            break
        beta = -np.sum(r * A @ p) / np.sum(p * A @ p)
        p = r + beta * p
    return theta
