import numpy as np 
import costFunction as cf

def gradientDescent(x_data, y_data, alpha, theta, delta):
    m = np.size(x_data, 0) #数据的组数
    J_0, grad = cf.costFunction(x_data, y_data, theta) #根据参数计算梯度
    i = 0
    while True:
        thetat = theta - alpha * grad #进行梯度下降
        J, grad = cf.costFunction(x_data, y_data, thetat)
        if np.abs(J_0 - J) < delta:
            print("梯度下降次数为：%d"%(i))
            break
        else:
            i = i + 1
            if J > J_0:
                alpha = alpha / 2
            J_0 = J
            theta = thetat
    return theta