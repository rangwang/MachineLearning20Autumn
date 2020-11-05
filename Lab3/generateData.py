import numpy as np

# 生成数据
# label为标签,表示二维点的种类,mu为期望,sigma为方差,size为点的数量
# 最后返回的Data为size * (mu.size + 1)的矩阵
# 前mu.size列为x1,x2,...,xn的值,最后为标签y的值
def generateData(mu, sigma, size, label):
    Y = np.ones([size, 1]) * label
    X = np.random.multivariate_normal(mu, sigma, size)
    Data = np.c_[X, Y]
    return Data