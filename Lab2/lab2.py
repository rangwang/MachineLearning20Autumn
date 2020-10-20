import numpy as np
from matplotlib import pyplot as plt
from generateData import generateData
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

# 设定期望向量、协方差矩阵、数据集大小、初始的权重向量
mu_1 = np.array([1, 1])
mu_2 = np.array([3, 3])
sigma = np.eye(2)
sigma_nBayes = sigma.copy()
sigma_nBayes[0, 1] = -1
sigma_nBayes[1, 0] = -1
sigma = sigma_nBayes
train_size = 50
test_size = 100
Theta_0 = np.zeros([3, 1])

# 生成训练集和测试集数据
train_Data1 = generateData(mu_1, sigma, train_size, 1)
train_Data2 = generateData(mu_2, sigma, train_size, 0)
test_Data1 = generateData(mu_1, sigma, test_size, 1)
test_Data2 = generateData(mu_2, sigma, test_size, 0)

# 将训练集数据拆分为X和Y
train_Data = np.r_[train_Data1, train_Data2]
X_train = train_Data[:, :train_Data.shape[1]-1]
X_train = addConstant(X_train)
Y_train = train_Data[:, train_Data.shape[1]-1:]

# 将测试集数据拆分为X和Y
test_Data = np.r_[test_Data1, test_Data2]
X_test = test_Data[:, :train_Data.shape[1]-1]
X_test = addConstant(X_test)
Y_test = test_Data[:, train_Data.shape[1]-1:]



# 计算无正则项条件下的logistic回归并作图
Theta = gradientDescent(X_train, Y_train, 0.1, Theta_0, 1e-8, 0)
p = predict(Theta, X_train)
print("无正则条件下的准确率为：%f"%(np.mean(p == Y_train) * 100))
p_test = predict(Theta, X_test)
print("无正则条件下在测试集的准确率为：%f"%(np.mean(p_test == Y_test) * 100))
X_plot = [min(X_train[:, 1]), max(X_train[:, 1])]
Y_plot = (-Theta[0] - Theta[1]*X_plot) / Theta[2]
X_tplot = [min(X_test[:, 1]), max(X_test[:, 1])]
Y_tplot = (-Theta[0] - Theta[1]*X_tplot) / Theta[2]
plt.subplot(221)
plt.scatter(train_Data1[:, 0], train_Data1[:, 1])
plt.scatter(train_Data2[:, 0], train_Data2[:, 1])
plt.plot(X_plot, Y_plot, label="no reg train set")
plt.legend()
plt.subplot(223)
plt.scatter(test_Data1[:, 0], test_Data1[:, 1])
plt.scatter(test_Data2[:, 0], test_Data2[:, 1])
plt.plot(X_tplot, Y_tplot, label="no reg test set")
plt.legend()

# 计算有正则项条件下的logistic回归并作图
ThetaReg = gradientDescent(X_train, Y_train, 0.1, Theta_0, 1e-8, 1e-4)
pReg = predict(ThetaReg, X_train)
print("有正则条件下的准确率为：%f"%(np.mean(pReg == Y_train) * 100))
pReg_test = predict(ThetaReg, X_test)
print("有正则条件下在测试集的准确率为：%f"%(np.mean(pReg_test == Y_test) * 100))
X_plotReg = [min(X_train[:, 1]), max(X_train[:, 1])]
Y_plotReg = (-ThetaReg[0] - ThetaReg[1]*X_plotReg) / ThetaReg[2]
X_tplotReg = [min(X_test[:, 1]), max(X_test[:, 1])]
Y_tplotReg = (-ThetaReg[0] - ThetaReg[1]*X_tplotReg) / ThetaReg[2]
plt.subplot(222)
plt.scatter(train_Data1[:, 0], train_Data1[:, 1])
plt.scatter(train_Data2[:, 0], train_Data2[:, 1])
plt.plot(X_plotReg, Y_plotReg, label="with reg train set")
plt.legend()
plt.subplot(224)
plt.scatter(test_Data1[:, 0], test_Data1[:, 1])
plt.scatter(test_Data2[:, 0], test_Data2[:, 1])
plt.plot(X_tplotReg, Y_tplotReg, label="with reg test set")
plt.legend()
plt.show()