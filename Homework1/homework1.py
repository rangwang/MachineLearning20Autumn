import g_gradientDescent as g_gradD
import g_normalEquation as gne
import normalEquation as ne
import generateData as gd
from matplotlib import pyplot as plt
import conjugateGradient as cg
import gradientDescent as gradD
import polyFeatures as pf
import numpy as np
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sample_size = 10  # 样本数目
p_lambda = np.exp(-10)  # 超参数$\lambda$
j = 9  # 拟合的多项式曲线次数

plt.figure(figsize=(20, 10))
title = "sample_size = " + str(sample_size) + " degree = " + str(j)

x_origin = np.linspace(0, 1, 100)
y_origin = gd.original_func(x_origin)

# 生成带有噪声的样本点训练集
x_train, y_train = gd.add_noise(gd.original_func, sample_size, 0.25)
x_poly = pf.polyfeatures(x_train, j) # 为训练集样本点增加高次特征

theta = np.zeros(j+1).reshape(j+1, 1)  # 初始化参数为零向量
x_test = pf.polyfeatures(x_origin.reshape(100, 1), j) # 测试集

# 多项式次数的选择比对
RMSTrainlist = list()
RMSTestlist = list()
x_ptest, y_ptest = gd.add_noise(gd.original_func, 100, 0.4)
for i in range(0, 10):
      x_polyi = pf.polyfeatures(x_train, i)
      theta_i = ne.normalEquation(x_polyi, y_train)
      J = np.mean((x_polyi @ theta_i - y_train)**2)
      RMSTrainlist.append(np.sqrt(J))
      x_tpolyi = pf.polyfeatures(x_ptest, i)
      J_t = np.mean((x_tpolyi @ theta_i - y_ptest)**2)
      RMSTestlist.append(np.sqrt(J_t))
plt.subplot(336)
plt.plot(RMSTrainlist, 'o-', c="g", label="Train Set")
plt.plot(RMSTestlist, 'o-', c="r", label="Test Set")
plt.xlabel("degree")
plt.ylabel("E_RMS")
plt.legend()


# 无惩罚项的解析解
plt.subplot(331)
# plt.ylim(-2, 2)
plt.plot(x_origin, y_origin, c="b", label="$\sin(2\pi x)$")
plt.scatter(x_train, y_train, edgecolor="g",
            facecolor="none", s=50, label="trainning data")
nE_theta = ne.normalEquation(x_poly, y_train)
y_test = x_test @ nE_theta
plt.plot(x_origin, y_test, "r", label = "normalEquation")
print("数据集大小为%d，拟合多项式次数为%d时的无惩罚项的解析解系数为" % (sample_size, j))
print(nE_theta)
plt.title(title)
plt.legend()

# 惩罚项系数超参数的选择
RMSpTestlist = list()
RMSpTrainlist = list()
hyperlist = range(-50, 1)
for i in hyperlist:
      x_polyi = pf.polyfeatures(x_train, 9)
      theta_i = gne.g_normalEquation(x_polyi, y_train, np.exp(i))
      J = np.mean((x_polyi @ theta_i - y_train)**2) / 2
      RMSpTrainlist.append(np.sqrt(J))
      x_tpolyi = pf.polyfeatures(x_ptest, 9)
      J_t = np.mean((x_tpolyi @ theta_i - y_ptest)**2) / 2
      RMSpTestlist.append(np.sqrt(J_t))
plt.subplot(337)
plt.plot(hyperlist, RMSpTrainlist, 'o-', c="g", label="Train Set")
plt.plot(hyperlist, RMSpTestlist, 'o-', c="r", label="Test Set")
plt.xlabel("-ln$\lambda$")
plt.ylabel("E_RMS")
plt.legend()

# 有惩罚项的解析解
plt.subplot(332)
# plt.ylim(-2, 2)
plt.plot(x_origin, y_origin, c="b", label="$\sin(2\pi x)$")
plt.scatter(x_train, y_train, edgecolor="g",
            facecolor="none", s=50, label="trainning data")
gnE_theta = gne.g_normalEquation(x_poly, y_train, p_lambda)
y_test = x_test @ gnE_theta
plt.plot(x_origin, y_test, "r", label = "g_normalEquation")
print("数据集大小为%d，拟合多项式次数为%d时，惩罚项系数lambda为%f的有惩罚项的解析解系数为" %
      (sample_size, j, p_lambda))
print(gnE_theta)
plt.legend()

# 无惩罚项的梯度下降法
plt.subplot(333)
# plt.ylim(-2, 1.5)
plt.plot(x_origin, y_origin, c="b", label="$\sin(2\pi x)$")
plt.scatter(x_train, y_train, edgecolor="g",
            facecolor="none", s=50, label="trainning data")
gd_theta = gradD.gradientDescent(x_poly, y_train, 0.1, theta, 1e-7)

y_test = np.dot(x_test, gd_theta)
plt.plot(x_origin, y_test, "r", label = "gradientDescent")
print("数据集大小为%d，拟合多项式次数为%d时的无惩罚项的梯度下降法系数为" % (sample_size, j))
print(gd_theta)
plt.legend()

# 有惩罚项的梯度下降法
plt.subplot(334)
# plt.ylim(-1.5, 1.5)
plt.plot(x_origin, y_origin, c="b", label="$\sin(2\pi x)$")
plt.scatter(x_train, y_train, edgecolor="g",
            facecolor="none", s=50, label="trainning data")
ggd_theta = g_gradD.g_gradientDescent(
    x_poly, y_train, 0.1, theta, 1e-7, p_lambda)
y_test = x_test @ ggd_theta
plt.plot(x_origin, y_test, "r", label = "g_gradientDescent")
print("数据集大小为%d，拟合多项式次数为%d时，惩罚项系数lambda为%f的有惩罚项的梯度下降法系数为" %
      (sample_size, j, p_lambda))
print(ggd_theta)
plt.legend()

# 共轭梯度法
plt.subplot(335)
# plt.ylim(-1.5, 1.5)
plt.plot(x_origin, y_origin, c="b", label="$\sin(2\pi x)$")
plt.scatter(x_train, y_train, edgecolor="g",
            facecolor="none", s=50, label="trainning data")
cg_theta = cg.conjugateGradient(x_poly, y_train, p_lambda, 1e-7) 
y_test = x_test @ cg_theta
plt.plot(x_origin, y_test, "r", label = "conjugateGradient")
print("数据集大小为%d，拟合多项式次数为%d时，惩罚项系数lambda为%f的共轭梯度法系数为" %
      (sample_size, j, p_lambda))
print(cg_theta)
plt.legend()
plt.show()