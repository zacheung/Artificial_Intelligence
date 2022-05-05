"""
@author: Zach
@time:   2022/4/13 下午6:33
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   batch_gradient_descent.py
@Intro:  全量梯度下降
"""
import numpy as np

# 创建数据集
np.random.seed(66)
X = np.random.rand(100, 1)
y = 5 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]

# 创建超参数
n_iterations = 10000
t0, t1 = 5, 500

# 1. 初始化W0,...,Wn，标准正态分页W
theta = np.random.randn(2, 1)
# 4. 判断是否收敛，一般不会去设定阈值，而是直接采用设置相对较大的迭代次数
for i in range(n_iterations):
    # 2. 求梯度，计算gradient
    gradients = X_b.T.dot(X_b.dot(theta) - y)
    # 3. 应用梯度下降法公式去调整theta值，theta_t+1=theta_t-eta*gradient
    learning_rate = t0 / (i + t1)  # 学习率随迭代次数增多而逐渐减小
    theta = theta - learning_rate * gradients

print(theta)
