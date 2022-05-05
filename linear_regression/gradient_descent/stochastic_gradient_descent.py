"""
@author: Zach
@time:   2022/4/13 下午10:12
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   stochastic_gradient_descent.py
@Intro:  随机梯度下降
"""
import numpy as np

X = np.random.rand(100, 1)
y = 5 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]

n_epochs = 10000
m = 100
t0, t1 = 5, 500

theta = np.random.randn(2, 1)
for epoch in range(n_epochs):
    # 每个轮次开始时打乱数据索引顺序
    arr = np.arange(len(X_b))
    np.random.shuffle(arr)
    X_b = X_b[arr]
    y = y[arr]

    for i in range(m):
        xi = X_b[i:i + 1]  # 这样写是二维，X_b[i]是一维
        yi = y[i:i + 1]  # 这样写是二维，y[i]是一维
        gradients = xi.T.dot(xi.dot(theta) - yi)
        learning_rate = t0 / (epoch * m + i + t1)  # 学习率随迭代次数增多而逐渐减小
        theta = theta - learning_rate * gradients

print(theta)
