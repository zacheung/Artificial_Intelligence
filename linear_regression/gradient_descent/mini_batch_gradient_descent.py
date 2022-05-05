"""
@author: Zach
@time:   2022/4/13 下午10:44
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   mini_batch_gradient_descent.py
@Intro:  小批量梯度下降
"""
import numpy as np

np.random.seed(88)
X = 2 * np.random.rand(100, 1)
y = 5 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]

n_epochs = 10000  # 轮次
m = 100  # 小批量的数量
batch_size = 10  # 每批多少条
num_batches = int(m / batch_size)  # 批次
t0, t1 = 5, 500

theta = np.random.randn(2, 1)
for epoch in range(n_epochs):
    arr = np.arange(len(X_b))
    np.random.shuffle(arr)
    X_b = X_b[arr]
    y = y[arr]

    for i in range(num_batches):  # 10个批次，每次10条
        x_batch = X_b[i * batch_size:i * batch_size + batch_size]
        y_batch = y[i * batch_size:i * batch_size + batch_size]
        gradients = x_batch.T.dot(x_batch.dot(theta) - y_batch)
        learning_rate = t0 / (epoch * num_batches + i + t1)
        theta = theta - learning_rate * gradients

print(theta)
