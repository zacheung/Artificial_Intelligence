"""
@author: Zach
@time:   2022/4/12:上午10:56
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   analytic_solution.py
@Intro:  解析解求解模型
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12)
# 回归，有监督的机器学习，X，y
X1 = 2 * np.random.rand(100, 1)
X2 = 3 * np.random.rand(100, 1)
y = 6 + 4 * X1 + 3 * X2 + np.random.randn(100, 1)  # y为真实值，误差符合正态分布
# 为了求W0截距项，我们给X矩阵一开始加上一列全为1的X0
X_b = np.c_[np.ones((100, 1)), X1, X2]
# 用解析解公式来求解theta
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print('theta:\n', theta)

# 使用模型去做预测
X_new = np.array([[0, 0],
                  [2, 3]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
print('X_new_b:\n', X_new_b)
y_predict = X_new_b.dot(theta)  # 利用theta对X_new_b进行预测得到y
print('y_predict:\n', y_predict)

# exit(-1)
# 绘图展示真实的数据点和我们预测用的模型
plt.plot(X_new[:, 1], y_predict, 'r-')
# plt.plot(X_new, y_predict, 'r-')
plt.plot(X2, y, 'b.')
plt.axis([0, 3, 0, 25])
plt.show()
