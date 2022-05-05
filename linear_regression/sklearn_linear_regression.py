"""
@author: Zach
@time: 2022/4/12:下午5:43
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   sklearn_linear_regression.py
@Intro:  调用scikit-learn模块求解线性回归模型
"""
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

np.random.seed(15)
X1 = 2 * np.random.rand(100, 1)
X2 = 3 * np.random.rand(100, 1)
X = np.c_[X1, X2]  # 拼接矩阵
y = 4 + 3 * X1 + 2 * X2 + np.random.randn(100, 1)  # 真实值y

# reg = LinearRegression(fit_intercept=False)  # 默认为True，有截距项
reg = LinearRegression()
reg.fit(X, y)
print(reg.intercept_, reg.coef_)  # 输出theta

# 进行预测
X_new = np.array([[0, 0],
                  [2, 1],
                  [2, 4]])
y_predict = reg.predict(X_new)
print(y_predict)

# 绘图
plt.plot(X_new[:, 0], y_predict, 'r-')
plt.plot(X1, y, 'b.')
plt.axis([0, 2, 0, 25])
plt.show()
