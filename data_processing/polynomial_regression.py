"""
@author: Zach
@time:   2022/4/14 下午9:17
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   polynomial_regression.py
@Intro:  多项式升维
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

np.random.seed(88)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 5 + X + 0.5 * X ** 2 + np.random.randn(m, 1)
plt.plot(X, y, 'b.')

X_train = X[:80]
y_train = y[:80]
X_test = X[80:]
y_test = y[80:]

d = {1: 'g-', 2: 'r+', 10: 'y*'}
for i in d:  # 遍历字典的key，即维度
    poly_features = PolynomialFeatures(degree=i, include_bias=True)
    X_poly_train = poly_features.fit_transform(X_train)  # 对训练集升维
    X_poly_test = poly_features.fit_transform(X_test)  # 对测试集升维
    print(poly_features, X_poly_train[0], X_poly_test[0], X_train.shape, X_poly_train.shape)

    lin_reg = LinearRegression(fit_intercept=False)
    lin_reg.fit(X_poly_train, y_train)
    y_train_predict = lin_reg.predict(X_poly_train)  # 传入升维后的数据集
    y_test_predict = lin_reg.predict(X_poly_test)  # 传入升维后的数据集
    print(lin_reg.intercept_, lin_reg.coef_)
    # print(y_train_predict, y_test_predict)

    plt.plot(X_poly_train[:, 1], y_train_predict, d[i])
    print(mean_squared_error(y_train, y_train_predict))
    print(mean_squared_error(y_test, y_test_predict))
    print('-' * 50)

plt.show()
