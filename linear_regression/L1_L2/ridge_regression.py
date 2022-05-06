"""
@author: Zach
@time:   2022/4/14 下午7:49
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   ridge_regression.py
@Intro:  岭回归（添加惩罚项，MSE+L2）
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor

np.random.seed(88)
X = 2 * np.random.rand(100, 1)
y = 5 + 3 * X + np.random.randn(100, 1)

ridge_reg = Ridge(alpha=0.4, solver='sag', max_iter=10000)
ridge_reg.fit(X, y)
print(ridge_reg.predict([[2]]), ridge_reg.intercept_, ridge_reg.coef_)

sgd_reg = SGDRegressor(penalty='l2', max_iter=10000)
sgd_reg.fit(X, y.ravel())  # ravel()方法将维度拉成一维数组
print(sgd_reg.predict([[2]]), sgd_reg.intercept_, sgd_reg.coef_)
