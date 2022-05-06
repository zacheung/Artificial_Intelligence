"""
@author: Zach
@time:   2022/4/14 下午8:14
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   lasso_regression.py
@Intro:  Lasso回归（添加惩罚项，MSE+L1）
"""
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor

np.random.seed(88)
X = 2 * np.random.rand(100, 1)
y = 5 + 3 * X + np.random.randn(100, 1)

lasso_reg = Lasso(alpha=0.1, max_iter=10000)
lasso_reg.fit(X, y)
print(lasso_reg.predict([[2]]), lasso_reg.intercept_, lasso_reg.coef_)

sgd_reg = SGDRegressor(penalty='l1', max_iter=10000)
sgd_reg.fit(X, y.ravel())
print(sgd_reg.predict([[2]]), sgd_reg.intercept_, sgd_reg.coef_)
