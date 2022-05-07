"""
@author: Zach
@time:   2022/4/14 下午8:25
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   elasticnet_regression.py
@Intro:  既有L1，又有L2
"""
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor

np.random.seed(88)
X = 2 * np.random.rand(100, 1)
y = 5 + 3 * X + np.random.randn(100, 1)

elastic_reg = ElasticNet(alpha=0.04, l1_ratio=0.05, max_iter=10000)
elastic_reg.fit(X, y)
print(elastic_reg.predict([[2]]), elastic_reg.intercept_, elastic_reg.coef_)

sgd_reg = SGDRegressor(penalty='elasticnet', max_iter=10000)
sgd_reg.fit(X, y.ravel())
print(sgd_reg.predict([[2]]), sgd_reg.intercept_, sgd_reg.coef_)
