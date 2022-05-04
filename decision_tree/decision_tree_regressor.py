"""
@author: Zach
@time:   2022/4/28 下午1:53
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   decision_tree_regressor.py
@Intro:  不同深度回归树对SineWave的拟合程度
"""
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

N = 100
x = np.random.rand(N) * 6 - 3  # 均匀分布，一维数组
x.sort()
y = np.sin(x) + np.random.rand(N) * 0.05
x = x.reshape(-1, 1)  # 二维
x_test = np.linspace(-3, 3, 50).reshape(-1, 1)

dt_reg = DecisionTreeRegressor(criterion='mse', max_depth=3)
dt_reg.fit(x, y)
y_test = dt_reg.predict(x_test)

fig1 = plt.figure()
plt.plot(x, y, 'y*', label='actual')
plt.plot(x_test, y_test, 'b-', linewidth=2, label='predict')
plt.legend(loc='upper left')

# 比较不同深度的决策树
fig2 = plt.figure()
depth = [2, 4, 6, 8, 10]
color = 'rgbmy'
dt_reg_d = DecisionTreeRegressor()
plt.plot(x, y, 'ko', label='actual')
for d, c in zip(depth, color):
    dt_reg_d.set_params(max_depth=d)
    dt_reg_d.fit(x, y)
    y_test_d = dt_reg_d.predict(x_test)
    plt.plot(x_test, y_test_d, '-', color=c, label='depth=%d' % d)
plt.legend()
plt.grid(b=True)  # b是否显示网格线，默认是True
plt.show()
# plt.savefig('./compare_decision_tree_depth')  # 保存到本地
