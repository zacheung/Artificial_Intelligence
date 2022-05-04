"""
@author: Zach
@time:   2022/4/14 下午1:23
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   standard_normalization.py
@Intro:  标准归一化
"""
from sklearn.preprocessing import StandardScaler
import numpy as np

temp = np.array([1, 2, 3, 5, 50001])
temp = temp.reshape(-1, 1)  # 转成1列
scaler = StandardScaler()
scaler.fit(temp)
print(scaler, scaler.mean_, scaler.var_)  # 输出平均值和方差

print(scaler.transform(temp))  # 输出标准归一化后的矩阵
