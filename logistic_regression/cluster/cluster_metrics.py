"""
@author: Zach
@time:   2022/5/10 上午9:02
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   cluster_metrics.py
@Intro:  聚类的同一性和完整性对比，兰德系数计算
"""
from sklearn import metrics


def metrics_score(y_true, y_predict):
    h = metrics.homogeneity_score(y_true, y_predict)  # 如果所有聚类只包含属于单个类的数据点，则聚类结果满足同质性。
    c = metrics.completeness_score(y_true, y_predict)  # 如果属于给定类的所有数据点都是同一个簇的元素，则聚类结果满足完备性。
    v = metrics.v_measure_score(y_true, y_predict)  # 前两个的调和平均值
    v2 = 2.0 * c * h / (c + h)
    print('同一性(Homogeneity)：', h)
    print('完整性(Completeness)：', c)
    print('V-Measure:', v, v2)
    print('-' * 100)


y = [0, 0, 0, 1, 1, 1]
y_hat1 = [0, 0, 1, 1, 2, 2]
y_hat2 = [0, 0, 1, 3, 3, 3]
y_hat3 = [1, 1, 1, 0, 0, 0]

metrics_score(y, y_hat1)
metrics_score(y, y_hat2)
metrics_score(y, y_hat3)  # 允许不同值

y = [0, 0, 1, 1]
y_hat = [0, 1, 0, 1]
ari = metrics.adjusted_rand_score(y, y_hat)  # 兰德系数，取值在［－1，1］之间，负数代表结果不好，值越大意味着聚类结果与真实情况越吻合
print(ari)

y = [0, 0, 0, 1, 1, 1]
y_hat = [0, 0, 1, 1, 2, 2]
ari = metrics.adjusted_rand_score(y, y_hat)
print(ari)
