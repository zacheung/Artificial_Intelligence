"""
@author: Zach
@time:   2022/5/7 下午1:59
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   cluster_DBSCAN.py
@Intro:  比较不同参数的DBSCAN聚类效果
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def expand(a, b):
    """增大较大值，减小较小值"""
    d = (b - a) * 0.1
    return a - d, b + d


N = 1000
centers = [[1, 2], [-1, -1], [1, -1], [-1, 1]]
# n_features表示每一个样本有多少特征值。centers是聚类中心点的个数，可以理解为label的种类数。
# cluster_std设置每个类别的方差。random_state是随机种子，可以固定生成的数据。
X, y = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=[0.5, 0.25, 0.7, 0.5], random_state=99)  # (1000, 2) (1000,)
X = StandardScaler().fit_transform(X)  # (1000, 2) 标准归一化

# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False
x1_min, x2_min = np.min(X, axis=0)
x1_max, x2_max = np.max(X, axis=0)
x1_min, X1_max = expand(x1_min, x1_max)  # 使最小值变得更小，最大值变得更大
x2_min, x2_max = expand(x2_min, x2_max)
plt.figure(figsize=(12, 8), facecolor='w')
plt.suptitle('DBSCAN聚类', fontsize=20)

params = ((0.2, 5), (0.2, 10), (0.2, 15), (0.3, 5), (0.3, 10), (0.3, 15))
for i in range(len(params)):  # 比较不同参数的DBSCAN聚类效果
    eps, min_samples = params[i]
    model = DBSCAN(eps=eps, min_samples=min_samples)  # eps两个样本之间的最大距离。min_samples将某个点视为核心点的邻域中的样本数
    model.fit(X)
    y_hat = model.labels_  # (1000,)
    # print(y_hat)

    core_indices = np.zeros_like(y_hat, dtype=bool)  # (1000,) dtype覆盖结果的数据类型，将数值转换为bool
    core_indices[model.core_sample_indices_] = True  # 核心点的索引，因为labels_不能区分核心点还是边界点，所以需要用这个索引确定核心点。

    y_unique = np.unique(y_hat)  # 该函数是去除数组中的重复数字，并进行排序之后输出。
    n_clusters = y_unique.size - (1 if -1 in y_hat else 0)  # y_unique.size就是长度
    print(y_unique, '聚类簇的个数为：', n_clusters)

    plt.subplot(2, 3, i + 1)
    clrs = plt.cm.Spectral(np.linspace(0, 0.8, y_unique.size))  # 实现的功能是给不同的label不同的颜色
    # print(clrs)
    for k, clr in zip(y_unique, clrs):  # 类别与颜色
        cur = (y_hat == k)  # (1000,) 与y_hat一一比较，相等则为True，否则为False
        if k == -1:
            plt.scatter(X[cur, 0], X[cur, 1], s=20, c='k')  # 从X中取cur是True的行第0列作为横坐标轴，取cur是True的行第1列作为纵坐标轴
        else:
            plt.scatter(X[cur, 0], X[cur, 1], s=30, c=clr, edgecolors='k')
            # cur & core_indices都为True则为True，否则为False
            plt.scatter(X[cur & core_indices][:, 0], X[cur & core_indices][:, 1], s=60, c=clr, edgecolors='k')

    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid()
    plt.title('epsilon=%.1f m=%d 聚类数目：%d' % (eps, min_samples, n_clusters), fontsize=16)

plt.tight_layout()
# plt.subplots_adjust(top=0.9)
plt.show()
