"""
@author: Zach
@time:   2022/5/10 下午1:56
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   cluster_spectral.py
@Intro:  谱聚类
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.cluster import spectral_clustering
from sklearn.metrics import euclidean_distances


def expand(a, b):
    d = (b - a) * 0.1
    return a - d, b + d


t = np.arange(0, 2 * np.pi, 0.1)  # (63,)
X1 = np.vstack((np.cos(t), np.sin(t))).T  # (63, 2)
X2 = np.vstack((2 * np.cos(t), 2 * np.sin(t))).T  # (63, 2)
X3 = np.vstack((3 * np.cos(t), 3 * np.sin(t))).T  # (63, 2)
X = np.vstack((X1, X2, X3))  # (189, 2)

n_clusters = 3
m = euclidean_distances(X, squared=True)  # (189, 189) 欧式距离dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12, 8), facecolor='w')
plt.suptitle('谱聚类', fontsize=20)
x1_min, x2_min = np.min(X, axis=0)
x1_max, x2_max = np.max(X, axis=0)
x1_min, x1_max = expand(x1_min, x1_max)
x2_min, x2_max = expand(x2_min, x2_max)
clrs = plt.cm.Spectral(np.linspace(0, 0.8, n_clusters))

# print(np.logspace(-2, 0, 6))  # [0.01       0.02511886 0.06309573 0.15848932 0.39810717 1.        ]
for i, s in enumerate(np.logspace(-2, 0, 6)):  # logspace等⽐数列，默认以10为底，-2到0为指数
    print(i, s)
    af = np.exp(-m ** 2 / (s ** 2)) + 1e-6  # (189, 189)
    # assign_labels用于在嵌入空间中分配标签的策略。k-means可以应用，是一个流行的选择，但它也可能对初始化敏感。discretize离散化是另一种对随机初始化不太敏感的方法。
    y_hat = spectral_clustering(af, n_clusters=n_clusters, assign_labels='kmeans', random_state=99)  # (189,) 当簇是二维平面上的嵌套圆时，谱聚类是非常有用的
    plt.subplot(2, 3, i + 1)
    for k, clr in enumerate(clrs):
        cur = (y_hat == k)
        plt.scatter(X[cur, 0], X[cur, 1], s=40, c=clr.reshape(1, -1), edgecolors='k')  # 对clr升维到二维数组，否则报警告

    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid()
    plt.title('sigma=%.2f' % s, fontsize=16)

plt.tight_layout()
# plt.subplots_adjust(top=0.9)
plt.show()
