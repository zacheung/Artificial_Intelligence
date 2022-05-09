"""
@author: Zach
@time:   2022/5/9 下午10:11
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   cluster_KMeans.py
@Intro:  数据分布对KMeans聚类的影响
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture


def expand(a, b):
    d = (b - a) * 0.1
    return a - d, b + d


N = 400
centers = 4
X, y = ds.make_blobs(N, n_features=2, centers=centers, random_state=99)  # (400, 2) (400,)
X2, y2 = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=(1, 2.5, 0.5, 2), random_state=99)  # (400, 2) (400,)
X3 = np.vstack((X[y == 0][:], X[y == 1][:50], X[y == 2][:20], X[y == 3][:5]))  # (175, 2)
# print(X[y == 0][:].shape, X[y == 1][:50].shape, X[y == 2][:20].shape, X[y == 3][:5].shape)  # (100, 2) (50, 2) (20, 2) (5, 2)
y3 = np.array([0] * 100 + [1] * 50 + [2] * 20 + [3] * 5)  # (175,) 由100个0，50个1，20个2，5个3组成的一维数组
m = np.array(((1, 1), (1, 3)))  # (2, 2)
X_r = X.dot(m)  # 旋转后的数据

cls = KMeans(n_clusters=centers, init='k-means++')
# cls = MiniBatchKMeans(n_clusters=centers)
# cls = GaussianMixture(n_components=centers)  # 高斯混合
y_hat = cls.fit_predict(X)  # (400,) equivalent to calling fit(X) followed by predict(X).
y2_hat = cls.fit_predict(X2)  # (400,)
y3_hat = cls.fit_predict(X3)  # (175,)
y_r_hat = cls.fit_predict(X_r)  # (400,)

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
cm = matplotlib.colors.ListedColormap(list('rgbm'))  # 用于从颜色列表创建colarmap对象
plt.figure(figsize=(9, 10), facecolor='w')

plt.subplot(421)
plt.title('原始数据')
# 表示的是色彩或颜色序列，可选，默认蓝色’b’。但是c不应该是一个单一的RGB数字，也不应该是一个RGBA的序列，因为不便区分。c可以是一个RGB或RGBA二维行数组。
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cm)
x1_min, x2_min = np.min(X, axis=0)
x1_max, x2_max = np.max(X, axis=0)
x1_min, x1_max = expand(x1_min, x1_max)
x2_min, x2_max = expand(x2_min, x2_max)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.grid()

plt.subplot(422)
plt.title('KMeans++聚类')
plt.scatter(X[:, 0], X[:, 1], c=y_hat, s=30, cmap=cm)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.grid()

plt.subplot(423)
plt.title('旋转后的数据')
plt.scatter(X_r[:, 0], X_r[:, 1], c=y, s=30, cmap=cm)
x1_min, x2_min = np.min(X_r, axis=0)
x1_max, x2_max = np.max(X_r, axis=0)
x1_min, x1_max = expand(x1_min, x1_max)
x2_min, x2_max = expand(x2_min, x2_max)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.grid()

plt.subplot(424)
plt.title('旋转后KMeans++聚类')
plt.scatter(X_r[:, 0], X_r[:, 1], c=y_r_hat, s=30, cmap=cm)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.grid()

plt.subplot(425)
plt.title('方差不相等数据')
plt.scatter(X2[:, 0], X2[:, 1], c=y2, s=30, cmap=cm)
x1_min, x2_min = np.min(X2, axis=0)
x1_max, x2_max = np.max(X2, axis=0)
x1_min, x1_max = expand(x1_min, x1_max)
x2_min, x2_max = expand(x2_min, x2_max)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.grid()

plt.subplot(426)
plt.title('方差不相等KMeans++聚类')
plt.scatter(X2[:, 0], X2[:, 1], c=y2_hat, s=30, cmap=cm)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.grid()

plt.subplot(427)
plt.title('数量不相等数据')
plt.scatter(X3[:, 0], X3[:, 1], c=y3, s=30, cmap=cm)
x1_min, x2_min = np.min(X3, axis=0)
x1_max, x2_max = np.max(X3, axis=0)
x1_min, x1_max = expand(x1_min, x1_max)
x2_min, x2_max = expand(x2_min, x2_max)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.grid()

plt.subplot(428)
plt.title('数量不相等KMeans++聚类')
plt.scatter(X3[:, 0], X3[:, 1], c=y3_hat, s=30, cmap=cm)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.grid()

plt.tight_layout(pad=2, rect=(0, 0, 1, 0.97))
plt.suptitle('数据分布对KMeans聚类的影响', fontsize=18)
plt.show()
