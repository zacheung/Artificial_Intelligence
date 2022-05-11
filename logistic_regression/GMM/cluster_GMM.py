"""
@author: Zach
@time:   2022/5/11 下午8:59
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   cluster_GMM.py
@Intro:  使用GMM作图像聚类
"""
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

im = Image.open('./data/flower.png')
raw_shape = np.array(im).shape
new_data = np.array(im).reshape(-1, 4)[:, :3]  # (800, 820, 4)转换为(656000, 3)

gmm = GaussianMixture(n_components=4, covariance_type='tied')  # n_components混合组成部分的数量。covariance_type协方差参数类型
gmm.fit(new_data)
cluster1 = gmm.predict(new_data)  # (656000,)
cluster1 = cluster1.reshape(raw_shape[0], raw_shape[1])

cluster2 = gmm.predict_proba(new_data)  # (656000, 4)
cluster2 = (cluster2 > 0.98).argmax(axis=1)  # (656000,) 当axis=1，是在行中比较，选出最大的列索引
cluster2 = cluster2.reshape(raw_shape[0], raw_shape[1])

plt.subplot(121)
plt.imshow(cluster1)
plt.subplot(122)
plt.imshow(cluster2)
plt.tight_layout()
plt.show()
