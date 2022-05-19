"""
@author: Zach
@time:   2022/5/19 上午9:10
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   img_compress.py
@Intro:  使用KMeans聚类算法进行图像压缩
"""
from skimage import io
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

image = io.imread('./data/tiger.png')
plt.subplot(121)
plt.axis('off')
plt.imshow(image)  # 展示原图

rows, cols, dummy = image.shape  # 720 1280 3
image = image.reshape(rows * cols, 3)  # (921600, 3)
model = KMeans(n_clusters=128, n_init=10, max_iter=200)
model.fit(image)
print('finished KMeans.')

clusters = np.asarray(model.cluster_centers_, dtype=np.uint8)  # (128, 3) model.cluster_centers_簇中心的坐标
# print(model.cluster_centers_.shape)  # (128, 3)
labels = np.asarray(model.labels_, dtype=np.uint8)  # (921600,)
# print(model.labels_.shape)  # (921600,)
labels = labels.reshape(rows, cols)  # (720, 1280)
np.save('./data/codebook_tiger.npy', clusters)
io.imsave('./data/compressed_tiger.png', labels)
print('done.')

plt.subplot(122)
plt.axis('off')
plt.imshow(labels)
plt.tight_layout()
plt.show()
