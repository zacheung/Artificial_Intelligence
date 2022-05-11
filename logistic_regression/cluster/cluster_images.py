"""
@author: Zach
@time:   2022/5/11 上午11:30
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   cluster_images.py
@Intro:  使用KMeans++作图像聚类
"""
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt


def restore_image(cb, cluster, shape):
    """用预测后的结果重新绘图

    Parameters
        cb: 聚类中心
        cluster: 聚类结果
    """
    row, col, dummy = shape  # 1090 1100 4
    image = np.empty((row, col, 3))  # (1090, 1100, 3)
    index = 0
    for r in range(row):
        for c in range(col):
            image[r, c] = cb[cluster[index]]
            index += 1  # 1090*1100=1199000

    return image


def show_scatter(a):
    """对图像的颜色分布进行画图分析"""
    N = 10
    print('原始数据：\n', a)
    density, edges = np.histogramdd(a, bins=[N, N, N], range=[(0, 1), (0, 1), (0, 1)])  # histogramdd计算某些数据的多维直方图
    print(type(density), type(edges))
    print(density.shape, edges[0].size, edges[1].size, edges[2].size)
    density /= density.max()  # (10, 10, 10)
    x = y = z = np.arange(N)  # [0 1 2 3 4 5 6 7 8 9] [0 1 2 3 4 5 6 7 8 9] [0 1 2 3 4 5 6 7 8 9]
    d = np.meshgrid(x, y, z)  # 转换为3个array，每个列表里有10个二维矩阵

    fig = plt.figure(1, facecolor='w')
    ax = fig.add_subplot(111, projection='3d')
    # depthshade: Whether to shade the scatter markers to give the appearance of depth.（True为半透明，可以看出深度）
    ax.scatter(d[1], d[0], d[2], c='r', s=100 * density, marker='o', depthshade=True)
    ax.set_xlabel('红色分量')
    ax.set_ylabel('绿色分量')
    ax.set_zlabel('蓝色分量')
    plt.title('图像颜色三维频数分布', fontsize=20)

    plt.figure(2, facecolor='w')
    den = density[density > 0]  # (168,)
    # print(density[density <= 0].shape)  # (832,)
    den = np.sort(den)[::-1]  # 先从小到大排序，再从大到小排序
    t = np.arange(len(den))
    plt.plot(t, den, 'r-', t, den, 'go', lw=2)
    plt.title('图像颜色频数分布', fontsize=18)
    plt.grid()
    plt.show()


matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
im = Image.open('./data/Lenna.png')  # <class 'PIL.PngImagePlugin.PngImageFile'>
image = np.array(im).astype(float) / 255  # (1090, 1100, 4)，astype()转换类型
image_v = image.reshape((-1, 4))[:, :3]  # (1199000, 4) 只取RGB
show_scatter(image_v)

N = image_v.shape[0]  # 图像像素总数
idx = np.random.randint(0, N, size=1000)  # (1000,)
image_sample = image_v[idx]  # 随机取image_v里的数据

num_vq = 20
model = KMeans(num_vq)
model.fit(image_sample)
c = model.predict(image_v)  # (1199000,)
print(list(set(c)))
print('聚类结果：\n', c)
print('聚类中心：\n', model.cluster_centers_)

plt.figure(figsize=(15, 8), facecolor='w')
plt.subplot(121)
plt.axis('off')  # 关闭坐标轴
plt.title('原始图片', fontsize=18)
plt.imshow(image)

plt.subplot(122)
vq_image = restore_image(model.cluster_centers_, c, image.shape)
plt.axis('off')
plt.title('矢量量化后图片：%d色' % num_vq, fontsize=20)
plt.imshow(vq_image)

plt.tight_layout()
plt.show()
