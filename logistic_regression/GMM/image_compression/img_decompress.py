"""
@author: Zach
@time:   2022/5/19 上午10:08
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   img_decompress.py
@Intro:  解压缩展示图像
"""
from skimage import io
import numpy as np

centers = np.load('./data/codebook_tiger.npy')  # (128, 3)
c_image = io.imread('./data/compressed_tiger.png')  # (720, 1280)

image = np.zeros((c_image.shape[0], c_image.shape[1], 3), dtype=np.uint8)  # (720, 1280, 3) 为图像创建一个容器
for i in range(c_image.shape[0]):
    for j in range(c_image.shape[1]):
        image[i, j, :] = centers[c_image[i, j], :]

io.imsave('./data/reconstructed_tiger.png', image)
io.imshow(image)
io.show()
