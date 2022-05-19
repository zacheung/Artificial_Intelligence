"""
@author: Zach
@time:   2022/5/4 下午3:01
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   ensemble_adaboost.py
@Intro:  Adaboost算法
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

# cov协方差矩阵将是该值乘以单位矩阵。该数据集仅产生对称正态分布。n_samples平均分配给不同类的总数。n_features每个样本的特征数。
# mean多维正态分布的平均值。如果没有，则使用原点（0，0，…）。
X1, y1 = make_gaussian_quantiles(cov=2, n_samples=200, n_features=2, n_classes=2, random_state=1)  # (200, 2) (200,)
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5, n_samples=300, n_features=2, n_classes=2, random_state=1)  # (300, 2) (300,)
X = np.concatenate((X1, X2))  # (500, 2)
y = np.concatenate((y1, -y2 + 1))  # (500,)

# Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm='SAMME', n_estimators=200)  # n_estimators最大估计次数
bdt.fit(X, y)

plot_step = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # x是坐标值，-4.950018688068095 7.712026704298365
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # y是坐标值，-4.935097981925363 8.848278359647372
xx, yy = np.meshgrid(np.arange(x_min, y_max, plot_step), np.arange(y_min, y_max, plot_step))  # (690, 690) (690, 690)
# print(np.arange(x_min, y_max, plot_step).shape, np.arange(y_min, y_max, plot_step).shape)  # (690,) (690,)
Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])  # np.c_[(476100,), (476100,)]  (476100, 2)
# print(Z.shape)  # (476100,)
Z = Z.reshape(xx.shape)  # (690, 690)

plot_colors = 'br'
class_names = 'AB'
plt.figure(figsize=(10, 5))
plt.subplot(121)
cs = plt.contour(xx, yy, Z, cmap=plt.cm.Paired)  # contour画等高线图，plt.cm.Paired表示两个相近色彩输出
# print(cs)  # <matplotlib.contour.QuadContourSet object at 0x7fa9403cfb80>

for i, n, c in zip(range(2), class_names, plot_colors):  # Plot the training points
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=c, cmap=plt.cm.Paired, s=20, edgecolors='k', label='Class %s' % n)  # s是数据点的大小

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.axis('tight')  # 坐标轴适应数据大小
plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Boundary')

# Plot the two-class decision scores
two_class_output = bdt.decision_function(X)  # (500,)
plot_range = (two_class_output.min(), two_class_output.max())
plt.subplot(122)
for i, n, c in zip(range(2), class_names, plot_colors):
    # x作直方图所要用的数据，必须是一维数组。bins直方图的柱数，即要分的组数，默认为10。alpha透明度
    # range：元组(tuple)或None；剔除较大和较小的离群值，给出全局范围；如果为None，则默认为(x.min(), x.max())；即x轴的范围
    plt.hist(x=two_class_output[y == i], bins=10, range=plot_range, facecolor=c, label='Class %s' % n, alpha=.5, edgecolor='k')

x1, x2, y1, y2 = plt.axis()  # 返回x轴和y轴的范围
plt.axis((x1, x2, y1, y2 * 1.2))  # 设置坐标轴的范围
plt.legend(loc='upper right')
plt.xlabel('Score')
plt.ylabel('Samples')
plt.title('Decision Scores')

plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.subplots_adjust(wspace=0.35)  # 调整子图布局，wspace、hspace分别表示子图之间左右、上下的间距
plt.show()
