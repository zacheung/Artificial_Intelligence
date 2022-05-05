"""
@author: Zach
@time:   2022/4/28 下午9:02
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   iris_decision_tree.py
@Intro:  1.分类树对鸢尾花数据集分类；2.比较不同深度分类树的错误率
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib as mpl

iris = load_iris()
data = pd.DataFrame(iris.data)  # (150, 4)
data.columns = iris.feature_names  # 添加列名
data['Species'] = iris.target  # 添加一列（y）
# print(data)

X = data.iloc[:, 2:4]  # 取花瓣的长度和宽度两列
y = data.iloc[:, -1]  # 取最后一列y
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=22)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)  # (112, 2) (38, 2) (112,) (38,)

# tree_clf = DecisionTreeClassifier(max_depth=4, criterion='gini')
tree_clf = DecisionTreeClassifier(max_depth=3, criterion='entropy')
tree_clf.fit(X_train, y_train)
y_test_hat = tree_clf.predict(X_test)
print('acc score:', accuracy_score(y_test, y_test_hat))  # 模型准确率
print(tree_clf.feature_importances_)  # 查看数据的两个维度的重要程度

export_graphviz(
    tree_clf,
    out_file='./iris_decision_tree.dot',
    feature_names=iris.feature_names[2:4],
    class_names=iris.target_names,  # ['setosa' 'versicolor' 'virginica'] 鸢尾花种类名称
    rounded=True,  # 圆角矩形
    filled=True  # 填充颜色
)
# dot -Tpng iris_decision_tree.dot -o iris_decision_tree.png

print(tree_clf.predict_proba([[5, 1.5]]))
print(tree_clf.predict([[5, 1.5]]))

# 比较不同深度分类树的错误率
depth = np.arange(1, 15)
err_list = []
for d in depth:
    clf_d = DecisionTreeClassifier(max_depth=d, criterion='gini')
    # clf_d = DecisionTreeRegressor(max_depth=d, criterion='mse')
    clf_d.fit(X_train, y_train)
    y_test_d_hat = clf_d.predict(X_test)  # (38,)
    result = (y_test_d_hat == y_test)  # (38,)
    err = 1 - np.mean(result)  # 错误率
    err_list.append(err)
    print(d, '错误率：%.2f%%' % (err * 100))

# mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(facecolor='y')
plt.plot(depth, err_list, 'ro-', lw=2)
plt.xlabel('决策树深度', fontsize=15)
plt.ylabel('错误率', fontsize=15)
plt.title('决策树深度和拟合程度', fontsize=18)
plt.grid()
plt.show()
