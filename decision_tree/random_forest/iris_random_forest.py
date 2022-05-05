"""
@author: Zach
@time:   2022/4/29 下午4:37
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   iris_random_forest.py
@Intro:  用随机森林算法对鸢尾花数据集进行分类
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data[:, :2]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=22)

# n_estimators是森林中树木的数量，n_jobs是要并行运行的作业数，oob_score默认为False
rnd_clf = RandomForestClassifier(n_estimators=15, max_leaf_nodes=16, n_jobs=1, oob_score=True)
rnd_clf.fit(X_train, y_train)
print(rnd_clf.oob_score_)  # oob_score必须为True才有oob_score_

# splitter用于在每个节点上选择拆分的策略；max_samples是从X中抽取样本数来训练每个基本估计量，float类型为百分比；bootstrap有放回地对原始数据集进行均匀抽样
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(splitter='random', max_leaf_nodes=16), n_estimators=15, max_samples=1.0, bootstrap=True, n_jobs=1
)
bag_clf.fit(X_train, y_train)

y_predict_rf = rnd_clf.predict(X_test)
y_predict_bag = bag_clf.predict(X_test)
print(accuracy_score(y_test, y_predict_rf))  # RF模型预测准确率
print(accuracy_score(y_test, y_predict_bag))  # Bagging模型预测准确率

# Feature Importance
rnd_clf2 = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf2.fit(iris['data'], iris['target'])  # (150, 4) (150,)
for name, score in zip(iris['feature_names'], rnd_clf2.feature_importances_):
    print(name, score)
