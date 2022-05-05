"""
@author: Zach
@time:   2022/5/5 下午3:14
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   iris_random_forest.py
@Intro:  比较RF和Bagging对iris分类的准确率
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:, :2]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=99)

rnd_clf = RandomForestClassifier(n_estimators=15, max_leaf_nodes=16, n_jobs=1, oob_score=True)  # oob_score是否使用袋外样本来评估泛化精度
rnd_clf.fit(X_train, y_train)
print(rnd_clf.oob_score_)

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(splitter='random', max_leaf_nodes=16), n_estimators=15, max_samples=1.0, bootstrap=True, n_jobs=1
)  # splitter用在每个节点上的拆分策略。bootstrap为True表示可重复抽样，为False表示不放回抽样
bag_clf.fit(X_train, y_train)

y_predict_rf = rnd_clf.predict(X_test)
y_predict_bag = bag_clf.predict(X_test)
print(accuracy_score(y_test, y_predict_rf))
print(accuracy_score(y_test, y_predict_bag))

# Feature Importance
rnd_clf_f = RandomForestClassifier(n_estimators=500, n_jobs=1)
rnd_clf_f.fit(iris.data, iris.target)
for name, score in zip(iris.feature_names, rnd_clf_f.feature_importances_):
    print(name, score)
