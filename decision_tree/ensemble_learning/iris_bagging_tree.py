"""
@author: Zach
@time:   2022/5/4 下午8:29
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   iris_bagging_tree.py
@Intro:  对比iris数据集在各算法下的分类准确率（OOB）
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()
voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='hard')  # hard使用预测类标签进行多数规则投票

iris = load_iris()
X = iris.data[:, :2]  # 花萼长度和宽度
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=99)

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):  # 比较每种算法的预测准确率
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_predict))

# max_samples从X中提取的样本数，用于训练每个base estimator。bootstrap为True表示可重复抽样，为False表示不放回抽样
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=10, max_samples=1.0, bootstrap=True, n_jobs=1)
bag_clf.fit(X_train, y_train)
y_predict = bag_clf.predict(X_test)
y_predict_prob = bag_clf.predict_proba(X_test)
# print(y_predict, y_predict_prob)
print('bagging', accuracy_score(y_test, y_predict))

# oob，oob_score是否使用袋外样本来评估泛化精度
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, bootstrap=True, n_jobs=1, oob_score=True)
bag_clf.fit(X_train, y_train)
print(bag_clf.oob_score_)  # 只有当“oob_score”为True时，此属性才存在。用OOB获得的数据集得分
y_predict = bag_clf.predict(X_test)
print('oob bagging', accuracy_score(y_test, y_predict))
# print(bag_clf.oob_decision_function_)  # 只有当“oob_score”为True时，此属性才存在
