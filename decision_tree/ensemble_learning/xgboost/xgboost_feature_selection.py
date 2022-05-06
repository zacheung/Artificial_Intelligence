"""
@author: Zach
@time:   2022/5/6 下午12:52
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   xgboost_feature_selection.py
@Intro:  用XGBoost对皮马印第安人糖尿病数据集进行特征选择
"""
from numpy import loadtxt
from numpy import sort
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

data_set = loadtxt('./pima-indians-diabetes.csv', delimiter=',')
X = data_set[:, 0:8]  # (768, 8)
y = data_set[:, 8]  # (768,)

model = XGBClassifier(use_label_encoder=False)
model.fit(X, y)
print(model.feature_importances_)
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)  # bar画柱状图
# plot feature importance
plot_importance(model)  # 基于拟合树画feature重要性
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=99)  # <class 'numpy.ndarray'>
xgb = XGBClassifier(use_label_encoder=False)
xgb.fit(X_train, y_train)
y_predict = xgb.predict(X_test)  # <class 'numpy.ndarray'>
# predictions = [round(value) for value in y_predict]  # <class 'list'>
accuracy = accuracy_score(y_test, y_predict)
print('Accuracy: %.2f%%' % (accuracy * 100))
print('-' * 100)

# Fit model using each importance as a threshold
thresholds = sort(model.feature_importances_)  # 将特征重要性进行排序
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)  # threshold用于特征选择的阈值。prefit模型是否直接传递给构造函数
    select_X_train = selection.transform(X_train)
    # print(select_X_train.shape)

    # train model
    selection_model = XGBClassifier(use_label_encoder=False)
    selection_model.fit(select_X_train, y_train)

    # eval model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Thresh=%.3f, n=%d, Accuracy:%.2f%%' % (thresh, select_X_test.shape[1], accuracy * 100))
