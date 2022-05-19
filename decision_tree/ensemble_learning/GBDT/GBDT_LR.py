"""
@author: Zach
@time:   2022/5/19 上午11:07
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   GBDT_LR.py
@Intro:  GBDT作为编码器，LR作为分类器对鸢尾花数据集作分类
"""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class GradientBoostingWithLR(object):
    def __init__(self):
        self.gbdt_model = None
        self.gbdt_lr_model = None
        self.gbdt_encoder = None
        self.X_train_leafs = None
        self.X_test_leafs = None
        self.X_trans = None

    @staticmethod
    def gbdt_train(X_train, y_train):
        """定义GBDT模型"""
        # n_estimators通过提前停止选择的估计数。verbose启用详细输出。如果为1，则每隔一段时间打印一次进度和性能。
        gbdt_model = GradientBoostingClassifier(n_estimators=10, max_depth=6, verbose=0, max_features=0.5)  # max_features寻找最佳分割时要考虑的特征数
        gbdt_model.fit(X_train, y_train)
        return gbdt_model

    @staticmethod
    def lr_train(X_train, y_train):
        """定义LR模型"""
        lr_model = LogisticRegression()
        lr_model.fit(X_train, y_train)
        return lr_model

    def gbdt_lr_train(self, X_train, y_train):
        """训练GBDT+LR模型"""
        self.gbdt_model = self.gbdt_train(X_train, y_train)
        self.X_train_leafs = self.gbdt_model.apply(X_train)[:, :, 0]  # (90, 10) 使用GBDT的apply方法对原有特征进行编码
        # print(self.gbdt_model.apply(X_train).shape)  # (90, 10, 1)

        # 对特征进行one-hot编码
        self.gbdt_encoder = OneHotEncoder(categories='auto', sparse=False)  # categories每个特征的类别。sparse如果设置为True，将返回稀疏矩阵，否则将返回数组。
        self.X_trans = self.gbdt_encoder.fit_transform(self.X_train_leafs)  # (90, 114)

        self.gbdt_lr_model = self.lr_train(self.X_trans, y_train)  # 采用LR进行训练

    def gbdt_lr_pred(self, model, X_test, y_test):
        """预测及AUC评估"""
        self.X_test_leafs = self.gbdt_model.apply(X_test)[:, :, 0]
        (train_rows, cols) = self.X_train_leafs.shape
        X_trans_all = self.gbdt_encoder.fit_transform(np.concatenate((self.X_train_leafs, self.X_test_leafs), axis=0))  # 全量X数据

        y_pred = model.predict_proba(X_trans_all[train_rows:])[:, 1]  # 取90行之后的，即测试数据
        auc_score = roc_auc_score(y_test, y_pred)
        print('GBDT+LR AUC score: %.5f' % auc_score)
        return auc_score

    @staticmethod
    def model_assessment(model, X_test, y_test, model_name='GBDT'):
        """模型评估"""
        y_pred = model.predict_proba(X_test)[:, 1]  # (60,)
        # print(model.predict_proba(X_test).shape)  # (60, 2) 两列相加等于1
        auc_score = roc_auc_score(y_test, y_pred)
        print('%s AUC score: %.5f' % (model_name, auc_score))
        return auc_score


def load_data():
    iris_data = load_iris()
    X = iris_data['data']
    y = iris_data['target'] == 2  # 将多类数据构造成二分类数据
    return train_test_split(X, y, test_size=0.4, random_state=99)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()  # (90, 4) (60, 4) (90,) (60,)
    gblr = GradientBoostingWithLR()
    gblr.gbdt_lr_train(X_train, y_train)
    gblr.model_assessment(gblr.gbdt_model, X_test, y_test)  # 单独评估GBDT分类器
    gblr.gbdt_lr_pred(gblr.gbdt_lr_model, X_test, y_test)  # 评估GBDT作为编码器+LR作为分类器
