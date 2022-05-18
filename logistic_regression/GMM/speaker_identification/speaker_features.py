"""
@author: Zach
@time:   2022/5/18 下午1:46
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   speaker_features.py
@Intro:  extract 40 dimensional MFCC & delta MFCC features
"""
import numpy as np
from sklearn import preprocessing
from python_speech_features import mfcc


def calculate_delta(array):
    """Calculate and returns the delta of given feature vector matrix. 计算并返回给定特征向量矩阵的增量"""
    rows, cols = array.shape
    deltas = np.zeros((rows, 20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i - j < 0:
                first = 0
            else:
                first = i - j
            if i + j > rows - 1:
                second = rows - 1
            else:
                second = i + j
            index.append((second, first))
            j += 1
        deltas[i] = (array[index[0][0]] - array[index[0][1]] + (2 * (array[index[1][0]] - array[index[1][1]]))) / 10
        # print(deltas.shape)
    return deltas


def extract_features(audio, rate):
    """
        extract 20 dim mfcc features from an audio, performs CMS and combines delta to make it 40 dim feature vector.
        从音频中提取20个dim mfcc特征，执行CMS并将delta合并为40个dim特征向量
    """
    mfcc_feat = mfcc(audio, rate, 0.025, 0.01, 20, appendEnergy=True)
    # print(mfcc_feat.shape)
    mfcc_feat = preprocessing.scale(mfcc_feat)
    delta = calculate_delta(mfcc_feat)
    combined = np.hstack((mfcc_feat, delta))  # 将参数元组的元素数组按水平方向进行叠加
    # print(combined.shape)
    return combined
