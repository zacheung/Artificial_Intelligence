"""
@author: Zach
@time:   2022/5/17 上午11:59
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   test_gender.py
@Intro:  测试高斯模型训练的男女生声音模型
"""
import os
import pickle
import numpy as np
from scipy.io.wavfile import read
from python_speech_features import mfcc
from sklearn import preprocessing
import warnings

warnings.filterwarnings('ignore')


def get_mfcc(sr, audio):
    """对音频作Mel频率倒谱系数解析获取特征并归一化"""
    # winlen分析窗口的长度（以秒为单位），默认值为0.025s（25毫秒）。winstep以秒为单位的连续窗口之间的步长，默认值为0.01秒（10毫秒）
    features = mfcc(audio, sr, winlen=0.025, winstep=0.01, numcep=13, appendEnergy=False)  # (999, 13) numcep要返回的倒谱数，默认为13
    feat = np.asarray(())
    for i in range(features.shape[0]):  # 取每一行遍历
        temp = features[i, :]
        if np.isnan(np.min(temp)):  # np.isnan()是判断是否是空值
            continue
        else:
            if feat.size == 0:
                feat = temp
            else:
                feat = np.vstack((feat, temp))
    features = preprocessing.scale(feat)  # 标准化数据集
    return features


source_path = './test_data/AudioSet/male_clips'  # 男
# source_path = './test_data/AudioSet/female_clips'  # 女
model_path = './'
gmm_files = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith('.GMM')]  # ['./male.GMM', './female.GMM']
models = [pickle.load(open(f, 'rb')) for f in gmm_files]  # 加载模型
genders = [f.split('/')[-1].split('.GMM')[0] for f in gmm_files]  # 获取文件名（性别名）
files = [os.path.join(source_path, f) for f in os.listdir(source_path) if f.endswith('.wav')]  # 获取所有测试文件的全路径

count = 0
for f in files:  # 遍历所有测试文件
    print(f.split('/')[-1])  # 打印文件名
    sr, audio = read(f)
    features = get_mfcc(sr, audio)
    log_likelihood = np.zeros(len(models))  # (2,)
    for i in range(len(models)):
        gmm = models[i]  # checking with each model one by one
        log_likelihood[i] = gmm.score(features)
    winner = np.argmax(log_likelihood)  # 返回log_likelihood中最大值所对应的索引值
    print('detected as -', genders[winner], '\n\tscores: male ', log_likelihood[0], ', female ', log_likelihood[1], '\n')
    if genders[winner] == 'male':
        count += 1

print('ACC is ', count / len(files))  # 准确率
