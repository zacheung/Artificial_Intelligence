"""
@author: Zach
@time:   2022/5/17 上午8:55
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   train_models.py
@Intro:  用GaussianMixture训练男女生声音模型
"""
import os
import pickle
import numpy as np
from scipy.io.wavfile import read
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from python_speech_features import mfcc
import warnings

warnings.filterwarnings('ignore')
# source = './train_data/youtube/male'
source = './train_data/youtube/female'
files = [os.path.join(source, f) for f in os.listdir(source) if f.endswith('.wav')]  # os.path.join()将路径和文件名拼接
features = np.asarray(())

for f in files:
    sr, audio = read(f)  # sr为采样率
    print(sr, audio.shape)
    # winlen分析窗口的长度（以秒为单位），默认值为0.025s（25毫秒）。winstep以秒为单位的连续窗口之间的步长，默认值为0.01秒（10毫秒）
    vector = mfcc(audio, sr, winlen=0.025, winstep=0.01, numcep=13, appendEnergy=False)  # numcep要返回的倒谱数，默认为13
    vector = preprocessing.scale(vector)  # 标准化数据集
    if features.size == 0:  # 将vector堆叠到features
        features = vector
    else:
        features = np.vstack((features, vector))

print(features.shape)  # (30169, 13)
gmm = GaussianMixture(n_components=8, max_iter=200, covariance_type='diag', n_init=3)  # n_init要执行的初始化次数。保持最佳结果。
gmm.fit(features)

# model saved as xx.GMM
pickle_file = source.split('/')[-1] + '.GMM'
pickle.dump(gmm, open('./' + pickle_file, 'wb'))
print('modeling completed for gender:', pickle_file)
