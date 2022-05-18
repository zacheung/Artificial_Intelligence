"""
@author: Zach
@time:   2022/5/18 下午1:38
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   train_models.py
@Intro:  训练人声模型并存储
"""
import os
import pickle
import warnings
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture
from speaker_features import extract_features

warnings.filterwarnings('ignore')
source = './data/development_set'  # path to training data
dest = './data/speaker_models'  # path where training speakers will be saved
train_file = './data/development_set_enroll.txt'
file_paths = open(train_file, 'r')

count = 1
features = np.asarray(())
for path in file_paths:
    path = path.strip()
    # print(path)
    sr, audio = read(os.path.join(source, path).replace('\\', '/'))  # read the audio
    vector = extract_features(audio, sr)  # extract 40 dimensional MFCC & delta MFCC features

    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))

    # when features of 5 files of speaker are concatenated, then do model training. 将说话人的5个文件的特征串联起来，然后进行模型训练
    if count == 5:
        gmm = GaussianMixture(n_components=16, max_iter=200, covariance_type='diag', n_init=3)
        gmm.fit(features)

        # dumping the trained gaussian model
        pickle_file = path.split('-')[0] + '.gmm'
        pickle.dump(gmm, open(os.path.join(dest, pickle_file), 'wb'))
        print('+ modeling completed for speaker:', pickle_file, ' with data point = ', features.shape)
        features = np.asarray(())
        count = 0
    count += 1
