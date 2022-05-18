"""
@author: Zach
@time:   2022/5/18 下午3:36
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   test_speaker.py
@Intro:  读取并测试高斯人声模型
"""
import os
import pickle
import warnings
import numpy as np
from scipy.io.wavfile import read
from speaker_features import extract_features

warnings.filterwarnings('ignore')
source_path = './data/development_set'
model_path = './data/speaker_models'
test_file = './data/development_set_test.txt'
file_paths = open(test_file, 'r')
gmm_files = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith('.gmm')]
models = [pickle.load(open(f, 'rb')) for f in gmm_files]  # Load the Gaussian gender Models
speakers = [f.split('/')[-1].split('.gmm')[0] for f in gmm_files]  # 人名列表（文件名去掉后缀）

true_count = 0
total_count = 0
for path in file_paths:  # 遍历每一个测试文件
    path = path.strip()
    true_name = path.split('-')[0]  # 测试文件的真实人名
    sr, audio = read(os.path.join(source_path, path).replace('\\', '/'))  # read the audio
    vector = extract_features(audio, sr)  # extract 40 dimensional MFCC & delta MFCC features

    log_likelihood = np.zeros(len(models))  # 每个测试语音每个模型得到一个得分列表
    for i in range(len(models)):  # 遍历每一个gmm模型
        gmm = models[i]
        log_likelihood[i] = gmm.score(vector)

    winner = np.argmax(log_likelihood)  # 选取分数最大的那个值的下标
    print('detected as - ', speakers[winner])
    total_count += 1
    if speakers[winner] == true_name:
        true_count += 1

print('ACC is:', true_count / total_count)
