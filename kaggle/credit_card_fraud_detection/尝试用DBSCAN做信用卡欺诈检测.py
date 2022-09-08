#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import pandas as pd 
import numpy as np 


# In[2]:


os.getcwd()


# ##### 数据来源：https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# In[3]:


# 数据读取
data = pd.read_csv('./data/creditcard.csv')


# In[5]:


data.head(10)


# In[6]:


data.Class.value_counts()


# In[7]:


import matplotlib.pyplot as plt


# In[10]:


num_nonfraud = np.sum(data['Class'] == 0)  # 正常
num_fraud = np.sum(data['Class'] == 1)  # 欺诈


# In[11]:


plt.bar(['Fraud', 'non-fraud'], [num_fraud, num_nonfraud], color='red')
plt.show()


# In[12]:


# 数据集划分 + 简单的特征工程
data['Hour'] = data["Time"].apply(lambda x : divmod(x, 3600)[0])  # divmod 计算a除以b的商和余数，并返回一个包含商和余数的元组
data


# In[16]:


X = data.drop(['Time','Class'],axis=1)
Y = data.Class  # 标签
Y


# In[21]:


# 数据归一化
from sklearn.preprocessing import StandardScaler
sd        = StandardScaler()
column    = X.columns
X[column] = sd.fit_transform(X[column])
X.columns
X


# In[22]:


#训练集交叉验证，开发集测试泛化性能
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.55,random_state=0)


# In[23]:


from sklearn.cluster import DBSCAN


# In[24]:


clustering = DBSCAN(eps=3.8, min_samples=13).fit(X_train)


# In[25]:


pd.Series(clustering.labels_).value_counts()


# In[31]:


X_train['labels_'] = clustering.labels_
X_train['labels']  = Y_train
clustering.labels_


# In[32]:


X_train[X_train['labels']==1]


# In[34]:


X_train[(X_train['labels']==1)&(X_train['labels_']==-1)]


# ##### 可以看到，异常簇有6016个样本，覆盖了186个样本，所以从覆盖的角度来说，还是可以的，但是准确率就比较差了，
# ##### 从这个案例来看，在比较高维度的实际数据中，这个算法并不是很适合用来做异常检测，只有在维度比较低的情况下，做异常检测效果不错。
# ##### https://blog.csdn.net/qq_33431368/article/details/125476567

# In[ ]:




