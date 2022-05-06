#!/usr/bin/env python
# coding: utf-8

# # 读取数据

# In[3]:


import numpy as np
import pandas as pd

data=pd.read_csv('./data/insurance.csv', sep=',')
data.head(n=6)


# # EDA数据探索

# In[4]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(data['charges'])


# In[5]:


plt.hist(np.log(data['charges']))  # np.log()函数对数据作正态分布


# # 特征工程

# In[6]:


data=pd.get_dummies(data)  # get_dummies()函数可以将非数值数据做离散化
data.head()


# In[7]:


x=data.drop('charges', axis=1)
y=data['charges']

x.fillna(0, inplace=True)  # True为直接修改原对象，False为创建一个副本，修改副本，原对象不变（缺省默认）
y.fillna(0, inplace=True)
x.tail()


# In[8]:


y.head()


# In[9]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)  # 将数据切分为训练集和测试集


# In[10]:


from sklearn.preprocessing import StandardScaler

scaler=StandardScaler(with_mean=True, with_std=True)  # 实际上将使用数据的真实μ和σ
scaler.fit(x_train)
x_train_scaler=scaler.transform(x_train)  # 归一化
x_test_scaler=scaler.transform(x_test)
x_train_scaler


# In[11]:


from sklearn.preprocessing import PolynomialFeatures
poly_features=PolynomialFeatures(degree=2, include_bias=False)  # 升维
x_train_scaled=poly_features.fit_transform(x_train_scaler)
x_test_scaled=poly_features.fit_transform(x_test_scaler)


# # 模型训练

# In[25]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

line_reg=LinearRegression()
ridg_reg=Ridge(alpha=10)
grad_reg=GradientBoostingRegressor()


# In[26]:


line_reg.fit(x_train_scaled, np.log1p(y_train))  # np.log1p()是优化后的np.log()，对数据做正态分布
ridg_reg.fit(x_train_scaled, np.log1p(y_train))
grad_reg.fit(x_train_scaled, np.log1p(y_train))


# In[27]:


y_line_predict=line_reg.predict(x_test_scaled)
y_ridg_predict=ridg_reg.predict(x_test_scaled)
y_grad_predict=grad_reg.predict(x_test_scaled)


# # 模型评估

# In[28]:


from sklearn.metrics import mean_squared_error

line_train_error=np.sqrt(mean_squared_error(y_true=np.log1p(y_train), y_pred=line_reg.predict(x_train_scaled)))  # 训练集误差
line_test_error =np.sqrt(mean_squared_error(y_true=np.log1p(y_test), y_pred=y_line_predict))  # 测试集误差

ridge_train_error=np.sqrt(mean_squared_error(y_true=np.log1p(y_train), y_pred=ridg_reg.predict(x_train_scaled)))  # 训练集误差
ridge_test_error =np.sqrt(mean_squared_error(y_true=np.log1p(y_test), y_pred=y_ridg_predict))  # 测试集误差

grad_train_error=np.sqrt(mean_squared_error(y_true=np.log1p(y_train), y_pred=grad_reg.predict(x_train_scaled)))  # 训练集误差
grad_test_error =np.sqrt(mean_squared_error(y_true=np.log1p(y_test), y_pred=y_grad_predict))  # 测试集误差

line_train_error, line_test_error, ridge_train_error, ridge_test_error, grad_train_error, grad_test_error


# In[29]:


line_train_error=np.sqrt(mean_squared_error(y_true=y_train, y_pred=np.exp(line_reg.predict(x_train_scaled))))
line_test_error =np.sqrt(mean_squared_error(y_true=y_test, y_pred=np.exp(y_line_predict)))

ridge_train_error=np.sqrt(mean_squared_error(y_true=y_train, y_pred=np.exp(ridg_reg.predict(x_train_scaled))))
ridge_test_error =np.sqrt(mean_squared_error(y_true=y_test, y_pred=np.exp(y_ridg_predict)))

grad_train_error=np.sqrt(mean_squared_error(y_true=y_train, y_pred=np.exp(grad_reg.predict(x_train_scaled))))
grad_test_error =np.sqrt(mean_squared_error(y_true=y_test, y_pred=np.exp(y_grad_predict)))

line_train_error, line_test_error, ridge_train_error, ridge_test_error, grad_train_error, grad_test_error


# In[ ]:




