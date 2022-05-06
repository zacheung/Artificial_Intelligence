#!/usr/bin/env python
# coding: utf-8

# # 读取数据

# In[40]:


import numpy as np
import pandas as pd

data=pd.read_csv('./data/insurance.csv')
data.head()


# # EDA数据探索

# In[41]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(data['charges'])


# In[42]:


plt.hist(np.log1p(data['charges']))


# In[43]:


import seaborn as sns

sns.kdeplot(data.loc[data.sex=='male','charges'], shade=True, label='male')
sns.kdeplot(data.loc[data.sex=='female','charges'], shade=True, label='male')


# In[44]:


sns.kdeplot(data.loc[data.region=='northwest', 'charges'], shade=True, label='northwest')
sns.kdeplot(data.loc[data.region=='northeast', 'charges'], shade=True, label='northeast')
sns.kdeplot(data.loc[data.region=='southwest', 'charges'], shade=True, label='southwest')
sns.kdeplot(data.loc[data.region=='southeast', 'charges'], shade=True, label='southeast')


# In[45]:


sns.kdeplot(data.loc[data.smoker=='yes', 'charges'], shade=True, label='smoker_yes')
sns.kdeplot(data.loc[data.smoker=='no', 'charges'], shade=True, label='smoker_no')


# In[46]:


sns.kdeplot(data.loc[data.children==0, 'charges'], shade=True, label='0')
sns.kdeplot(data.loc[data.children==1, 'charges'], shade=True, label='1')
sns.kdeplot(data.loc[data.children==2, 'charges'], shade=True, label='2')
sns.kdeplot(data.loc[data.children==3, 'charges'], shade=True, label='3')
sns.kdeplot(data.loc[data.children==4, 'charges'], shade=True, label='4')
sns.kdeplot(data.loc[data.children==5, 'charges'], shade=True, label='5')


# In[47]:


sns.kdeplot(data.loc[data.age<30, 'charges'], shade=True, label='20')
sns.kdeplot(data.loc[data.age>=30, 'charges'], shade=True, label='30')


# # 特征工程

# In[48]:


data=data.drop(['region', 'sex'], axis=1)
data.head()


# In[49]:


def greater(df, bmi, num_child):
    df['bmi']=('over' if df['bmi'] >= bmi else 'under')
    df['children']=('no' if df['children']==num_child else 'yes')
    return df

data=data.apply(greater, axis=1, args=(30,0,))  # 已经改变了原data，不能再次执行
data.head()


# In[56]:


data=pd.get_dummies(data)  # get_dummies()函数可以将非数值数据做离散化
data.head()


# In[59]:


x=data.drop('charges', axis=1)
y=data['charges']
x.fillna(0, inplace=True)
y.fillna(0, inplace=True)
x.tail()


# In[60]:


y.head()


# In[61]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3)


# In[69]:


from sklearn.preprocessing import StandardScaler

scaler=StandardScaler(with_mean=True, with_std=True)
scaler.fit(x_train)
x_train_scaled=scaler.transform(x_train)  # 归一化
x_test_scaled=scaler.transform(x_test)


# In[70]:


from sklearn.preprocessing import PolynomialFeatures

poly_features=PolynomialFeatures(degree=2, include_bias=False)  # 升维
x_train_poly=poly_features.fit_transform(x_train_scaled)
x_test_poly=poly_features.fit_transform(x_test_scaled)


# # 模型训练

# In[71]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

line_reg=LinearRegression()
ridg_reg=Ridge(alpha=10)
grad_reg=GradientBoostingRegressor()


# In[72]:


line_reg.fit(x_train_poly, np.log1p(y_train))  # np.log1p()是优化后的np.log()，对数据做正态分布
ridg_reg.fit(x_train_poly, np.log1p(y_train))
grad_reg.fit(x_train_poly, np.log1p(y_train))


# In[73]:


y_line_predict=line_reg.predict(x_test_poly)
y_ridg_predict=ridg_reg.predict(x_test_poly)
y_grad_predict=grad_reg.predict(x_test_poly)


# # 模型评估

# In[74]:


from sklearn.metrics import mean_squared_error

line_train_error=np.sqrt(mean_squared_error(y_true=np.log1p(y_train), y_pred=line_reg.predict(x_train_poly)))  # 训练集误差
line_test_error =np.sqrt(mean_squared_error(y_true=np.log1p(y_test), y_pred=y_line_predict))  # 测试集误差

ridge_train_error=np.sqrt(mean_squared_error(y_true=np.log1p(y_train), y_pred=ridg_reg.predict(x_train_poly)))  # 训练集误差
ridge_test_error =np.sqrt(mean_squared_error(y_true=np.log1p(y_test), y_pred=y_ridg_predict))  # 测试集误差

grad_train_error=np.sqrt(mean_squared_error(y_true=np.log1p(y_train), y_pred=grad_reg.predict(x_train_poly)))  # 训练集误差
grad_test_error =np.sqrt(mean_squared_error(y_true=np.log1p(y_test), y_pred=y_grad_predict))  # 测试集误差

line_train_error, line_test_error, ridge_train_error, ridge_test_error, grad_train_error, grad_test_error


# In[75]:


line_train_error=np.sqrt(mean_squared_error(y_true=y_train, y_pred=np.exp(line_reg.predict(x_train_poly))))
line_test_error =np.sqrt(mean_squared_error(y_true=y_test, y_pred=np.exp(y_line_predict)))

ridge_train_error=np.sqrt(mean_squared_error(y_true=y_train, y_pred=np.exp(ridg_reg.predict(x_train_poly))))
ridge_test_error =np.sqrt(mean_squared_error(y_true=y_test, y_pred=np.exp(y_ridg_predict)))

grad_train_error=np.sqrt(mean_squared_error(y_true=y_train, y_pred=np.exp(grad_reg.predict(x_train_poly))))
grad_test_error =np.sqrt(mean_squared_error(y_true=y_test, y_pred=np.exp(y_grad_predict)))

line_train_error, line_test_error, ridge_train_error, ridge_test_error, grad_train_error, grad_test_error


# In[ ]:




