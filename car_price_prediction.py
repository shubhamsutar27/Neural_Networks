#!/usr/bin/env python
# coding: utf-8

# In[128]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras

import warnings
warnings.filterwarnings('ignore')


# In[129]:


df = pd.read_csv('car_purchasing.csv',encoding='latin1')
df.head()


# In[130]:


# customer name, customer email , country and gender are not useful for predicting car purchase, so I will be dropping those features
df.drop(['customer name','customer e-mail','country','gender'],axis=1,inplace=True)


# In[131]:


df.head()


# In[132]:


df.describe()


# In[133]:


df.info()


# In[134]:


df.isna().sum()


# In[135]:


df.duplicated().sum()


# In[136]:


from matplotlib import cm
color = cm.inferno_r(np.linspace(.4, .8, 30))


# In[137]:


plt.figure(figsize=(10,4))
plt.title("Distribution of age Variable.")
sns.distplot(df['age'],color='#8B1A1A');


# In[138]:


plt.figure(figsize=(10,4))
plt.title("Distribution of annual Salary Variable.")
sns.distplot(df['annual Salary'],color='#A52A2A');


# In[139]:


plt.figure(figsize=(10,4))
plt.title("Distribution of credit card debt	 Variable.")
sns.distplot(df['credit card debt'],color='#A52A2A');


# In[140]:


plt.figure(figsize=(10,4))
plt.title("Distribution of net worth Variable.")
sns.distplot(df['net worth'],color='#A52A2A');


# In[141]:


plt.figure(figsize=(10,4))
plt.title("Distribution of car purchase amount Variable.")
sns.distplot(df['car purchase amount'],color='#A52A2A');


# In[142]:


sns.pairplot(df)


# In[143]:


'''# Normalizing all the numerical features
df_norm = (df-df.min())/ (df.max() - df.min())
df_norm'''


# In[144]:


x = df_norm.iloc[:,0:4]
y = df_norm['car purchase amount']


# In[145]:


from sklearn.preprocessing import MinMaxScaler
MMS = MinMaxScaler()
x_scaled = MMS.fit_transform(x)
y_scaled = MMS.fit_transform(y.values.reshape(-1,1))


# In[ ]:





# In[153]:


# splitting the total data into train and test.
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_scaled,y_scaled,test_size=0.25,random_state=101)


# In[154]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[155]:


import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[156]:


model=Sequential()
model.add(Dense(units=10,input_dim=10,activation ='relu',kernel_initializer='normal'))
model.add(Dense(units=6,activation='tanh',kernel_initializer='normal'))
model.add(Dense(units=1,activation='relu',kernel_initializer='normal'))


# In[157]:


model.compile(optimizer='adam',loss='mean_absolute_error',metrics=['mse'])


# In[158]:


from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import InputLayer,Dense
import tensorflow as tf
#Model Validation
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error


# In[160]:


model.fit(x_train,y_train, epochs=100, batch_size=20)


# In[161]:


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
model = keras.Sequential([
    layers.Dense(50, activation='relu', input_shape=[4]),
    layers.Dense(25, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1,activation='linear')
])
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
)


# In[163]:


history = model.fit(x_train, y_train, batch_size=16,validation_split=0.2,epochs=50)


# In[165]:


loss = model.evaluate(x_test, y_test)
print(loss)


# In[166]:


X_random_sample = np.array([[ 55, 65000, 11600, 562341]])
y_predict = model.predict(X_random_sample)
print('Predicted Purchase Amount is =', y_predict[:,0])

