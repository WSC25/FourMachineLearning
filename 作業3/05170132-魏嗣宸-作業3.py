#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import seaborn as sns


# In[61]:


A =pd.read_csv("C:/Users/wei/Documents/大四/機器學導論/Concrete_Data.csv")


# In[63]:


A


# In[67]:


A.rename(columns={'Cement ':'cement'}, inplace=True)
A.rename(columns={'Blast Furnace Slag ':'blast'}, inplace=True)
A.rename(columns={'Fly Ash ':'fly'}, inplace=True)
A.rename(columns={'Water ':'water'}, inplace=True)
A.rename(columns={'Superplasticizer ':'superp'}, inplace=True)
A.rename(columns={'Coarse Aggregate ':'coarse'}, inplace=True)
A.rename(columns={'Fine Aggregate ':'fine'}, inplace=True)
A.rename(columns={'Age ':'age'}, inplace=True)
A.rename(columns={'Concrete compressive strength':'CCS'}, inplace=True)


# In[78]:


A.keys()


# In[79]:


#第3題
cement = pd.DataFrame(A['cement'])
blast = pd.DataFrame(A['blast'])
fly = pd.DataFrame(A['fly'])
water = pd.DataFrame(A['Water  '])
superp = pd.DataFrame(A['superp'])
coarse = pd.DataFrame(A['coarse'])
fine = pd.DataFrame(A['fine'])
age = pd.DataFrame(A['age'])
ccs = pd.DataFrame(A['CCS'])


# In[91]:


cols = ['cement', 'blast', 'fly', 'Water  ', 'superp', 'coarse', 'fine', 'age']

sns.pairplot(A[cols])
plt.tight_layout()
plt.show()


# In[81]:


cols2 = ['cement', 'blast', 'fly', 'Water  ', 'superp', 'coarse', 'fine', 'age','CCS']
cm = np.corrcoef(A[cols2].values.T)


# In[82]:


#第2題
hm = sns.heatmap(cm,
                cbar=True,
                annot=True,
                square=True,
                fmt='.2f',
                annot_kws={'size':10},
                yticklabels=cols,
                xticklabels=cols)

plt.tight_layout()
plt.show()


# In[237]:


#完全不做預處理
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(A[A.columns[:-1]],
                                                    A[[A.columns[-1]]],
                                                    random_state = 8)


# In[238]:


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

slr = LinearRegression()

slr.fit (x_train,y_train)
y_train_pred = slr.predict(x_train)
y_test_pred = slr.predict(x_test)

print('MSE train: %.3f, test:%.3f' %(
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test:%.3f' %(
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# In[86]:


print(slr.coef_)


# In[89]:


A.corr()


# In[403]:


#預處理1
B = A.drop('fine',axis = 1) 
B = B.drop('coarse',axis = 1)
B = B.drop('fly',axis = 1)
B


# In[408]:


x_train, x_test, y_train, y_test = train_test_split(B[B.columns[:-1]],
                                                    B[[B.columns[-1]]],
                                                    random_state = 8)


# In[409]:


slr2 = LinearRegression()

slr2.fit (x_train,y_train)
y_train_pred = slr2.predict(x_train)
y_test_pred = slr2.predict(x_test)

print('MSE train: %.3f, test:%.3f' %(
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test:%.3f' %(
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# In[410]:


print(slr2.coef_)


# In[411]:


B.corr()


# In[338]:


#預處理2
C= A.drop(A[A['fly'] == 0].index)
C= C.drop(C[C['blast'] == 0].index)
C= C.drop(C[C['cement'] == 0].index)
C= C.drop(C[C['Water  '] == 0].index)
C= C.drop(C[C['superp'] == 0].index)
C= C.drop(C[C['coarse'] == 0].index)
C= C.drop(C[C['fine'] == 0].index)
C= C.drop(C[C['age'] == 0].index)


# In[339]:


C


# In[340]:


x_train, x_test, y_train, y_test = train_test_split(C[C.columns[:-1]],
                                                    C[[C.columns[-1]]],
                                                    random_state = 8)


# In[341]:


slr3 = LinearRegression()

slr3.fit (x_train,y_train)
y_train_pred = slr3.predict(x_train)
y_test_pred = slr3.predict(x_test)

print('MSE train: %.3f, test:%.3f' %(
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test:%.3f' %(
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# In[342]:


print(slr3.coef_)


# In[381]:


C.corr()


# In[ ]:




