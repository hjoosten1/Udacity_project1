#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

## import data and check what features and data it contains
df = pd.read_csv('housing.csv')
df.dataframeName = 'housing.csv'
df.head()


# In[3]:

## check for outliers, missing data
df.describe()


# In[4]:

## check for null values per column
nan_count = df.isnull().sum()
print(nan_count)

## Interpolate the missing bedroom values
df['total_bedrooms'].fillna(df['total_bedrooms'].mean(), inplace=True)

nan_count = df.isnull().sum()
print(nan_count)


# In[5]:

## calculate the average bedrooms per home and filter out values greater than 500001 as this is an error in the census 
df['avg_pop_household'] = df['population'] / df['households']
df['avg_rooms_household'] = df['total_rooms'] / df['households']
df['avg_bedrooms_household'] = df['total_bedrooms'] / df['households']
df = df[df['median_house_value']<500000]


# In[6]:

## hot encode the categorical values 
def ocean_cat(value):
    if (value == '<1H OCEAN') or (value == 'ISLAND'):
        return 0.5
    elif (value == 'NEAR OCEAN') or (value == 'NEAR BAY'):
        return 1
    else:
        return 0

df['ocean_cat'] = df['ocean_proximity'].apply(ocean_cat)
        


# In[7]:

## create histogram and scatter plots to determine correlations
df.hist(bins=50,figsize=(10,12))
plt.show()


# In[8]:


from pandas.plotting import scatter_matrix
df2 = df.drop(labels=['longitude','latitude','population','total_rooms','total_bedrooms'],axis=1)

scatter_matrix(df2, figsize=(12,10))
plt.show()


# In[9]:


df_no_string = df.drop(['ocean_proximity'], axis=1)
corr_matrix = df_no_string.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

## median income and ocean_proximity are most important features

# In[10]:


import seaborn as sns
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation matrix heatmap')
plt.show()


# In[11]:


df.head()


# In[12]:


X = df[['housing_median_age','median_income','avg_pop_household','population','avg_rooms_household','avg_bedrooms_household','ocean_cat']]
y = df['median_house_value']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=42)


# In[13]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth=3)

rf.fit(X_train,y_train)

y_pred_train = rf.predict(X_train)

y_pred_test = rf.predict(X_test)


# In[16]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
mae_train = mean_absolute_error(y_train, y_pred_train)

mae_test = mean_absolute_error(y_test, y_pred_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test,y_pred_test)


# In[15]:


print(mae_train)
print(mae_test)
## mae train and test are around 5000 

# In[17]:


print(r2_train)
print(r2_test)
## r2 train of 31%
## r2 test of 32% 
## not great but not terrible either

# In[ ]:




