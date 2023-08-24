#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# In[2]:


data = pd.read_csv('TitanicData.csv')


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.describe()


# In[6]:


data.info()


# In[7]:


data.shape


# In[8]:


data.isnull().sum()


# In[9]:


# Preprocessing
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Fare'].fillna(data['Fare'].mean(),inplace=True)
data.drop(['Cabin','Name','Ticket'], axis=1, inplace=True)


# In[10]:


data['Age'] = data['Age'].astype(int)
data['Fare'] = data['Fare'].astype(int)


# In[11]:


data


# In[12]:


data.info()


# In[13]:


data['Sex'] = LabelEncoder().fit_transform(data['Sex'])
data['Embarked'] = LabelEncoder().fit_transform(data['Embarked'])


# In[14]:


X = data.drop(['Survived'], axis=1)
y = data['Survived']


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[16]:


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[17]:


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}%")


# In[ ]:




