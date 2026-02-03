#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


dataset= pd.read_csv("creditcard.csv")
dataset.head(5)


# In[28]:


plt.figure(figsize=(12,5))
sns.scatterplot(x="Time", y="Class", data=dataset)
plt.show()


# In[32]:


plt.figure(figsize=(12,5))
sns.scatterplot(x="V1", y="Class", data=dataset)
plt.show()


# In[12]:


x = dataset[["Time"]]
y = dataset[["Class"]]


# In[13]:


from sklearn.model_selection import train_test_split


# In[15]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)


# In[16]:


from sklearn.linear_model import LogisticRegression


# In[17]:


lr = LogisticRegression()
lr.fit(x_train,y_train)


# In[18]:


lr.score(x_test, y_test)*100


# In[20]:


lr.predict([[171524]])


# In[25]:


plt.figure(figsize=(12,5))
sns.scatterplot(x="Time", y="Class", data=dataset)
#sns.lineplot(x = "Time", y = lr.predict(x), data=dataset, color="red")
sns.scatterplot(x = "Time", y = lr.predict(x), data=dataset, color="red")
plt.show()


# In[33]:


plt.figure(figsize=(12,5))
sns.scatterplot(x="V1", y="Class", data=dataset)
#sns.lineplot(x = "Time", y = lr.predict(x), data=dataset, color="red")
sns.scatterplot(x = "V1", y = lr.predict(x), data=dataset, color="red")
plt.show()


# In[ ]:




