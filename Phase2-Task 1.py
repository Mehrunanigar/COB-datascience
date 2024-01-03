#!/usr/bin/env python
# coding: utf-8

# # creating graphs using seaborn and matplotlib.Dataset 

# In[8]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[9]:


x=pd.read_csv("netflix1.csv")
x


# In[26]:


x['date_added'] = pd.to_datetime(x['date_added'], format='%m/%d/%Y')


# In[27]:


x['date_added'] = pd.to_datetime(x['date_added'], format='%d-%m-%Y')


# In[28]:


x['date_added']=x['date_added'].dt.strftime('%d-%m-%Y')
x


# In[29]:


x.type.value_counts()


# In[30]:


sns.countplot(x='type',data=x)
plt.title("count VS Type of Shows")


# In[31]:


x['country'].value_counts().head(10)


# In[32]:


plt.figure(figsize=(12,6))
sns.countplot(y='country',order=df['country'].value_counts().index[0:10],data=x)
plt.title('country wise content on NetFlix')


# In[20]:


movie_countries=df[df['type']=='Movie']
tv_show_countries=df[df['type']=='Tv Show']
plt.figure(figsize=(12,6))
sns.countplot(y='country',order=df['country'].value_counts().index[0:20],data=movie_countries)
plt.title('Top 10 countries producing movies on Netflix')


# In[33]:


x.rating.value_counts()


# In[34]:


plt.figure(figsize=(10,6))
sns.countplot(x='rating',order=x['rating'].value_counts().index[0:10],data=x)
plt.title('Rating of Shows on Netfix VS Count')


# In[35]:


x.release_year.value_counts()[:20]


# In[36]:


plt.figure(figsize=(9,6))
sns.countplot(x='release_year',order=df['release_year'].value_counts().index[0:20],data=x)
plt.title('Content Release in years on Netfix VS Count')


# In[26]:


plt.figure(figsize=(12,8))
sns.countplot(y='listed_in',order=df['listed_in'].value_counts().index[0:20],data=df)
plt.title('Top 20 Genre on Netfix')


# In[ ]:




