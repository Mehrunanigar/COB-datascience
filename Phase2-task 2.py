#!/usr/bin/env python
# coding: utf-8

# # Linear Regression model on train dataset

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# In[3]:


Train_url='https://docs.google.com/spreadsheets/d/e/2PACX-1vRTK2NvcndgPX41Czu6Ft2Ho_nE-z50BgTqdzwFW0rsJ2nvyNLe2DoIg1COzUbgw80oaRBjfy5-WtFk/pubhtml?urp=gmail_link'


# In[4]:


response=requests.get(Train_url)
response.raise_for_status()
soup=BeautifulSoup(response.text,"html.parser")
s=soup.find('table',{'class':'waffle'})
tr=s.find_all('tr')
x=[0,0]
y=[]
for i in tr:
    x=i.find_all('td',{'class':'s1'})
    if len(x)>1:
        y.append({
            'x':int(x[0].text.strip()),
            'y':x[1].text.strip()
        })


# In[5]:


m=pd.DataFrame(y)
m['y']=pd.to_numeric(m['y'])
m.to_csv('Train.csv',index=False)
print("saved to train.csv")


# In[6]:


plt.xlabel('x')
plt.ylabel('y')
plt.scatter(m['x'],m['y'])
plt.show()


# In[7]:


data=pd.read_csv('train.csv')
data.head()


# In[8]:


x_train=data[['x']]
x_train
y_train=data['y']
y_train


# # Test Dataset

# In[10]:


from sklearn import linear_model
model=linear_model.LinearRegression()
model.fit(x_train,y_train)


# In[11]:


Test_url='https://docs.google.com/spreadsheets/d/e/2PACX-1vRyvZ7lknwiSghK9aen1SaTEYoN3JS40rrGLpcyrsVZy1tB2T4gn6Y3-cdzPUFCPMmmqREWefW3kl4_/pubhtml'


# In[12]:


response=requests.get(Test_url)
response.raise_for_status()
soup=BeautifulSoup(response.text,"html.parser")
s=soup.find('table',{'class':'waffle'})
tr=s.find_all('tr')
x=[0,0]
y=[]
for i in tr:
    x=i.find_all('td',{'class':'s1'})
    if len(x)>1:
        y.append({
            'x':int(x[0].text.strip()),
            'y':x[1].text.strip()
        })


# In[14]:


m=pd.DataFrame(y)
m['y']=pd.to_numeric(m['y'])
m.to_csv('Test.csv',index=False)
print("saved to test.csv")


# In[15]:


x_test=data[['x']]
x_test
y_test=data['y']
y_test


# In[16]:


y_pred=model.predict(x_test)
y_pred


# In[17]:


plt.scatter(x_test,y_test,color='blue',label='Output')
plt.plot(x_test,y_pred,color='red',linewidth=2,label='Predicted')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()


# In[18]:


r2=r2_score(y_test,y_pred)
print("R.squarel (r2) Score:",r2)


# In[ ]:




