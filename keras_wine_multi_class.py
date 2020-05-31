#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam


# In[2]:


df = pd.read_csv('wines.csv')
df


# In[3]:


df.info()


# In[4]:


y = df['Class']


# In[5]:


y.value_counts()


# In[6]:


y_cat = pd.get_dummies(y)


# In[7]:


y


# In[ ]:





# In[8]:


df.columns


# In[9]:


X = df.drop('Class' , axis=1)


# In[10]:


X.info()


# In[11]:


model  =  Sequential()


# In[12]:


X.info()


# In[13]:


X.shape


# In[14]:


y_cat.shape


# In[15]:


# spliting the dataset
x_train, x_test, y_train, y_test = train_test_split(X,y_cat,test_size=0.1,random_state=20)


# In[16]:


model.add(Dense(units=64 , input_shape=(13,), 
                activation='relu'))


# In[17]:


model.add(Dense(units=3, activation='softmax'))


# In[18]:


model.summary()


# In[19]:


model.compile(optimizer=Adam(),  
              loss='categorical_crossentropy',
             metrics=['accuracy']
             )


# In[20]:


ep = 20


# In[21]:


model.fit(x_train,y_train, epochs=ep)


# In[23]:


accuracy = model.evaluate(x_test, y_test, verbose=0)
accuracy = accuracy[1]*100
accuracy
 
import os
os.system("sudo touch /root/permdata/accuracy.txt")
os.system("echo {} > /root/permdata/accuracy.txt".format(accuracy))


model.save('/root/permdata/multiclassDL.h5')


# In[ ]:




