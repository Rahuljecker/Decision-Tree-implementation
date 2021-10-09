#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings(action="ignore")

import DATASET
            
# In[3]:


Data=pd.read_csv("churn_prediction_simple.csv")


# In[4]:


Data.head()


# In[6]:


x=Data.drop(columns=["churn","customer_id"])
y=Data["churn"]


# In[9]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_x=scaler.fit_transform(x)


# In[13]:


from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(scaled_x, y, train_size = 0.80, stratify = y)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[10]:


from sklearn.tree import DecisionTreeClassifier as DTC
classifier=DTC(class_weight="balance")
classifier=DTC()


# In[14]:


classifier.fit(x_train,y_train)
predicted_values=classifier.predict(x_train)


# In[16]:


from sklearn.metrics import classification_report
print(classification_report(y_train,predicted_values))


# In[21]:


predicted_values = classifier.predict(x_test)
print(classification_report(y_test, predicted_values))

VISUALISING DECISION TREE
# In[22]:


get_ipython().system('pip install graphviz')


# In[25]:


from sklearn.tree import export_graphviz
export_graphviz(decision_tree=classifier,out_file="tree_viz",
               max_depth=None,feature_names=x.columns,label=None,impurity=False)

