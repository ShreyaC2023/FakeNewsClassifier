#!/usr/bin/env python
# coding: utf-8

# **FAKE NEWS DETECTION**

# In[ ]:


IMPORING LIBRARIES


# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string


# INSERING THE DATASETS

# In[2]:


df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")


# In[4]:


df_fake.tail(10)


# In[5]:


df_true.tail(10)


# CATEGORISING TRUE AND FAKE NEWS

# In[6]:


df_fake["class"] = 0
df_true["class"] = 1


# In[7]:


df_fake.shape, df_true.shape


# In[9]:


df_fake_manual_testing = df_fake.tail(10)
for i in range(23480,23470,-1):
    df_fake.drop([i], axis = 0, inplace = True)
df_true_manual_testing = df_true.tail(10)
for i in range(21416,21406,-1):
    df_true.drop([i], axis = 0, inplace = True)


# In[14]:


print(df_fake.index)
print(df_true.index)


# In[15]:


import pandas as pd

df_fake_manual_testing = df_fake.tail(10).copy()
df_true_manual_testing = df_true.tail(10).copy()


df_fake_manual_testing.loc[:, "class"] = 0
df_true_manual_testing.loc[:, "class"] = 1

# Drop the rows from the original DataFrames 
for i in range(23470, 23460, -1):
    df_fake.drop([i], axis=0, inplace=True)

for i in range(21406, 21396, -1):
    df_true.drop([i], axis=0, inplace=True)


# In[16]:


df_fake_manual_testing.head(10)


# In[17]:


df_true_manual_testing.head(10)


# In[18]:


df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing], axis = 0)
df_manual_testing.to_csv("manual_testing.csv")


# MERGING BOTH TRUE AND FAKE DATAFRAMES

# In[19]:


df_marge = pd.concat([df_fake, df_true], axis =0 )
df_marge.head(10)


# In[21]:


df_marge.columns


# In[22]:


df = df_marge.drop(["title", "subject","date"], axis = 1)  


# In[23]:


df.isnull().sum()


# SHUFFLING THE DATASET

# In[24]:


df = df.sample(frac = 1)


# In[66]:


df.tail()


# In[26]:


df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)


# In[27]:


df.columns


# In[64]:


df.tail()


# TEXT PREPROCESSING FUNCTION FOR LOWERCASING, REMOVING SPECIAL CHARACTERS, SPACES, AND URLS

# In[29]:


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text


# In[34]:


df["text"] = df["text"].apply(wordopt)


# DEFINING DEPENDENT (Y) AND INDEPENDENT (X) VARIABLES

# In[35]:


x = df["text"]
y = df["class"]


# In[36]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


# CONVERTING TEXT TO VECTORS

# In[38]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


# LOGISTIC REGRESSION

# In[39]:


# Importing the LogisticRegression class
from sklearn.linear_model import LogisticRegression

# Creating an instance of LogisticRegression
LR = LogisticRegression()

# Fitting the model to the training data
LR.fit(xv_train, y_train)

# Making predictions
pred_lr = LR.predict(xv_test)

# Calculating the accuracy score
accuracy_lr = LR.score(xv_test, y_test)
print("Accuracy:", accuracy_lr)

# Printing the classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, pred_lr))


# RANDOM FOREST CLASSIFIER

# In[56]:


# Importing the Random Forest Classifier from scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Creating an instance of the Random Forest Classifier with a random_state
RFC = RandomForestClassifier(random_state=0)

# Fitting the model to the training data
RFC.fit(xv_train, y_train)

# Making predictions on the test data
pred_rfc = RFC.predict(xv_test)

# Calculating the accuracy score (optional)
accuracy_rfc = RFC.score(xv_test, y_test)
print("Accuracy:", accuracy_rfc)

# Printing the classification report
print(classification_report(y_test, pred_rfc))


# DESCISION TREE

# In[47]:


# Importing the Decision Tree Classifier from scikit-learn
from sklearn.tree import DecisionTreeClassifier

# Creating an instance of the Decision Tree Classifier
DT = DecisionTreeClassifier()

# Fitting the model to the training data
DT.fit(xv_train, y_train)

# Making predictions on the test data
pred_dt = DT.predict(xv_test)

# Calculating the accuracy score
accuracy_dt = DT.score(xv_test, y_test)
print("Accuracy:", accuracy_dt)

# Printing the classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, pred_dt))


# MANUAL TESTING

# In[61]:


def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_RFC = np.array(RFC.predict(new_xv_test)) 

    return print("\n\nLOGISTIC REGRESSION Prediction: {} \nDESCISION TREE Prediction: {} \nRANDOM FOREST CLASSIFIER Prediction: {}".format(output_lable(pred_LR[0]),
                                                                                          output_lable(pred_DT[0]),
                                                                                          output_lable(pred_RFC[0])))


# In[62]:


news = str(input())
manual_testing(news)


# In[63]:


news = str(input())
manual_testing(news)


# In[ ]:




