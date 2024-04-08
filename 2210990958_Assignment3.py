#!/usr/bin/env python
# coding: utf-8

# ### Heart Disease Prediction Using Logistic Regression

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("C:\\Users\\saksh\\Downloads\\framingham.csv")
df


# ### Analysis of Data

# In[4]:


df.shape


# In[5]:


df.keys()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.isna().sum()


# ### Removing NaN / NULL vales from the data

# In[9]:


df
df.dropna(axis = 0, inplace = True) 
print(df.shape)


# In[10]:


df['TenYearCHD'].value_counts()


# ### Data Visualization with Correlation Matrix

# In[12]:


plt.figure(figsize = (14, 10)) 
sns.heatmap(df.corr(), cmap='Purples',annot=True, linecolor='Green', linewidths=1.0)
plt.show()


# ### Pairplot

# In[14]:


sns.pairplot(df)
plt.show()


# ##### Countplot of people based on their sex and whether they are Current Smoker or not

# In[15]:


sns.catplot(data=df, kind='count', x='male',hue='currentSmoker')
plt.show()


# #### Countplot - subplots of No. of people affecting with CHD on basis of their sex and current smoking.

# In[17]:


sns.catplot(data=df, kind='count', x='TenYearCHD', col='male',row='currentSmoker', palette='Blues')
plt.show()


# ### Machine Learning Part

# ### Separating the data into feature and target data.

# In[18]:


X = df.iloc[:,0:15]
y = df.iloc[:,15:16]


# In[19]:


X.head()


# In[20]:


y.head()


# ### Importing the model and assigning the data for training and test set

# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=21)


# ### Applying the ML model - Logistic Regression

# In[22]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# ### Training the data

# In[23]:


logreg.fit(X_train, y_train)


# ### Testing the data

# In[24]:


y_pred = logreg.predict(X_test)


# ### Predicting the score

# In[25]:


score = logreg.score(X_test, y_test)
print("Prediction score is:",score)


# ### Getting the Confusion Matrix and Classification Report

# ## CONFUSION MATRIX

# In[26]:


from sklearn.metrics import confusion_matrix, classification_report 
cm = confusion_matrix(y_test, y_pred) 
print("Confusion Matrix is:\n",cm)


# ### Classification Report

# In[28]:


print("Classification Report is:\n\n",classification_report(y_test,y_pred))


# # Plotting the confusion matrix

# In[30]:


conf_matrix = pd.DataFrame(data = cm,  
                           columns = ['Predicted:0', 'Predicted:1'],  
                           index =['Actual:0', 'Actual:1']) 
plt.figure(figsize = (10, 6)) 
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = "Greens", linecolor="Blue", linewidths=1.5) 
plt.show() 

