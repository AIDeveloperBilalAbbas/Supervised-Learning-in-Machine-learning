#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# In[2]:


# Simulated heart disease dataset (small sample)
data = {
    'age': [29, 45, 64, 50, 39, 62, 42, 36, 55, 48],
    'cholesterol': [210, 250, 180, 240, 190, 230, 220, 200, 260, 195],
    'blood_pressure': [130, 140, 120, 150, 135, 160, 125, 118, 145, 130],
    'has_disease': [0, 1, 1, 1, 0, 1, 1, 0, 1, 0]  # 1 = Yes, 0 = No
}


# In[3]:


# Convert to DataFrame
df = pd.DataFrame(data)


# In[4]:


# Features and target
X = df[['age', 'cholesterol', 'blood_pressure']]
y = df['has_disease']


# In[5]:


# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[6]:


# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# In[7]:


# Predict on test set
y_pred = model.predict(X_test)


# In[8]:


# Results
print("Predictions:", y_pred)
print("Actual:     ", y_test.values)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[ ]:




