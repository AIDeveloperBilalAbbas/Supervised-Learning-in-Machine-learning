#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


# In[2]:


# Step 1: Create dummy data
# Features: Has_COVID, Has_Flu
# Target: Fever (Yes=1, No=0)
data = {
    'Has_COVID': [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
    'Has_Flu':   [1, 1, 0, 0, 1, 1, 0, 0, 1, 0],
    'Fever':     [1, 1, 1, 0, 1, 1, 1, 0, 1, 1]
}


# In[3]:


df = pd.DataFrame(data)


# In[4]:


df


# In[5]:


# Step 2: Prepare features and target
X = df[['Has_COVID', 'Has_Flu']]
y = df['Fever']


# In[6]:


# Step 3: Train the Naive Bayes model
model = GaussianNB()
model.fit(X, y)


# In[13]:


# Step 4: Make predictions on dummy test data
test_data = pd.DataFrame({
    'Has_COVID': [0, 1, 0, 1],
    'Has_Flu':   [0, 0, 1, 1]
})
predictions = model.predict(test_data)
test_data['Predicted_Fever'] = predictions
print(predictions)
print(test_data)


# In[11]:


# Step 5: Visualize the predictions
colors = ['red' if label == 0 else 'green' for label in predictions]
labels = ['No Fever' if label == 0 else 'Fever' for label in predictions]

plt.figure(figsize=(8,5))
plt.scatter(test_data['Has_COVID'], test_data['Has_Flu'], c=colors, s=100, edgecolors='black')

for i, label in enumerate(labels):
    plt.text(test_data['Has_COVID'][i] + 0.05, test_data['Has_Flu'][i], label)

plt.xlabel('Has COVID (1=Yes, 0=No)')
plt.ylabel('Has Flu (1=Yes, 0=No)')
plt.title('Naive Bayes Prediction of Fever')
plt.grid(True)
plt.show()    


# In[ ]:




