#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[32]:


data = {
    "Hours" : [1,2,4,5,6],
    "Marks" : [50,60,80,90,100]
}
df = pd.DataFrame(data)
print(df)


# In[33]:


X = df[['Hours']].values
Y = df[['Marks']].values


# In[34]:


model = LinearRegression()
model.fit(X,Y)


# In[35]:


slope = (model.coef_[0])
intercept = (model.intercept_)


# In[36]:


print(f"Slope(m): {slope[0]}")
print(f"Intercept (b): {intercept[0]}")
print(f"Equation of the regression line: y = {slope[0]:.2f}x + {intercept[0]:.2f}")


# In[37]:


df['Predicted Marks'] = model.predict(X)


# In[38]:


df.head()


# In[40]:


plt.scatter(df['Hours'], df['Marks'], color = 'blue', label = 'Actual Marks')


# In[41]:


plt.plot(df['Hours'], df['Predicted Marks'], color = 'red', label = 'Regression line')
plt.xlabel('Hours studied')
plt.ylabel('Marks')
plt.title('Linear Regression: Hours vs Marks')
plt.legend()


# In[42]:


# predicting a grade for 3 hours of study
hours_to_predict = np.array([[3]])
predicted_grade = model.predict(hours_to_predict)[0]
print(f"Predicted grade for 3 hours of study: {predicted_grade}")


# In[ ]:




