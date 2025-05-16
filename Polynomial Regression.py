#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# In[2]:


# Step 1: Sample input data (X) and output data (y)
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # input (feature)
y = np.array([4, 9, 16, 25, 36])             # output (target) ~ looks like y = x^2 + something


# In[4]:


# Step 2: Create polynomial features (degree = 2 means x and x^2)
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)


# In[5]:


# Step 3: Create and train the model
model = LinearRegression()
model.fit(x_poly, y)


# In[6]:


# Step 4: Predict using the trained model
y_pred = model.predict(x_poly)


# In[7]:


# Step 5: Plot the results
plt.scatter(x, y, color='blue', label='Actual Data')        # real data points
plt.plot(x, y_pred, color='red', label='Polynomial Curve')  # predicted curve
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression Example')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




