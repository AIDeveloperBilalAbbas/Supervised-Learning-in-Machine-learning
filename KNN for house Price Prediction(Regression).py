#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import numpy as np


# In[3]:


# Simple dataset: [Size (sqft), Number of Bedrooms] --> Price (in thousands)
X = np.array([
    [1500, 3],  # Size: 1500 sqft, Bedrooms: 3
    [1800, 3],  # Size: 1800 sqft, Bedrooms: 3
    [2400, 4],  # Size: 2400 sqft, Bedrooms: 4
    [3000, 5],  # Size: 3000 sqft, Bedrooms: 5
    [3500, 5],  # Size: 3500 sqft, Bedrooms: 5
    [4000, 6],  # Size: 4000 sqft, Bedrooms: 6
])


# In[4]:


# House prices (in thousands)
y = np.array([400, 450, 500, 600, 650, 700])


# In[5]:


# Create and train the KNN regressor model (using k=3 neighbors)
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X, y)


# In[6]:


# Predict the price of a new house
new_house = np.array([[2800, 4]])  # Size: 2800 sqft, Bedrooms: 4
predicted_price = knn.predict(new_house)


# In[7]:


# Print the predicted price
print(f"Predicted price for a house of 2800 sqft and 4 bedrooms: ${predicted_price[0]:.2f}K")


# In[13]:


# Plot the data (Size vs Price)
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], y, color='blue', label='Existing Houses', s=100, edgecolor='black')

# Plot the predicted price for the new house
plt.scatter(new_house[0][0], predicted_price, color='green', marker='x', s=200, label='New House Prediction')

# Adding titles and labels
plt.title('House Price Prediction Using K-Nearest Neighbors', fontsize=16, weight='bold')
plt.xlabel('Size (sqft)', fontsize=14)
plt.ylabel('Price (in thousands)', fontsize=14)

# Add a legend
plt.legend(loc='upper left', fontsize=12)

# Plot the predicted price for the new house
plt.axvline(x=new_house[0][0], color='green', linestyle=':', linewidth=1)

# Display grid
plt.grid(True, linestyle='--', alpha=0.5)


# In[ ]:




