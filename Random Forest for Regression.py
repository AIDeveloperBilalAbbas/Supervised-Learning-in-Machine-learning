#!/usr/bin/env python
# coding: utf-8

# In[12]:


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd


# In[15]:


# Step 1: Load the data
housing = fetch_california_housing()
X = housing.data              # Features
y = housing.target            # Target (house prices)


# In[17]:


# Create the DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)


# In[18]:


df


# In[4]:


# Step 2: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[5]:


# Step 3: Train the model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)


# In[6]:


# Step 4: Predict on test data
y_pred = rf_regressor.predict(X_test)


# In[7]:


# Step 5: Evaluate
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))


# In[22]:


# Plot actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='teal', edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Random Forest Regression: Actual vs Predicted')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[23]:


# --- 2. Residual Plot: Actual vs Predicted Difference ---
residuals = y_test - y_pred

plt.figure(figsize=(10, 5))
plt.scatter(y_pred, residuals, alpha=0.6, color='tomato', edgecolor='k')
plt.axhline(0, color='blue', linestyle='--', linewidth=2)
plt.title("Residuals: Actual - Predicted")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:




