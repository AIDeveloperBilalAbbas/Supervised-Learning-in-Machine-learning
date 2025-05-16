#!/usr/bin/env python
# coding: utf-8

# <center><b>Multi Linear Regression Practical Implementation</b></center>
# 

# <center><b>Importing Libraries</b></center>

# In[44]:


# Step 1: Import required libraries
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd


# <center><b>Importing dataset from sklearn</b></center>

# In[29]:


# Step 2: Load the dataset
diabetes = load_diabetes()


# In[30]:


# Step 3: Convert to DataFrame
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target
print(df)


# <center><b>Define X and Y (dependent and independent variable)</b></center>

# In[31]:


# Step 4: Split data into features (X) and target (y)
X = df.drop('target', axis=1)   # Independent variables
y = df['target']                # Dependent variable


# In[32]:


X


# In[33]:


y


# <center><b>set the dataset set in training set and test set</b></center>

# In[34]:


# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# <center><b>Train the model on training dataset</b></center>

# In[35]:


# Step 6: Create the Linear Regression model
model = LinearRegression()


# In[36]:


# Step 7: Train the model
model.fit(X_train, y_train)


# <center><b>Predict the test set result</b></center>

# In[37]:


y_pred = model.predict(X_test)
print(y_pred)


# In[40]:


model.predict([[-0.001882,-0.044642,	-0.051474,	-0.026328,	-0.008449,	-0.019163,	0.074412,	-0.039493,	-0.068332,	-0.092204]])


# <center><b>Evaluate the Model</b></center>

# In[42]:


# Step 9: Evaluate the model
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score (Accuracy):", r2_score(y_test, y_pred))


# <center><b>Plot the results</b></center>

# In[45]:


# Predict the test set results
y_pred = model.predict(X_test)

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolors='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Ideal line
plt.xlabel('Actual Target Values')
plt.ylabel('Predicted Target Values')
plt.title('Actual vs Predicted Diabetes Progression')
plt.grid(True)
plt.show()


# <center><b>Predicted values</b></center>

# In[46]:


# Create a DataFrame to show actual, predicted, and difference
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Difference (Actual - Predicted)': y_test - y_pred
})

# Display the first 10 rows
print(results_df.head(10))


# In[ ]:




