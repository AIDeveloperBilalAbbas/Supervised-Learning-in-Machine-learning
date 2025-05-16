#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


# In[17]:


# 1. Simulated 2D data for spam (1) and not spam (0)
X = np.array([
    [1, 2], [2, 3], [3, 3],     # not spam
    [6, 6], [7, 8], [8, 8]      # spam
])
y = [0, 0, 0, 1, 1, 1]         # Labels: 0 = not spam, 1 = spam


# In[18]:


# 2. Train the SVM model (linear)
clf = svm.SVC(kernel='linear')
clf.fit(X, y)


# In[19]:


# 3. Get weights and bias
w = clf.coef_[0]
b = clf.intercept_[0]
print(f"Weights: {w}, Bias: {b}")


# In[20]:


# 4. Line equation: y = mx + c
m = -w[0] / w[1]
c = -b / w[1]


# In[21]:


# 5. Margins: offset from decision line
margin = 1 / np.sqrt(np.sum(w ** 2))
offset = np.sqrt(1 + m**2) * margin


# In[22]:


# 6. Create line space for drawing
x_vals = np.linspace(0, 10, 100)
y_hyperplane = m * x_vals + c
y_margin_up = m * x_vals + (c + offset)
y_margin_down = m * x_vals + (c - offset)


# In[23]:


# 7. Plot all on one graph
plt.figure(figsize=(8, 6))


# In[27]:


# Plot data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=70, edgecolors='k', label='Emails')

# Plot decision boundary and margins
plt.plot(x_vals, y_hyperplane, 'k-', label='Decision Boundary')
plt.plot(x_vals, y_margin_up, 'k--', label='Margins')
plt.plot(x_vals, y_margin_down, 'k--')

# Highlight support vectors
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=120, facecolors='none', edgecolors='k', linewidths=1.5,
            label='Support Vectors')

# Decorate the plot
plt.title("SVM: Email Spam Classification")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:




