#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


# In[10]:


# Data: [Rating, Duration]
X = np.array([
    [8.5, 130],  # Action
    [7.8, 125],  # Action
    [8.0, 140],  # Action
    [6.5, 95],   # Comedy
    [7.0, 100],  # Comedy
    [6.8, 90],   # Comedy
])


# In[12]:


y = np.array([0, 0, 0, 1, 1, 1])  # 0: Action, 1: Comedy


# In[13]:


# Train KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)


# In[6]:


# New movie: predict its genre
new_movie = np.array([[7.5, 110]])
prediction = knn.predict(new_movie)[0]


# In[14]:


# Print the prediction
genre = 'Comedy' if prediction == 1 else 'Action'
print(f"The predicted genre for movie with rating 7.5 and duration 110 min is: {genre}")


# In[17]:


# Plotting
plt.figure(figsize=(10, 6))
for i, label in enumerate(y):
    color = 'red' if label == 0 else 'blue'
    plt.scatter(X[i][0], X[i][1], color=color, label='Action' if label == 0 else 'Comedy')

    # Plot the new movie
plt.scatter(new_movie[0][0], new_movie[0][1], color='green', marker='x', s=100, label='New Movie')

# Adding labels for the axes
plt.xlabel('Rating', fontsize=14)
plt.ylabel('Duration (minutes)', fontsize=14)

# Adding title
plt.title('Movie Genre Classification (Action vs Comedy)', fontsize=16, weight='bold')

# Add legend
plt.legend(loc='upper left', fontsize=12, title='Genre')

# Set grid
plt.grid(True, linestyle='--', alpha=0.5)

# Set axis limits for better visualization
plt.xlim(5, 10)  # Rating between 5 and 10
plt.ylim(80, 160)  # Duration between 80 and 160 minutes

# Add a vertical line for the new movie to indicate classification
plt.axvline(x=new_movie[0][0], color='green', linestyle=':', linewidth=1)

# Adding text annotations for the points
for i, label in enumerate(y):
    plt.text(X[i][0] + 0.1, X[i][1] + 1, 'Action' if label == 0 else 'Comedy', fontsize=9)

# Show plot
plt.show()


# In[ ]:




