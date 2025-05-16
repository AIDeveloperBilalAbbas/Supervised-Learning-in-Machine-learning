#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Step 1: Import libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# In[2]:


# Step 2: Sample email dataset
emails = [
    "Win a free iPhone now",       # spam
    "Lowest price on car loans",   # spam
    "Hi, can we meet tomorrow?",   # not spam
    "Don't forget our meeting",    # not spam
    "Congratulations, you won!",   # spam
    "Let's grab lunch today",      # not spam
]


# In[3]:


labels = [1, 1, 0, 0, 1, 0]  # 1 = spam, 0 = not spam


# In[4]:


# Step 3: Convert text into numeric features (Bag of Words)
vectorizer = CountVectorizer()
features = vectorizer.fit_transform(emails)


# In[5]:


# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)


# In[6]:


# Step 5: Train the classifier
model = MultinomialNB()
model.fit(X_train, y_train)


# In[7]:


# Step 6: Predict on test set
predictions = model.predict(X_test)


# In[8]:


# Step 7: Evaluate the model
print("Accuracy:", accuracy_score(y_test, predictions))


# In[23]:


# Step 8: Test on new data
new_emails = ["Claim your free prize now", "Lowest price on bile loans"]
new_features = vectorizer.transform(new_emails)
new_predictions = model.predict(new_features)


# In[24]:


# Show predictions
for email, label in zip(new_emails, new_predictions):
    print(f"'{email}' => {'Spam' if label == 1 else 'Not Spam'}")


# In[ ]:




