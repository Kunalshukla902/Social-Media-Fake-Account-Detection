# -*- coding: utf-8 -*-
"""decision-tree.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vPXYOuUqFR7oeThZHxJs01jd4gPO_sPp
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#from google.colab import drive
#drive.mount('/content/drive')

df_train = pd.read_csv('datasets/insta_train.csv')

df_train.head()

df_train.shape

df_train.info()

df_train.describe()

df_train.isnull().sum()

df_train['fake'].value_counts()

df_train.nunique()

df_train.corr()

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
sns.heatmap(df_train.corr(), annot=True, cmap='coolwarm')

plt.show()

df_train.hist(figsize=(10,10))
plt.show()

plt.figure(figsize=(20,10))
sns.countplot(x='#followers', hue='fake', data=df_train)
plt.show()

# create bins for #followers column
bins = [0, 25,  50, 100, 200, 300, 400, 500, 1000, 5000, 10000]

# cut the #followers column into the bins
df_train['#followers_bins'] = pd.cut(df_train['#followers'], bins=bins)

# plot #followers with bins and show fake value counts
plt.figure(figsize=(20,10))
sns.countplot(x='#followers_bins', hue='fake', data=df_train)
plt.show()

# plot nums/length username and show fake value counts
plt.figure(figsize=(20,10))
sns.countplot(x='nums/length username', hue='fake', data=df_train)
plt.show()

# how is fake distributed amongst private accounts
plt.figure(figsize=(20,10))
sns.countplot(x='private', hue='fake', data=df_train)
plt.show()

from pandas.plotting import scatter_matrix
scatter_matrix(df_train, figsize=(22,22), diagonal='kde', c=df_train['fake'], cmap='coolwarm')


plt.show()

df_train.plot(kind='box', subplots=True, layout=(4,4), figsize=(10,10))

plt.show()



from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X = df_train.drop(['#followers_bins', 'fake'], axis=1)
y = df_train['fake']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()

print(X_train.dtypes)
print(y_train.dtypes)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy_score(y_test, y_pred)

confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

# Calculating and printing test accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {test_accuracy}")

# Calculating and printing training accuracy
train_accuracy = accuracy_score(y_train, model.predict(X_train))
print(f"Training Set Accuracy: {train_accuracy}")

# explain model with tree plot
from sklearn import tree
plt.figure(figsize=(20,20))
tree.plot_tree(model, filled=True, feature_names=X.columns, class_names=['real', 'fake'], rounded=True)

plt.show()

model.feature_importances_

# plot feature importance
plt.figure(figsize=(10,10))
plt.barh(X.columns, model.feature_importances_)
plt.show()



import joblib

# Define the filename for saving the trained model
model_filename = 'model_weights/decision_tree_model.pkl'

# Save the trained model to the specified filename using joblib
joblib.dump(model, model_filename)

print("Model saved successfully at:", model_filename)

df_test = pd.read_csv('datasets/insta_test.csv')

df_test.head()

X_test = df_test.drop('fake', axis=1)

y_pred = model.predict(X_test)

y_pred

accuracy_score(df_test['fake'], y_pred)

print(confusion_matrix(df_test['fake'], y_pred))

print(classification_report(df_test['fake'], y_pred))

plt.figure(figsize=(10,10))
plt.barh(X_test.columns, model.feature_importances_)
plt.show()

import joblib
import pandas as pd

# Load the trained model from file
model_filename = 'model_weights/decision_tree_model.pkl'
loaded_model = joblib.load(model_filename)

# Define the input features based on user_data keys
input_features = [
    'profile pic', 'nums/length username', 'fullname words',
    'nums/length fullname', 'name==username', 'description length',
    'external URL', 'private', '#posts', '#followers', '#follows'
]

# Function to get user inputs and make prediction
def predict_account(user_data):
    # Create a DataFrame from user_data dictionary
    user_input = pd.DataFrame([user_data], columns=input_features)

    # Make prediction using the loaded model
    prediction = loaded_model.predict(user_input)

    # Display prediction result
    if prediction[0] == 1:
        print("Prediction: Fake Account")
    else:
        print("Prediction: Real Account")

# Example usage:
user_data = {}
user_data['profile pic'] = int(input("Does the profile have a profile picture (1 for yes, 0 for no): "))
user_data['nums/length username'] = float(input("Enter the ratio of numbers to the length of the username: "))
user_data['fullname words'] = int(input("Enter the number of words in the full name: "))
user_data['nums/length fullname'] = float(input("Enter the ratio of numbers to the length of the full name: "))
user_data['name==username'] = int(input("Is the username same as the full name (1 for yes, 0 for no): "))
user_data['description length'] = int(input("Enter the length of the description (number of characters): "))
user_data['external URL'] = int(input("Does the profile have an external URL (1 for yes, 0 for no): "))
user_data['private'] = int(input("Is the profile private (1 for yes, 0 for no): "))
user_data['#posts'] = int(input("Enter the number of posts: "))
user_data['#followers'] = int(input("Enter the number of followers: "))
user_data['#follows'] = int(input("Enter the number following: "))

# Call the predict_account function with user_data as input
predict_account(user_data)