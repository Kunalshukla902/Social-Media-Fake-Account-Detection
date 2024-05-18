# -*- coding: utf-8 -*-
"""
Created on Sat May 18 13:36:56 2024

@author: kshuk
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_filepath = "C:/Users/kshuk/3D Objects/major project/major code/main folder/project code/insta_train.csv"
test_filepath = "C:/Users/kshuk/3D Objects/major project/major code/main folder/project code/insta_test.csv"
insta_train = pd.read_csv(train_filepath)
insta_test = pd.read_csv(test_filepath)

insta_train.head()
insta_train.describe()
insta_train.info()

print(insta_train.shape)
print(insta_test.shape)

print(insta_train.isna().values.any().sum())
print(insta_train.isna().values.any().sum())

corr= insta_train.corr()
sns.heatmap(corr)

train_Y = insta_train.fake
train_Y = pd.DataFrame(train_Y)
train_Y.tail(12)

train_X = insta_train.drop(columns="fake")
train_X.head()

test_Y = insta_test.fake
test_Y =pd.DataFrame(test_Y)
test_Y.tail(12)

test_X = insta_test.drop(columns="fake")
test_X.head()


#Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

logreg = LogisticRegression()
model1 = logreg.fit(train_X,train_Y)
logreg_predict = model1.predict(test_X)

accuracy_score(logreg_predict,test_Y)
print(classification_report(logreg_predict,test_Y))

def plot_confusion_matrix(test_Y,predict_y):
    C = confusion_matrix(test_Y,predict_y)
    A = (((C.T)/(C.sum(axis=1))).T)
    B = (C/C.sum(axis=0))
    plt.figure(figsize=(20,4))
    labels = [1,2]
    cmap=sns.light_palette("seagreen")
    plt.subplot(1,3,1)
    sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels,yticklabels=labels)
    plt.xlabel("Predicted Class")
    plt.ylabel("Original Class")
    plt.title("Confusion matrix")
    plt.subplot(1,3,2)
    sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels,yticklabels=labels)
    plt.xlabel("Predicted Class")
    plt.ylabel("Original Class")
    plt.title("Precision matrix")
    plt.subplot(1,3,3)
    sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels,yticklabels=labels)
    plt.xlabel("Predicted Class")
    plt.ylabel("Original Class")
    plt.title("Recall matrix")
    plt.show()
    
    
plot_confusion_matrix(test_Y,logreg_predict)

# Random Forest

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
model3= rfc.fit(train_X,train_Y)
rfc_predict = model3.predict(test_X)

accuracy_score(rfc_predict,test_Y)
print(classification_report(rfc_predict,test_Y))
plot_confusion_matrix(test_Y,rfc_predict)

# XGBoost
from xgboost import XGBClassifier
xgb= XGBClassifier()
model4 = xgb.fit(train_X,train_Y)
xgb_predict = model4.predict(test_X)

accuracy_score(xgb_predict,test_Y)
print(classification_report(xgb_predict,test_Y))
plot_confusion_matrix(test_Y,xgb_predict)

print('Logistic Regression Accuracy:',accuracy_score(logreg_predict,test_Y))
print('RFC Accuracy:',accuracy_score(rfc_predict,test_Y))
print('XGB Accuracy:',accuracy_score(xgb_predict,test_Y))


## user input test -- Random forest

import joblib

# Define the filename for saving the trained model
model_filename = 'Random_forest_model.pkl'

# Save the trained model to the specified filename using joblib
joblib.dump(model3, model_filename)

print("Model saved successfully at:", model_filename)

df_test = pd.read_csv('C:/Users/kshuk/3D Objects/major project/major code/main folder/project code/insta_test.csv')

df_test.head()

X_test = df_test.drop('fake', axis=1)

y_pred = model3.predict(X_test)

y_pred

accuracy_score(df_test['fake'], y_pred)

print(confusion_matrix(df_test['fake'], y_pred))

print(classification_report(df_test['fake'], y_pred))

plt.figure(figsize=(10,10))
plt.barh(X_test.columns, model3.feature_importances_)
plt.show()

import joblib
import pandas as pd

# Load the trained model from file
model_filename = 'C:/Users/kshuk/3D Objects/major project/major code/main folder/project code/Random_forest_model.pkl'
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
