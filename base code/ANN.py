# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 18:04:04 2024

@author: kshuk
"""

# Importing Libraries and Dataset

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,accuracy_score,roc_curve,confusion_matrix

# Load the training dataset
instagram_df_train=pd.read_csv('insta_train.csv')
instagram_df_train

# Load the testing data
instagram_df_test=pd.read_csv('insta_test.csv')
instagram_df_test

instagram_df_train.head()
instagram_df_train.tail()

instagram_df_test.head()
instagram_df_test.tail()

#Performing Exploratory Data Analysis EDA

# Getting dataframe info
instagram_df_train.info()

# Get the statistical summary of the dataframe
instagram_df_train.describe()

# Checking if null values exist
instagram_df_train.isnull().sum()

# Get the number of unique values in the "profile pic" feature
instagram_df_train['profile pic'].value_counts()

# Get the number of unique values in "fake" (Target column)
instagram_df_train['fake'].value_counts()

instagram_df_test.info()

instagram_df_test.describe()

instagram_df_test.isnull().sum()

instagram_df_test['fake'].value_counts()

# Perform Data Visualizations

# Visualize the data
sns.countplot(instagram_df_train['fake'])
plt.show()

# Visualize the private column data
sns.countplot(instagram_df_train['private'])
plt.show()

# Visualize the "profile pic" column data
sns.countplot(instagram_df_train['profile pic'])
plt.show()

# Visualize the data
plt.figure(figsize = (20, 10))
sns.distplot(instagram_df_train['nums/length username'])
plt.show()

# Correlation plot
plt.figure(figsize=(20, 20))
cm = instagram_df_train.corr()
ax = plt.subplot()
sns.heatmap(cm, annot = True, ax = ax)
plt.show()

sns.countplot(instagram_df_test['fake'])

sns.countplot(instagram_df_test['private'])

sns.countplot(instagram_df_test['profile pic'])

# Preparing Data to Train the Model

# Training and testing dataset (inputs)
X_train = instagram_df_train.drop(columns = ['fake'])
X_test = instagram_df_test.drop(columns = ['fake'])
X_train

X_test

# Training and testing dataset (Outputs)
y_train = instagram_df_train['fake']
y_test = instagram_df_test['fake']

y_train

y_test

# Scale the data before training the model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

y_train = tf.keras.utils.to_categorical(y_train, num_classes = 2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes = 2)

y_train

y_test

# print the shapes of training and testing datasets 
X_train.shape, X_test.shape, y_train.shape, y_test.shape

Training_data = len(X_train)/( len(X_test) + len(X_train) ) * 100
Training_data

Testing_data = len(X_test)/( len(X_test) + len(X_train) ) * 100
Testing_data

# Building and Training Deep Training Model

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(50, input_dim=11, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2,activation='softmax'))

model.summary()

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

epochs = 50

history = model.fit(X_train, y_train, epochs = epochs,  verbose = 1, validation_split = 0.1)



# Save the model after training
model.save('modelANN.h5')  # Replace 'your_model.h5' with your desired filename



print(history.history.keys())

# Plot training history: accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Print accuracy from the final epoch
final_epoch = epochs
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"Training accuracy: {final_train_acc:.4f} - val_accuracy: {final_val_acc:.4f}")


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss Progression During Training/Validation')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

predicted = model.predict(X_test)

predicted_value = []
test = []
for i in predicted:
    predicted_value.append(np.argmax(i))
    
for i in y_test:
    test.append(np.argmax(i))

print(classification_report(test, predicted_value))

plt.figure(figsize=(10, 10))
cm=confusion_matrix(test, predicted_value)
sns.heatmap(cm, annot=True)
plt.show()


# =============================================================================
# Let us explore 
# =============================================================================

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model (replace with your saved model filename)
model = load_model('modelANN.h5')

# Function to get user input
def get_user_input():
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
  user_data['# followers'] = int(input("Enter the number of followers: "))
  user_data['# follows'] = int(input("Enter the number following: "))
  return user_data

# Get user data
user_input = get_user_input()

# Convert user data to a NumPy array
user_data_array = np.array([list(user_input.values())])

# Preprocess the user data (if necessary) using the same preprocessing applied during training
# For example, if you used standardization, apply it here
user_data_array = scaler_x.transform(user_data_array)  # Assuming scaler_x from your training data

# Make prediction
prediction = model.predict(user_data_array)
predicted_class = np.argmax(prediction)

# Map predicted class to labels (real/fake) based on your target labels
labels = ['Real account', 'Fake account']  # Modify these labels based on your actual class names
predicted_label = labels[predicted_class]

print(f"Predicted profile: {predicted_label}")

