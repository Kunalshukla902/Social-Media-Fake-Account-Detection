# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:19:52 2024

@author: kshuk
"""

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
model = load_model('model_weights/modelANN.h5')
scaler_x = StandardScaler()  # Define scaler object

# Load the scaler that was used during training
scaler_x.fit(pd.read_csv('datasets/insta_train.csv').drop(columns=['fake']))

# Function to preprocess input data
def preprocess_input(df):
    feature_names = ['profile pic', 'nums/length username', 'fullname words', 
                     'nums/length fullname', 'name==username', 'description length', 
                     'external URL', 'private', '#posts', '#followers', '#follows']
    user_data_array = scaler_x.transform(df[feature_names])
    return user_data_array

# Function to make predictions
def predict(user_data_array):
    predictions = model.predict(user_data_array)
    predicted_classes = np.argmax(predictions, axis=1)
    labels = ['Real Account', 'Fake Account']  
    predicted_labels = [labels[pred] for pred in predicted_classes]
    return predicted_labels

# Function to get user input
def get_user_input():
    user_data = {}
    
    # Drop-down menu for profile pic
    profile_pic_options = ["No", "Yes"]
    profile_pic_selection = st.selectbox("Does the profile have a profile picture?", profile_pic_options)
    user_data['profile pic'] = 1 if profile_pic_selection == "Yes" else 0
    
    # Number input for nums/length username
    user_data['nums/length username'] = st.number_input("Enter the ratio of numbers to the length of the username:",
                                                         format='%f', step=0.01, value=0.00)
    
    # Number input for fullname words
    user_data['fullname words'] = st.number_input("Enter the number of words in the full name:", format='%d', value=0)
    
    # Number input for nums/length fullname
    user_data['nums/length fullname'] = st.number_input("Enter the ratio of numbers to the length of the full name:",
                                                        format='%f', step=0.01, value=0.00)
    
    # Drop-down menu for name==username
    name_username_options = ["No", "Yes"]
    name_username_selection = st.selectbox("Is the username same as the full name?", name_username_options)
    user_data['name==username'] = 1 if name_username_selection == "Yes" else 0
    
    # Number input for description length
    user_data['description length'] = st.number_input("Enter the length of the description (number of characters):",
                                                      format='%d', value=0)
    
    # Drop-down menu for external URL
    external_url_options = ["No", "Yes"]
    external_url_selection = st.selectbox("Does the profile have an external URL?", external_url_options)
    user_data['external URL'] = 1 if external_url_selection == "Yes" else 0
    
    # Drop-down menu for private
    private_options = ["No", "Yes"]
    private_selection = st.selectbox("Is the profile private?", private_options)
    user_data['private'] = 1 if private_selection == "Yes" else 0
    
    # Number input for #posts
    user_data['#posts'] = st.number_input("Enter the number of posts:", format='%d', value=0)
    
    # Number input for #followers
    user_data['#followers'] = st.number_input("Enter the number of followers:", format='%d', value=0)
    
    # Number input for #follows
    user_data['#follows'] = st.number_input("Enter the number following:", format='%d', value=0)
    
    if st.button("Submit"):
        return user_data

# Main function
def app():
    st.title('Welcome User')
    st.title("Instagram Account Authenticity Predictor Using Artificial Neural Network")
    
    # Radio button to select input method
    input_method = st.radio("Select Input Method:", ("Manual Input", "Upload CSV"))
    
    if input_method == "Manual Input":
        # Get user input
        st.subheader("Enter Instagram Account Information:")
        user_input = get_user_input()

        if user_input:
            # Preprocess input
            user_data_df = pd.DataFrame([user_input])

            # Make prediction
            user_data_array = preprocess_input(user_data_df)
            predicted_label = predict(user_data_array)

            # Display prediction
            st.subheader("Prediction:")
            st.write(f"The predicted profile is: {predicted_label[0]}")

    elif input_method == "Upload CSV":
        # File upload
        uploaded_file = st.file_uploader("Upload CSV file with account data", type="csv")

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)

            # Check if required columns are present
            required_columns = ['profile pic', 'nums/length username', 'fullname words', 
                                'nums/length fullname', 'name==username', 'description length', 
                                'external URL', 'private', '#posts', '#followers', '#follows']
            if not all(column in data.columns for column in required_columns):
                st.error("Uploaded CSV file must contain the required columns.")
            else:
                # Preprocess input
                user_data_array = preprocess_input(data)

                # Make predictions
                predictions = predict(user_data_array)

                # Add predictions to the DataFrame
                data['Prediction'] = predictions
                
                # Count the number of real and fake accounts
                num_real_accounts = sum(1 for label in predictions if label == 'Real Account')
                num_fake_accounts = sum(1 for label in predictions if label == 'Fake Account')

                # Display the number of real and fake accounts
                st.write(f"Number of Real Accounts: {num_real_accounts}")
                st.write(f"Number of Fake Accounts: {num_fake_accounts}")

                # Display the DataFrame with predictions
                st.subheader("Predictions:")
                st.write(data)

                # Download the DataFrame as a new CSV file
                csv = data.to_csv(index=False)
                st.download_button(label="Download Predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")

# Run the app
if __name__ == '__main__':
    app()


# Run the app
if __name__ == '__main__':
    app()
