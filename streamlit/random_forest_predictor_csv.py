# -*- coding: utf-8 -*-
"""
Created on Sat May 18 17:01:46 2024

@author: kshuk
"""

# Streamlit application to predict Instagram account authenticity using Random Forest

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained Random Forest model
Random_forest_model = joblib.load('model_weights/Random_forest_model.pkl')

# Function to preprocess user input
def preprocess_input(user_input):
    feature_names = ['profile pic', 'nums/length username', 'fullname words', 
                     'nums/length fullname', 'name==username', 'description length', 
                     'external URL', 'private', '#posts', '#followers', '#follows']
    user_df = pd.DataFrame([user_input], columns=feature_names)
    return user_df

# Function to make prediction using Random Forest model
def predict(user_data_df):
    prediction = Random_forest_model.predict(user_data_df)
    predicted_class = prediction[0]
    labels = ['Real Account', 'Fake Account']  
    predicted_label = labels[predicted_class]
    return predicted_label

# Main function to run the Streamlit app
def app():
    st.title('Instagram Account Authenticity Predictor Using Random Forest')

    # Radio button to select input method
    input_method = st.radio("Select Input Method:", ("Manual Input", "Upload CSV"))

    if input_method == "Manual Input":
        # Get user input
        st.subheader("Enter Instagram Account Information:")
        user_input = get_user_input()

        if user_input is not None:
            # Preprocess input
            user_data_df = preprocess_input(user_input)

            # Make prediction using Random Forest
            predicted_label = predict(user_data_df)

            # Display prediction
            st.subheader("Prediction:")
            st.write(f"The predicted profile is: {predicted_label}")

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
                user_data_df = data[required_columns]

                # Make predictions
                predictions = Random_forest_model.predict(user_data_df)

                # Count the number of real and fake accounts
                num_real_accounts = np.sum(predictions == 0)
                num_fake_accounts = np.sum(predictions == 1)

                # Display the number of real and fake accounts
               
                st.write(f"Number of Real Accounts: {num_real_accounts}")
                st.write(f"Number of Fake Accounts: {num_fake_accounts}")

                # Add predictions to the DataFrame
                data['Prediction'] = ['Real Account' if pred == 0 else 'Fake Account' for pred in predictions]

                # Display the DataFrame with predictions
                st.subheader("Predictions:")
                st.write(data)

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

# Run the app
if __name__ == '__main__':
    app()
