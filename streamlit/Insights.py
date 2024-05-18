# =============================================================================
# import streamlit as st
# 
# def app():
#     st.title("Page5")    
#     st.subheader('Trending thoughts')
# =============================================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def app():
    st.title("Fake Social Media Detection - Insights")

    # 1. Dataset information (JPEG format)
    st.header("1. Dataset Information")
    st.write("Training Dataset Details:")
    st.write("- Number of rows: 576", " And Number of columns: 12")
    #st.write("- Number of columns: 12")

    st.write("Test Dataset Details:")
    st.write("- Number of rows: 120", " And Number of columns: 12")
    #st.write("- Number of columns: 12")
    train_image = Image.open('train_data_info.png')
    st.image(train_image, caption='Train Dataset', use_column_width=True)
    test_image = Image.open('test_data_info.png')
    st.image(test_image, caption='Test Dataset', use_column_width=True)
    dataset_image = Image.open('Dataset_info.png')
    st.image(dataset_image, caption='Dataset Details', use_column_width=True)

    # 2. Data Visualizations (JPEG/PNG format)
    st.header("2. Data Visualizations")
    visualization_image = Image.open('v_heatmap.png')
    st.image(visualization_image, caption='Correlation plot- Heatmap', use_column_width=True)
    visualization_image2 = Image.open('v_realVsfake_histogram.png')
    st.image(visualization_image2, caption='followers with bins and show fake value counts', use_column_width=True)
    
    

    # 3. Model Selection and Details
    st.header("3. Model Selection and Details")

    # 3a. Artificial Neural Network (ANN) details
    st.subheader("a) Artificial Neural Network (ANN)")
    # Describe the ANN model
    st.write("The Artificial Neural Network (ANN) used for fake social media detection is a deep learning model.")
    st.write("It consists of multiple layers of interconnected neurons, including input, hidden, and output layers.")
    st.write("The activation function used in the hidden layers is typically ReLU (Rectified Linear Unit), while the output layer uses a  softmax activation for binary classification.")
    st.write("This ANN architecture is designed to process input data with 11 features and predict the authenticity of an Instagram account as either real or fake based on the learned patterns and features extracted through its layers. The inclusion of dropout layers helps prevent overfitting and improve the generalization capability of the model.")
    
    st.write("The activation function used in the hidden layers is typically ReLU (Rectified Linear Unit), while the output layer uses a  softmax activation for binary classification.")
    st.write("This ANN architecture is designed to process input data with 11 features and predict the authenticity of an Instagram account as either real or fake based on the learned patterns and features extracted through its layers. The inclusion of dropout layers helps prevent overfitting and improve the generalization capability of the model.")

    st.markdown("### Training Accuracy:")
    st.write("The training dataset achieved an accuracy of 95.17% during training.")

    st.markdown("### Validation Accuracy:")
    st.write("The validation dataset achieved an accuracy of 91.38% during training.")
    
    st.markdown("### Test Accuracy:")
    st.write("The tes dataset achieved an accuracy of 89%.")
   

# =============================================================================
#     # Display ANN model details, accuracy, and plots (if available)
#     ann_details_image = Image.open('ann_model_details.png')
#     st.image(ann_details_image, caption='ANN Model Details', use_column_width=True)
# 
# =============================================================================
    # 3b. Decision Tree details
    st.subheader("b) Decision Tree")
    st.write ("""
    The decision tree model is a type of supervised learning algorithm used for classification tasks. 
    It consists of a tree-like structure where each internal node represents a feature, each branch represents a decision rule, 
    and each leaf node represents the outcome (class label). During prediction, the model traverses the tree based on 
    the feature values of the input data until it reaches a leaf node.

    This decision tree model was trained to predict whether an Instagram account is real or fake based on various features 
    such as profile picture presence, username characteristics, follower counts, and other account attributes.
    """)
    
    st.markdown("### Training Accuracy:")
    st.write("The training dataset achieved an accuracy of 100% during training.")
    
    st.markdown("### Test Accuracy:")
    st.write("The tes dataset achieved an accuracy of 87%.")


    # Display Decision Tree model details, accuracy, and plots (if available)
    dt_details_image = Image.open('decision tree.png')
    st.image(dt_details_image, caption='Decision Tree Details', use_column_width=True)

    # 4. Screenshots of Code Snippets
    st.header("4. Screenshots of Code Snippets")
    code_snippet_image = Image.open('ANN_code.png')
    st.image(code_snippet_image, caption='Code Snippet Of ANN', use_column_width=True)
    code_snippet_image2 = Image.open('Decision_tree_code.png')
    st.image(code_snippet_image2, caption='Code Snippet of Decision Tree', use_column_width=True)

if __name__ == "__main__":
    app()
