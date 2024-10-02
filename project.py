import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('trained_model.pkl')

# Set up the Streamlit interface
st.title('Consumer Purchase Prediction App')

# Collect user input features
product_price = st.number_input('Product Price', min_value=0.0, step=0.1)
customer_age = st.number_input('Customer Age', min_value=0, max_value=100)
customer_gender = st.selectbox('Customer Gender', ['Male', 'Female'])
purchase_frequency = st.number_input('Purchase Frequency', min_value=0, step=1)
customer_satisfaction = st.slider('Customer Satisfaction', min_value=1, max_value=5)

# Map gender to the appropriate value used in the model
gender_map = {'Male': 0, 'Female': 1}
customer_gender = gender_map[customer_gender]

# Prepare input data for prediction (ensure the order matches the model's input)
input_data = np.array([[product_price, customer_age, customer_gender, purchase_frequency, customer_satisfaction]])

# Make prediction
if st.button('Predict'):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.write('The user is likely to purchase the product!')
    else:
        st.write('The user is not likely to purchase the product.')
