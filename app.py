import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Title and Problem Statement
st.title('CAR SALES PREDICTION APP')

st.write("### Problem Statement")
st.write("The goal of this project is to predict the car purchase amount based on various features such as customer demographics, car preferences, and other factors. Accurate predictions can help car dealerships forecast their sales and make informed business decisions.")

# Upload and Display Dataset
st.subheader('Upload the file')
df = pd.read_csv('app/car_purchasing.csv', encoding='ISO-8859-1')
st.write(df.head())

st.subheader('Summary statistics')
st.write('Brief summary statistics of the dataset')
st.write(df.describe())  # corrected typo

# Load Pre-trained Model
model = joblib.load('app/best_model.pkl')

# Country Selection Input
country = df['country'].unique()
country = st.selectbox('Which country are you in', country)

# Gender Selection
gender = st.radio('What gender are you?', ['Female', 'Male'])
gender_value = 1 if gender == 'Female' else 0

# Additional Input Fields
age = st.number_input('How old are you?', min_value=18, max_value=80)
annual_salary = st.number_input('What is your annual salary', max_value=10000000)
credit_card_debt = st.number_input('What is your credit card debt', max_value=10000000)
net_worth = st.number_input('What is your net worth', max_value=10000000)

# Create DataFrame for Model Input
user_data = {
    'country': country,
    'gender': gender_value,
    'age': age,
    'annual Salary': annual_salary,
    'credit card debt': credit_card_debt,
    'net worth': net_worth
}

input_df = pd.DataFrame(user_data, index=[0])

# Encode the 'country' column
label_encoder = LabelEncoder()
input_df['country'] = label_encoder.fit_transform(input_df['country'])

# Load scaler and ensure feature alignment
scaler = joblib.load('Data/scaler.pkl')

# Debugging outputs
#st.write("Scaler expected feature names:", scaler.feature_names_in_)
#st.write("Input DataFrame columns:", input_df.columns)

# Reindex the input DataFrame to match scaler expected columns
input_df = input_df.reindex(columns=scaler.feature_names_in_, fill_value=0)

# Prediction when the button is clicked
if st.button('Predict'):
    # Scale input data
    input_scaled = scaler.transform(input_df)

    # Predict using the pre-trained model
    prediction = model.predict(input_scaled)
    
    # Show the predicted result
    st.write("Predicted Car Purchase Amount:", prediction[0])
