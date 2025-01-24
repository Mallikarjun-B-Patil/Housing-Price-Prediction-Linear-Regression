import streamlit as st
import numpy as np
from joblib import load

# Load the trained model
try:
    model = load('model.joblib')
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'model.joblib' is in the same directory.")

# App Header
st.title("House Price Prediction App")
st.write("Enter the details of the house below to get a predicted price.")

# Input fields for housing features
st.sidebar.header("Input Features")

area = st.sidebar.number_input("Area (in sq feet)", min_value=1, max_value=20000, value=1000)
bedrooms = st.sidebar.number_input("Bedrooms", min_value=1, max_value=6, value=2)
bathrooms = st.sidebar.number_input("Bathrooms", min_value=1, max_value=4, value=1)
stories = st.sidebar.number_input("Stories", min_value=1, max_value=4, value=1)
parking = st.sidebar.number_input("Parking (Number of spaces)", min_value=0, max_value=3, value=1)

mainroad = st.sidebar.selectbox("Is it on the Main Road?", ('Yes', 'No'))
guestroom = st.sidebar.selectbox("Does it have a Guest Room?", ('Yes', 'No'))
basement = st.sidebar.selectbox("Does it have a Basement?", ('Yes', 'No'))
hotwaterheating = st.sidebar.selectbox("Does it have Hot Water Heating?", ('Yes', 'No'))
airconditioning = st.sidebar.selectbox("Does it have Air Conditioning?", ('Yes', 'No'))
prefarea = st.sidebar.selectbox("Is it in a Preferred Area?", ('Yes', 'No'))
# furnishingstatus = st.sidebar.selectbox("Furnishing Status", ('Furnished', 'Semi-furnished', 'Unfurnished'))

# Map categorical input values to numeric using a dictionary
label_mapping = {
    'Yes': 1,
    'No': 0,
    #'Furnished': 2,
    #'Semi-furnished': 1,
    #'Unfurnished': 0,
}
mainroad = label_mapping[mainroad]
guestroom = label_mapping[guestroom]
basement = label_mapping[basement]
hotwaterheating = label_mapping[hotwaterheating]
airconditioning = label_mapping[airconditioning]
prefarea = label_mapping[prefarea]
#furnishingstatus = label_mapping[furnishingstatus]

# Prediction
if st.sidebar.button("Predict Price"):
    try:
        # Ensure the input is in the correct format (2D array)
        features = np.array([[area, bedrooms, bathrooms, stories, parking,
                              mainroad, guestroom, basement, hotwaterheating,
                              airconditioning, prefarea]])

        prediction = model.predict(features)

        # Display the prediction result
        st.header("Prediction Result")
        st.success(f"The predicted house price is: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
