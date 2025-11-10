import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('car_price_model.pkl', 'rb'))

# Streamlit app title
st.title("🚗 Car Price Prediction App")
st.write("This app predicts the **Selling Price of a car (in ₹ lakh)** based on its Age, Mileage, and Horsepower.")

# Input fields
age = st.number_input("Car Age (in years)", min_value=0, max_value=20, value=5)
mileage = st.number_input("Mileage (in km/litre)", min_value=5, max_value=30, value=15)
horsepower = st.number_input("Horsepower (HP)", min_value=50, max_value=300, value=100)

# Predict button
if st.button("🔮 Predict Selling Price"):
    features = np.array([[age, mileage, horsepower]])
    prediction = model.predict(features)
    st.success(f"Estimated Selling Price: ₹ {prediction[0]:.2f} Lakh")

# Footer
st.caption("Built with ❤️ using Streamlit and Machine Learning")
