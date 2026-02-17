
import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction App")

st.write("Enter house details below to predict the sale price.")

# User Inputs
overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
gr_liv_area = st.number_input("Ground Living Area (sq ft)", min_value=500, max_value=5000, value=1500)
garage_cars = st.slider("Garage Capacity (Cars)", 0, 4, 1)
total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=3000, value=800)

if st.button("Predict Price"):
    features = np.array([[overall_qual, gr_liv_area, garage_cars, total_bsmt_sf]])
    prediction = model.predict(features)
    st.success(f"💰 Predicted House Price: ₹ {int(prediction[0]):,}")
