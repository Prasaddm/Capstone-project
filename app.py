import streamlit as st
import joblib
import numpy as np

# Load the saved model

model = joblib.load("rf_battery_life_model.pkl")


# Streamlit UI
st.title("Battery Life Prediction")

# User inputs for the features
temperature = st.number_input("Enter Temperature:")
charge_cycles = st.number_input("Enter Charge Cycles:")
discharge_rate = st.number_input("Enter Discharge Rate:")

# Prediction button
if st.button("Predict"):
    input_data = np.array([[temperature, charge_cycles, discharge_rate]])
    prediction = model.predict(input_data)
    st.write(f"Predicted Battery Life: {prediction[0]:.2f} hours")