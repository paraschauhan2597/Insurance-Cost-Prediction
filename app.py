import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set the page configuration for a professional look
st.set_page_config(page_title="Premium Predictor", page_icon="🏥", layout="centered")

# Load the saved model and scaler from the deployment folder
# (Using caching so it only loads once and runs faster)
@st.cache_resource
def load_models():
    rf = joblib.load('deployment/rf_model.pkl')
    sc = joblib.load('deployment/scaler.pkl')
    return rf, sc

model, scaler = load_models()

# App Header
st.title("🏥 Health Insurance Premium Predictor")
st.markdown("Enter the customer's demographic and health details to generate a real-time premium estimate.")
st.markdown("---")

# Create two columns for a neat input layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Details")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=150, value=70)
    surgeries = st.selectbox("Number of Major Surgeries", [0, 1, 2, 3])

with col2:
    st.subheader("Health Profile")
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    bp_problems = st.selectbox("Blood Pressure Problems", ["No", "Yes"])
    transplants = st.selectbox("Any Transplants", ["No", "Yes"])
    chronic_diseases = st.selectbox("Any Chronic Diseases", ["No", "Yes"])
    allergies = st.selectbox("Known Allergies", ["No", "Yes"])
    cancer_history = st.selectbox("History of Cancer in Family", ["No", "Yes"])

st.markdown("---")

# Prediction Button
if st.button("Calculate Premium Estimate", type="primary"):
    
    # 1. Convert "Yes"/"No" inputs into 1 and 0 for the model
    input_data = {
        'Age': age,
        'Diabetes': 1 if diabetes == "Yes" else 0,
        'BloodPressureProblems': 1 if bp_problems == "Yes" else 0,
        'AnyTransplants': 1 if transplants == "Yes" else 0,
        'AnyChronicDiseases': 1 if chronic_diseases == "Yes" else 0,
        'Height': height,
        'Weight': weight,
        'KnownAllergies': 1 if allergies == "Yes" else 0,
        'HistoryOfCancerInFamily': 1 if cancer_history == "Yes" else 0,
        'NumberOfMajorSurgeries': surgeries
    }
    
    # 2. Create a DataFrame (Ensuring columns match exactly how they were trained)
    input_df = pd.DataFrame([input_data])
    
    # 3. Scale the features using the saved scaler
    input_scaled = scaler.transform(input_df)
    
    # 4. Make the prediction
    prediction = model.predict(input_scaled)[0]
    
    # 5. Display the result beautifully
    st.success("Analysis Complete!")
    st.metric(label="Estimated Annual Premium", value=f"₹ {prediction:,.2f}")
    