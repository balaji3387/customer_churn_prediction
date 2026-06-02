import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page configuration
st.set_page_config(page_title="Telco Customer Churn Predictor", layout="centered")

st.title("📊 Telco Customer Churn Prediction App")
st.write("Enter the customer's details below to predict the likelihood of churn.")

# 1. Load the trained RandomForest model and LabelEncoders
@st.cache_resource
def load_assets():
    with open("customer_churn_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("encoders.pkl", "rb") as encoder_file:
        encoders = pickle.load(encoder_file)
    return model, encoders

try:
    model, encoders = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

# 2. Replicate the feature schema expected by your model
# Features: gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, 
# InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, 
# StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges

st.header("Customer Profile Information")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen (Age >= 65)", [0, 1])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)

with col2:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

st.header("Subscribed Security & Media Services")
col3, col4 = st.columns(2)

with col3:
    online_security = st.selectbox("Online Security Service", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup Service", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection Service", ["No", "Yes", "No internet service"])

with col4:
    tech_support = st.selectbox("Tech Support Service", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

st.header("Financial Metrics")
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)

# 3. Process inputs when user clicks Predict
if st.button("🔮 Predict Churn Status", type="primary"):
    # Construct a dataframe matching your exact training schema rows
    input_data = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": senior_citizen, "Partner": partner, 
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service, 
        "MultipleLines": multiple_lines, "InternetService": internet_service, 
        "OnlineSecurity": online_security, "OnlineBackup": online_backup, 
        "DeviceProtection": device_protection, "TechSupport": tech_support, 
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies, 
        "Contract": contract, "PaperlessBilling": paperless_billing, 
        "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges, 
        "TotalCharges": total_charges
    }])
    
    # Apply your saved Label Encoders to categorical input columns
    for col in input_data.columns:
        if col in encoders and col != "SeniorCitizen":
            try:
                input_data[col] = encoders[col].transform(input_data[col])
            except Exception as e:
                # Fallback safeguard in case of unexpected unseen labels
                input_data[col] = 0

    # Execute Prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    if prediction == 1 or prediction == "Yes":
        st.error(f"⚠️ **High Churn Risk:** The customer is likely to cancel service. (Churn Probability: {prediction_proba:.2%})")
    else:
        st.success(f"✅ **Low Churn Risk:** The customer is likely to stay retained. (Churn Probability: {prediction_proba:.2%})")