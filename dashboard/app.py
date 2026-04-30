"""Streamlit dashboard for churn prediction."""

import json

import requests
import streamlit as st

API_URL = "http://api:8000"

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Predict whether a telecom customer is likely to churn.")

# Sidebar for inputs
st.sidebar.header("Customer Features")

# Demographics
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox(
    "Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
)
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

# Services
tenure = st.sidebar.slider("Tenure (months)", 0, 100, 12)
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox(
    "Multiple Lines", ["Yes", "No", "No phone service"]
)
internet_service = st.sidebar.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)
online_security = st.sidebar.selectbox(
    "Online Security", ["Yes", "No", "No internet service"]
)
online_backup = st.sidebar.selectbox(
    "Online Backup", ["Yes", "No", "No internet service"]
)
device_protection = st.sidebar.selectbox(
    "Device Protection", ["Yes", "No", "No internet service"]
)
tech_support = st.sidebar.selectbox(
    "Tech Support", ["Yes", "No", "No internet service"]
)
streaming_tv = st.sidebar.selectbox(
    "Streaming TV", ["Yes", "No", "No internet service"]
)
streaming_movies = st.sidebar.selectbox(
    "Streaming Movies", ["Yes", "No", "No internet service"]
)

# Billing
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 0.0, 150.0, 70.0)
total_charges = st.sidebar.slider("Total Charges ($)", 0.0, 9000.0, 1000.0)

# Prediction button
if st.sidebar.button("Predict Churn", type="primary"):
    payload = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        result = response.json()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Churn Prediction", result["churn_label"])
        with col2:
            st.metric("Churn Probability", f"{result['churn_probability']:.2%}")
        with col3:
            risk = (
                "High"
                if result["churn_probability"] > 0.5
                else "Medium" if result["churn_probability"] > 0.3 else "Low"
            )
            st.metric("Risk Level", risk)

        # Visualization
        st.subheader("Prediction Visualization")
        prob = result["churn_probability"]
        st.progress(prob)

    except Exception as e:
        st.error(f"Error making prediction: {e}")

# Health check
st.sidebar.markdown("---")
if st.sidebar.button("Check API Health"):
    try:
        response = requests.get(f"{API_URL}/health")
        if response.json()["status"] == "healthy":
            st.sidebar.success("API is healthy")
        else:
            st.sidebar.warning("API is not healthy")
    except Exception:
        st.sidebar.error("Cannot connect to API")
