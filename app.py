import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
pipeline = pickle.load(open("pipeline.pkl", "rb"))

st.title("🏦 Home Loan Prediction")

# ===============================
# INPUTS
# ===============================
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

app_income = st.number_input("Applicant Income")
co_income = st.number_input("Coapplicant Income")
loan_amount = st.number_input("Loan Amount")
loan_term = st.number_input("Loan Term")
credit_history = st.selectbox("Credit History", [1.0, 0.0])
fraud_flag = st.selectbox("Fraud Flag", [0, 1])

# ===============================
# CREATE INPUT DATAFRAME
# ===============================
input_df = pd.DataFrame([{
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": app_income,
    "CoapplicantIncome": co_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history,
    "Property_Area": property_area,
    "Fraud_Flag": fraud_flag
}])

# ===============================
# PREDICTION
# ===============================
if st.button("Predict"):
    prediction = pipeline.predict(input_df)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
