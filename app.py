import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# PAGE
# -----------------------------
st.title("🏦 Home Loan Prediction")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("cleaned_dataset.csv")

# -----------------------------
# CLEANING
# -----------------------------
df.drop(columns=["Loan_ID"], inplace=True, errors="ignore")

df["Loan_Status"] = df["Loan_Status"].map({"Approved": 1, "Rejected": 0})

# Encode
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Married"] = df["Married"].map({"Yes": 1, "No": 0})
df["Education"] = df["Education"].map({"Graduate": 1, "Not Graduate": 0})
df["Self_Employed"] = df["Self_Employed"].map({"Yes": 1, "No": 0})
df["Property_Area"] = df["Property_Area"].map({"Urban": 2, "Semiurban": 1, "Rural": 0})
df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)

# Fill missing
df.fillna(df.median(numeric_only=True), inplace=True)

# -----------------------------
# FEATURES
# -----------------------------
features = [
    'Gender','Married','Dependents','Education',
    'Self_Employed','ApplicantIncome','CoapplicantIncome',
    'LoanAmount','Loan_Amount_Term',
    'Property_Area','Income_Stability'
]

X = df[features]
y = df["Loan_Status"]

# -----------------------------
# TRAIN MODEL (inside app)
# -----------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier())
])

pipeline.fit(X, y)

st.success("✅ Model Ready")

# -----------------------------
# INPUT FORM
# -----------------------------
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

# Encode input
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]
dependents = 3 if dependents == "3+" else int(dependents)

income_stability = 0 if self_employed == 1 else 1

# Prediction
if st.button("Predict"):
    features = np.array([[gender, married, dependents, education,
                          self_employed, app_income, co_income,
                          loan_amount, loan_term,
                          property_area, income_stability]])

    prediction = pipeline.predict(features)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
