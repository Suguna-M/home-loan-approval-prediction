import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from database import create_table, insert_application, get_all_data

# -----------------------------
# SIMPLE LOGIN SYSTEM
# -----------------------------
users = {
    "admin": {"password": "admin123", "role": "Admin"},
    "officer": {"password": "officer123", "role": "Loan Officer"},
    "user": {"password": "user123", "role": "Applicant"}
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.role = users[username]["role"]
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Home Loan System", layout="wide")
st.title("🏦 Home Loan Approval Prediction System")

# 🔹 THEME TOGGLE (ADDED)
theme = st.toggle("🌙 Dark Mode")

if theme:
    st.markdown(
        """
        <style>
        body {background-color: #0E1117; color: white;}
        </style>
        """,
        unsafe_allow_html=True
    )

create_table()

# -----------------------------
# ROLE SELECTOR
# -----------------------------
role = st.session_state.role
st.sidebar.success(f"Logged in as: {role}")

# 🔹 LOGOUT BUTTON (ADDED)
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("cleaned_dataset.csv")

# -----------------------------
# DATA CLEANING
# -----------------------------
df.drop(columns=["Loan_ID"], inplace=True, errors="ignore")

df["Loan_Status"] = df["Loan_Status"].map({"Approved": 1, "Rejected": 0})

df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Married"] = df["Married"].map({"Yes": 1, "No": 0})
df["Education"] = df["Education"].map({"Graduate": 1, "Not Graduate": 0})
df["Self_Employed"] = df["Self_Employed"].map({"Yes": 1, "No": 0})
df["Property_Area"] = df["Property_Area"].map({"Urban": 2, "Semiurban": 1, "Rural": 0})
df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)

# ✅ FIX CREDIT HISTORY
df["Credit_History"] = df["Credit_History"].apply(lambda x: 1 if x >= 0.5 else 0)

# Derived Feature
df["Income_Stability"] = df["Self_Employed"].apply(lambda x: 0 if x == 1 else 1)

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
# MODEL
# -----------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier())
])

pipeline.fit(X, y)

# ============================================================
# 👤 APPLICANT
# ============================================================
if role == "Applicant":

    st.header("📝 Loan Application Form")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])

    with col2:
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    app_income = st.number_input("Applicant Income")
    co_income = st.number_input("Coapplicant Income")
    loan_amount = st.number_input("Loan Amount")
    loan_term = st.number_input("Loan Term")

    # ✅ FIX CREDIT HISTORY UI
    credit_history = st.selectbox("Credit History", ["Good", "Bad"])
    credit_history = 1 if credit_history == "Good" else 0

    # Encoding
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0
    property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]
    dependents = 3 if dependents == "3+" else int(dependents)

    total_income = app_income + co_income
    dti = loan_amount / (total_income + 1)
    emi = loan_amount / (loan_term + 1)

    income_stability = 0 if self_employed == 1 else 1

    # ✅ AUTO FRAUD DETECTION
    fraud_flag = 0
    if app_income > 15000 and loan_amount < 50:
        fraud_flag = 1
    if credit_history == 0 and loan_amount > 300:
        fraud_flag = 1

    if st.button("Check Eligibility & Predict"):

        if total_income < 2000:
            st.error("❌ Rejected: Income too low")
        else:

            features_input = np.array([[gender, married, dependents, education,
                                        self_employed, app_income, co_income,
                                        loan_amount, loan_term,
                                        property_area, income_stability]])

            prediction = pipeline.predict(features_input)

            risk_score = (1 - credit_history)*0.5 + dti*0.3 + fraud_flag*0.2

            if risk_score < 0.02:
                risk_level = "Low"
            elif risk_score < 0.05:
                risk_level = "Medium"
            else:
                risk_level = "High"

            st.subheader("📊 Result")

            if prediction[0] == 1:
                st.success("✅ Loan Approved")
            else:
                st.error("❌ Loan Rejected")

            st.write(f"⚠️ Risk Level: {risk_level}")

            if fraud_flag == 1:
                st.error("🚨 Potential Fraud Detected")

            # ✅ DOWNLOAD REPORT (ADDED)
            report = f"""
Loan Prediction Report

Prediction: {"Approved" if prediction[0]==1 else "Rejected"}
Risk Level: {risk_level}

Applicant Income: {app_income}
Loan Amount: {loan_amount}
Credit History: {credit_history}
"""

            st.download_button(
                label="📄 Download Report",
                data=report,
                file_name="loan_report.txt",
                mime="text/plain"
            )

            # ✅ SAVE TO DATABASE
            insert_application((
                gender, married, dependents, education, self_employed,
                app_income, co_income,
                loan_amount, loan_term,
                property_area, income_stability,
                credit_history,
                int(prediction[0]), risk_level, fraud_flag
            ))

# ============================================================
# 🧑‍💼 LOAN OFFICER (UNCHANGED)
# ============================================================
elif role == "Loan Officer":

    st.header("🧑‍💼 Loan Officer Dashboard")

    st.metric("Approval Rate", f"{df['Loan_Status'].mean()*100:.2f}%")

    st.bar_chart(df["Risk_Score"])
    st.bar_chart(pd.crosstab(df["Fraud_Flag"], df["Loan_Status"]))
    st.bar_chart(pd.crosstab(df["Property_Area"], df["Loan_Status"]))

# ============================================================
# ⚙️ ADMIN (UNCHANGED + DB VIEW)
# ============================================================
elif role == "Admin":

    st.header("⚙️ Admin Dashboard")

    st.metric("Total Applications", len(df))
    st.metric("Fraud Cases", df["Fraud_Flag"].sum())
    st.metric("Approval Rate", f"{df['Loan_Status'].mean()*100:.2f}%")

    st.bar_chart(df["Loan_Status"].value_counts())
    st.bar_chart(df["Risk_Score"])
    st.line_chart(df["ApplicantIncome"].head(100))

    st.subheader("📄 Stored Applications")

    data = get_all_data()

    if data:
        st.dataframe(pd.DataFrame(data))
    else:
        st.write("No records yet")
