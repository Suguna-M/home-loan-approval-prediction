import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Home Loan System", layout="wide")
st.title("🏦 Home Loan Approval Prediction System")

# -----------------------------
# ROLE SELECTOR
# -----------------------------
role = st.sidebar.selectbox("Select Role", ["Applicant", "Loan Officer", "Admin"])

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

# Derived Feature
df["Income_Stability"] = df["Self_Employed"].apply(lambda x: 0 if x == 1 else 1)

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
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    fraud_flag = st.selectbox("Fraud Flag", [0, 1])

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

    # -----------------------------
    # ELIGIBILITY RULES
    # -----------------------------
    if st.button("Check Eligibility & Predict"):

        if total_income < 2000:
            st.error("❌ Rejected: Income too low")
        elif credit_history == 0:
            st.warning("⚠️ Poor Credit History → High Risk")

        else:
            features_input = np.array([[gender, married, dependents, education,
                                        self_employed, app_income, co_income,
                                        loan_amount, loan_term,
                                        property_area, income_stability]])

            prediction = pipeline.predict(features_input)

            # -----------------------------
            # RISK SCORE
            # -----------------------------
            risk_score = (1 - credit_history)*0.5 + dti*0.3 + fraud_flag*0.2

            # Risk Level
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

            # -----------------------------
            # FRAUD DETECTION
            # -----------------------------
            if fraud_flag == 1 or (app_income > 15000 and loan_amount < 50):
                st.error("🚨 Potential Fraud Detected")

            # -----------------------------
            # LOAN RECOMMENDATION
            # -----------------------------
            st.subheader("💡 Recommendation")

            if risk_level == "Low":
                st.write("✔ Long-term loan with low EMI")
            elif risk_level == "Medium":
                st.write("✔ Moderate term loan recommended")
            else:
                st.write("✔ Short-term loan / manual review required")

# ============================================================
# 🧑‍💼 LOAN OFFICER
# ============================================================
elif role == "Loan Officer":

    st.header("🧑‍💼 Loan Officer Dashboard")

    df_dash = df.copy()

    df_dash["Credit_History"] = df_dash["Credit_History"].apply(lambda x: 1 if x >= 0.5 else 0)

    st.metric("Approval Rate", f"{df_dash['Loan_Status'].mean()*100:.2f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Score Distribution")
        st.bar_chart(df_dash["Risk_Score"])

    with col2:
        st.subheader("Fraud vs Approval")
        st.bar_chart(pd.crosstab(df_dash["Fraud_Flag"], df_dash["Loan_Status"]))

    st.subheader("Property Area vs Approval")
    st.bar_chart(pd.crosstab(df_dash["Property_Area"], df_dash["Loan_Status"]))

    st.subheader("Insights")
    st.write("✔ High risk → manual review")
    st.write("✔ Fraud cases should be investigated")

# ============================================================
# ⚙️ ADMIN
# ============================================================
elif role == "Admin":

    st.header("⚙️ Admin Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Applications", len(df))
    col2.metric("Fraud Cases", df["Fraud_Flag"].sum())
    col3.metric("Approval Rate", f"{df['Loan_Status'].mean()*100:.2f}%")

    st.subheader("Loan Status Distribution")
    st.bar_chart(df["Loan_Status"].value_counts())

    st.subheader("Risk Score Distribution")
    st.bar_chart(df["Risk_Score"])

    st.subheader("Income Trend")
    st.line_chart(df["ApplicantIncome"].head(100))

    st.subheader("Insights")
    st.write("✔ System working efficiently")
    st.write("✔ Monitor fraud regularly")
