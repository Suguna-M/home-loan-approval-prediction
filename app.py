import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from database import create_table, insert_application, get_all_data

# -----------------------------
# LOAD MODEL
# -----------------------------
pipeline = pickle.load(open("pipeline.pkl", "rb"))

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Home Loan Prediction", layout="wide")
st.title("🏦 Home Loan Approval Prediction System")

create_table()

# -----------------------------
# ROLE SELECTION
# -----------------------------
role = st.sidebar.selectbox("Select Role", ["Applicant", "Loan Officer", "Admin"])

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("cleaned_dataset.csv")

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

    applicant_income = st.number_input("Applicant Income", min_value=0.0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0)
    loan_term = st.number_input("Loan Term", min_value=1.0)

    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    fraud_flag = st.selectbox("Fraud Flag", [0, 1])

    # Encoding
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0
    property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]
    dependents = 3 if dependents == "3+" else int(dependents)

    # Feature Engineering
    total_income = applicant_income + coapplicant_income
    dti = loan_amount / (total_income + 1)
    emi = loan_amount / (loan_term + 1)
    income_stability = 0 if self_employed == 1 else 1

    risk_score = (
        (1 - credit_history) * 0.5 +
        dti * 0.3 +
        fraud_flag * 0.2
    )

    features = np.array([[gender, married, dependents, education,
                          self_employed, applicant_income, coapplicant_income,
                          loan_amount, loan_term, credit_history,
                          property_area, fraud_flag,
                          total_income, dti, emi,
                          income_stability, risk_score]])

    if st.button("Predict Loan Status"):

        prediction = pipeline.predict(features)

        st.subheader("📊 Result")

        if prediction[0] == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

        # Risk display
        if risk_score < 0.02:
            risk_level = "Low"
            st.info("🟢 Low Risk Applicant")
        elif risk_score < 0.05:
            risk_level = "Medium"
            st.warning("⚠️ Medium Risk Applicant")
        else:
            risk_level = "High"
            st.error("🔴 High Risk Applicant")

        if fraud_flag == 1:
            st.error("🚨 Potential Fraud Detected")

        # ✅ SAVE TO DATABASE (NEW)
        insert_application((
            gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income,
            loan_amount, loan_term,
            property_area, income_stability,
            int(prediction[0]), risk_level, fraud_flag
        ))

# ============================================================
# 🧑‍💼 LOAN OFFICER
# ============================================================
elif role == "Loan Officer":

    st.header("🧑‍💼 Loan Officer Dashboard")

    st.subheader("📊 Risk Distribution")
    fig1, ax1 = plt.subplots()
    df['Risk_Score'].hist(bins=30, ax=ax1)
    st.pyplot(fig1)

    st.subheader("📊 Loan Status vs Fraud")
    fig2, ax2 = plt.subplots()
    pd.crosstab(df['Loan_Status'], df['Fraud_Flag']).plot(kind='bar', ax=ax2)
    st.pyplot(fig2)

    st.subheader("📊 Credit History vs Approval")
    fig3, ax3 = plt.subplots()
    pd.crosstab(df['Credit_History'], df['Loan_Status']).plot(kind='bar', ax=ax3)
    st.pyplot(fig3)

# ============================================================
# ⚙️ ADMIN
# ============================================================
elif role == "Admin":

    st.header("⚙️ Admin Dashboard")

    st.subheader("📊 Statistics")
    st.write("Total Applications:", len(df))
    st.write("Fraud Cases:", df['Fraud_Flag'].sum())

    st.subheader("📊 Loan Status")
    st.bar_chart(df['Loan_Status'].value_counts())

    st.subheader("🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include='number').corr(),
                annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # ✅ DATABASE VIEW (NEW)
    st.subheader("📄 Stored Applications")

    data = get_all_data()

    if data:
        df_db = pd.DataFrame(data, columns=[
            "ID","Gender","Married","Dependents","Education",
            "SelfEmployed","AppIncome","CoIncome","LoanAmount",
            "LoanTerm","PropertyArea","IncomeStability",
            "Prediction","RiskLevel","FraudFlag"
        ])
        st.dataframe(df_db)
    else:
        st.write("No records yet")
