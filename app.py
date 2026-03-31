import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from database import create_table, insert_application, get_all_data

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Home Loan System", layout="wide")
st.title("🏦 Home Loan Approval Prediction System")

create_table()

# -----------------------------
# ROLE SELECTOR
# -----------------------------
role = st.sidebar.selectbox("Select Role", ["Applicant", "Loan Officer", "Admin"])

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("cleaned_dataset.csv")

# -----------------------------
# CLEANING
# -----------------------------
df.drop(columns=["Loan_ID"], inplace=True, errors="ignore")

df["Loan_Status"] = df["Loan_Status"].map({"Approved": 1, "Rejected": 0})

df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Married"] = df["Married"].map({"Yes": 1, "No": 0})
df["Education"] = df["Education"].map({"Graduate": 1, "Not Graduate": 0})
df["Self_Employed"] = df["Self_Employed"].map({"Yes": 1, "No": 0})
df["Property_Area"] = df["Property_Area"].map({"Urban": 2, "Semiurban": 1, "Rural": 0})
df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)

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

    app_income = st.number_input("Applicant Income", min_value=0.0)
    co_income = st.number_input("Coapplicant Income", min_value=0.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0)
    loan_term = st.number_input("Loan Term", min_value=1.0)

    credit_history = st.selectbox("Good Credit History?", ["Yes", "No"])
    credit_history = 1 if credit_history == "Yes" else 0

    # Encoding
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0
    property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]
    dependents = 3 if dependents == "3+" else int(dependents)

    total_income = app_income + co_income
    dti = loan_amount / (total_income + 1)

    income_stability = 0 if self_employed == 1 else 1

    # Fraud detection
    fraud_flag = 0
    if app_income > 20000 and loan_amount < 50:
        fraud_flag = 1
    if credit_history == 0 and loan_amount > 300:
        fraud_flag = 1

    if st.button("Predict"):

        if total_income < 2000:
            st.error("❌ Rejected: Income too low")
        else:
            input_data = np.array([[gender, married, dependents, education,
                                    self_employed, app_income, co_income,
                                    loan_amount, loan_term,
                                    property_area, income_stability]])

            prediction = pipeline.predict(input_data)

            risk_score = (1-credit_history)*0.5 + dti*0.3 + fraud_flag*0.2

            if risk_score < 0.02:
                risk_level = "Low"
            elif risk_score < 0.05:
                risk_level = "Medium"
            else:
                risk_level = "High"

            if prediction[0] == 1:
                st.success("✅ Loan Approved")
            else:
                st.error("❌ Loan Rejected")

            st.write("Risk Level:", risk_level)

            if fraud_flag == 1:
                st.error("🚨 Fraud Detected")

            # Save to DB
            insert_application((
                gender, married, dependents, education, self_employed,
                app_income, co_income, loan_amount, loan_term,
                property_area, income_stability,
                int(prediction[0]), risk_level, fraud_flag
            ))

# ============================================================
# 🧑‍💼 LOAN OFFICER
# ============================================================
elif role == "Loan Officer":

    st.header("🧑‍💼 Loan Officer Dashboard")

    st.bar_chart(df["Loan_Status"].value_counts())
    st.bar_chart(df["Fraud_Flag"].value_counts())

# ============================================================
# ⚙️ ADMIN
# ============================================================
elif role == "Admin":

    st.header("⚙️ Admin Dashboard")

    st.metric("Total Applications", len(df))
    st.metric("Fraud Cases", df["Fraud_Flag"].sum())

    st.subheader("Stored Applications")

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
