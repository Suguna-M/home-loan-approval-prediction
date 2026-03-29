import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Home Loan Prediction", layout="wide")
st.title("🏦 Home Loan Approval System")

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

# Income Stability (important)
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

st.success("✅ Model Ready")

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

    # Encoding
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0
    property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]
    dependents = 3 if dependents == "3+" else int(dependents)

    income_stability = 0 if self_employed == 1 else 1

    if st.button("Predict Loan Status"):

        features_input = np.array([[gender, married, dependents, education,
                                    self_employed, app_income, co_income,
                                    loan_amount, loan_term,
                                    property_area, income_stability]])

        prediction = pipeline.predict(features_input)

        if prediction[0] == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

# ============================================================
# 🧑‍💼 LOAN OFFICER
# ============================================================
elif role == "Loan Officer":

    st.header("🧑‍💼 Loan Officer Dashboard")

    df_dash = df.copy()

    # Fix Credit History
    df_dash["Credit_History"] = df_dash["Credit_History"].apply(
        lambda x: 1 if x >= 0.5 else 0
    )

    # -----------------------------
    # METRIC
    # -----------------------------
    approval_rate = df_dash["Loan_Status"].mean() * 100
    st.metric("Approval Rate", f"{approval_rate:.2f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Risk Score Distribution")
        st.bar_chart(df_dash["Risk_Score"])

    with col2:
        st.subheader("🚨 Fraud vs Approval")
        st.bar_chart(pd.crosstab(df_dash["Fraud_Flag"], df_dash["Loan_Status"]))

    st.subheader("🏘️ Property Area vs Approval")
    st.bar_chart(pd.crosstab(df_dash["Property_Area"], df_dash["Loan_Status"]))

    st.subheader("📊 Income vs Loan Amount")
    st.scatter_chart(df_dash[["Total_Income", "LoanAmount"]])

    st.subheader("📌 Insights")
    st.write("✔ Good credit history increases approval chances")
    st.write("✔ High risk score reduces approval probability")
    st.write("✔ Fraud flag strongly impacts rejection")

# ============================================================
# ⚙️ ADMIN
# ============================================================
elif role == "Admin":

    st.header("⚙️ Admin Dashboard")

    st.subheader("📊 System Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Applications", len(df))
    col2.metric("Fraud Cases", df["Fraud_Flag"].sum())
    col3.metric("Approval Rate", f"{df['Loan_Status'].mean()*100:.2f}%")

    st.subheader("📊 Loan Status Distribution")
    st.bar_chart(df["Loan_Status"].value_counts())

    st.subheader("📊 Property Area Distribution")
    st.bar_chart(df["Property_Area"].value_counts())

    st.subheader("📊 Loan Amount Distribution")
    st.bar_chart(df["LoanAmount"])

    st.subheader("📊 Risk Score Distribution")
    st.bar_chart(df["Risk_Score"])

    st.subheader("📈 Income Trend (Sample)")
    st.line_chart(df["ApplicantIncome"].head(100))

    st.subheader("📌 Insights")
    st.write("✔ Majority applications come from Urban areas")
    st.write("✔ Fraud cases are low but impactful")
    st.write("✔ Income directly affects loan approval")
