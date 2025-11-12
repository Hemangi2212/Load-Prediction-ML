# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# 🎯 Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered",
)

# -----------------------------
# 💼 Load Model & Preprocessors
# -----------------------------
with open("loan_approval_model.pkl", "rb") as f:
    model, scaler, encoders = pickle.load(f)

# -----------------------------
# 🏦 App Title & Description
# -----------------------------
st.title("🏦 Loan Approval Prediction App")
st.markdown(
    """
    <div style="background-color:#f0f2f6;padding:15px;border-radius:10px;">
        <h4 style="color:#333;">Fill in the applicant details below to predict loan approval instantly.</h4>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# -----------------------------
# 📋 User Input Section
# -----------------------------
st.header("🔹 Applicant Information")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
with col2:
    applicant_income = st.number_input("Applicant Income", min_value=0, step=100)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, step=100)
    loan_amount = st.number_input("Loan Amount", min_value=0, step=10)
    loan_amount_term = st.selectbox(
        "Loan Amount Term (in months)", [12, 36, 60, 120, 180, 240, 300, 360, 480]
    )
    credit_history = st.selectbox("Credit History", [0, 1])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# -----------------------------
# 🧮 Data Preprocessing
# -----------------------------
input_dict = {
    "Gender": [gender],
    "Married": [married],
    "Dependents": [dependents],
    "Education": [education],
    "Self_Employed": [self_employed],
    "ApplicantIncome": [applicant_income],
    "CoapplicantIncome": [coapplicant_income],
    "LoanAmount": [loan_amount],
    "Loan_Amount_Term": [loan_amount_term],
    "Credit_History": [credit_history],
    "Property_Area": [property_area],
}

df_input = pd.DataFrame(input_dict)

# Apply encoders (must match training)
for col in df_input.columns:
    if col in encoders:
        df_input[col] = encoders[col].transform(df_input[col].astype(str))

# Scale numerical features
df_input_scaled = scaler.transform(df_input)

# -----------------------------
# 🔍 Prediction Section
# -----------------------------
st.markdown("---")
st.header("📊 Prediction Result")

if st.button("🔮 Predict Loan Approval"):
    prediction = model.predict(df_input_scaled)[0]
    if prediction == 1:
        st.success("✅ Loan Approved!")
        st.balloons()
    else:
        st.error("❌ Loan Rejected")

# -----------------------------
# 👨‍💻 Developer Info
# -----------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#555;">
        <h4>👨‍💻 Developed by <b>Hema Ransing</b></h4>
        <p>Email: <a href="mailto:hemaransing@gmail.com">hemaransing@gmail.com</a></p>
        <p>Contact: +91-8767509860</p>
        <p style="font-size:13px;">Made with ❤️ using Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True,
)
