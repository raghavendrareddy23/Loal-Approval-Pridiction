import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and preprocessor
model = joblib.load("model/model.pkl")
preprocessor = joblib.load("model/preprocessor.pkl")

st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰")
st.title("💰 Loan Approval Predictor")

# Input fields
person_age = st.slider("Person Age", 20, 144, 26)
person_gender = st.selectbox("Gender", ['Male', 'Female'])
person_education = st.selectbox("Education Level", ['Master', 'High School', 'Bachelor', 'Associate', 'Doctorate'])
person_income = st.number_input("Monthly Income", min_value=0)

person_emp_exp = st.number_input("Years of Employment Experience", 1, 125, 4)
person_home_ownership = st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])

loan_amnt = st.number_input("Loan Amount", min_value=100, max_value=50000)
loan_intent = st.selectbox("Loan Purpose", ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
loan_int_rate = st.slider("Loan Interest Rate (%)", 5.0, 20.0, 11.0)
loan_percent_income = st.slider("Loan Percent Income", 0.0, 0.66, 0.12)

cb_person_cred_hist_length = st.slider("Credit History Length (years)", 2, 30, 4)
credit_score = st.slider("Credit Score", 390, 850, 640)
previous_loan_defaults_on_file = st.selectbox("Default on Previous Loan?", ['Yes', 'No'])

if st.button("Predict Loan Approval"):
    # ✅ Construct input as DataFrame (no manual encoding)
    input_data = {
        'person_age': [person_age],
        'person_gender': [person_gender],
        'person_education': [person_education],
        'person_income': [person_income],
        'person_emp_exp': [person_emp_exp],
        'person_home_ownership': [person_home_ownership],
        'loan_amnt': [loan_amnt],
        'loan_intent': [loan_intent],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length],
        'credit_score': [credit_score],
        'previous_loan_defaults_on_file': [previous_loan_defaults_on_file],
    }

    input_df = pd.DataFrame(input_data)

    # ✅ Preprocess input
    try:
        X_processed = preprocessor.transform(input_df)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

    # ✅ Get probability prediction
    prob = model.predict_proba(X_processed)[0][1]  # Probability of class 1 (Approved)

    # 🎯 Custom threshold
    threshold = 0.3
    # st.write(f"🔎 Model confidence: {prob:.2f}")

    if prob >= threshold:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Not Approved")
