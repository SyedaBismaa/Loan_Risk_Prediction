# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:36:57 2026

@author: syeda
"""
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt

model=joblib.load("loan_approval_model.pkl")
explainer = shap.TreeExplainer(model)

st.title("Loan Risk Prediction system")

st.write("Enter the applicant's details below to predict loan approval")

st.divider()

st.header("Applicant Details")

age=st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=25
)

income=st.number_input("Income", min_value=0.0,value=50000.0)
loan_amount=st.number_input("Loan Amount",min_value=0.0,max_value=1000000.0,value=100000.0)
credit_score=st.number_input("Credit Score",min_value=300.0,max_value=850.0,value=650.0)
years_experience=st.number_input("Years of Experience",min_value=0.0,max_value=50.0,value=5.0)


gender=st.selectbox(
    "Gender",
    options=["Male","Female"]
    )

education = st.selectbox(
    "Education",
    ["High School", "Masters", "PhD", "Unkown"]
)

employment = st.selectbox(
    "Employment Type",
    ["Employed","Self-Employed","Unemployed"]
)

city = st.selectbox(
    "City",
    ["Houston","New York","San Francisco"]
)

if st.button("Predict"):

    affordability = income / loan_amount if loan_amount != 0 else 0
    gender_male = 1 if gender == "Male" else 0
    education_high_school = 1 if education == "High School" else 0
    education_masters = 1 if education == "Masters" else 0
    education_phd = 1 if education == "PhD" else 0
    education_unknown = 1 if education == "Unkown" else 0
    city_houston = 1 if city == "Houston" else 0
    city_new_york = 1 if city == "New York" else 0
    city_san_francisco = 1 if city == "San Francisco" else 0
    employment_self = 1 if employment == "Self-Employed" else 0
    employment_unemployed = 1 if employment == "Unemployed" else 0

    # Create DataFrame
    input_data = pd.DataFrame([{
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "YearsExperience": years_experience,
        "Gender_Male": gender_male,
        "Education_High School": education_high_school,
        "Education_Masters": education_masters,
        "Education_PhD": education_phd,
        "Education_Unkown": education_unknown,
        "City_Houston": city_houston,
        "City_New York": city_new_york,
        "City_San Francisco": city_san_francisco,
        "EmploymentType_Self-Employed": employment_self,
        "EmploymentType_Unemployed": employment_unemployed,
        "Affordability": affordability
    }])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    
    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    approval_probability = probability[0][1] * 100
    
    if approval_probability>=80:
        risk= "Low Risk"
    elif approval_probability >=50:
        risk = "Medium Risk"
    else:
       risk = "High Risk"

    st.write(f"### Approval Probability: {approval_probability:.2f}%")
    st.progress(float(approval_probability) / 100)
    
    st.subheader("Risk Assessment")
    st.info(f"Risk level :{risk}")
    
   
    st.subheader("Why This Prediction?")
    
    shap_values = explainer(input_data)
    
    fig=plt.figure(figsize=(10,6))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
    plt.close(fig)
    




