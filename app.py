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
from sqlalchemy import create_engine
import datetime

model=joblib.load("loan_approval_model.pkl")
explainer = shap.TreeExplainer(model)

tab1,tab2,tab3=st.tabs(["Tab1","Tab2","Tab3"])

conn = st.secrets["postgres"]
engine = create_engine(
    f"postgresql://{conn['user']}:{conn['password']}@{conn['host']}:{conn['port']}/{conn['dbname']}?sslmode=require"
)


st.divider()

with tab1:
    st.title("Loan Risk Prediction system")
    st.caption("Enter the applicant's details below to predict loan approval")
    
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
        
        input_data["Predictions"] = "Approved" if prediction[0] == 1 else "Rejected"
        input_data["Probability (%)"] = round(approval_probability, 2)
        input_data["Risk"] = risk
        input_data["batch_id"] = "single_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")



        input_data.to_sql("loan_predictions", engine, if_exists="append", index=False)
        st.toast("Saved to database!")
        


with tab2:
    st.title("Batch Loan Prediction system ")
    st.caption("Upload a CSV to predict loan approval for multiple applicants")
    with st.expander("Prerequisites"):
      st.write("""
      1. File must be in CSV format only.
      2. Data must be cleaned, with no missing values.
      3. The CSV must contain the following columns:
         - Age, Income, LoanAmount, CreditScore, YearsExperience
         - Gender_Male , Gender_Female
         - Education_High School, Education_Masters, Education_PhD, Education_Unkown
         - City_Houston, City_New York, City_San Francisco
         - EmploymentType_Self-Employed, EmploymentType_Unemployed
         - Affordability
      4. Column order will be auto-corrected if needed.
      5. Once you click **Predict**, results (Prediction, Probability, Risk) will be shown and automatically saved to the database.
      """)
    
    
    
    uploaded_file= st.file_uploader("Upload a CSV file only!", type="csv")
    
    if uploaded_file is not None:
        df=pd.read_csv(uploaded_file)
        required_cols={
            "Age","Income","LoanAmount","CreditScore","YearsExperience","Gender_Male",
        "Education_High School","Education_Masters","Education_PhD",
        "Education_Unkown","City_Houston","City_New York",
        "City_San Francisco","EmploymentType_Self-Employed",
        "EmploymentType_Unemployed","Affordability"
            }
        required_order=[    "Age","Income","LoanAmount","CreditScore","YearsExperience","Gender_Male",
        "Education_High School","Education_Masters","Education_PhD",
        "Education_Unkown","City_Houston","City_New York",
        "City_San Francisco","EmploymentType_Self-Employed",
        "EmploymentType_Unemployed","Affordability"]
        
        
        if df.empty:
            st.error("The CSV file is empty!")
        elif not required_cols.issubset(df.columns):
            missing=required_cols - set(df.columns)
            st.error(f"Missing Columns: {missing}")
        else:
            if list(df.columns)!=required_order:
                df=df[required_order]
                st.info("Columns Reorderd")
                
        
                    
            if st.button("Predict",key="predict_tab2"):

                X = df
                predictions = model.predict(X)
                probabilities=model.predict_proba(df)[:,1]
                
                
                df["Predictions"]=["Approved" if p==1 else "Rejected" for p in predictions]
                df["Probability (%)"]=( probabilities*100).round(2)
                df["Risk"]=df["Probability (%)"].apply(
                    lambda p:"Low Risk" if p>=80 else("Medium Risk" if p>=50 else "High Risk")
                    )
                
                batch_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                df["batch_id"] = batch_id
                
                st.write(df)
                total=len(df)
                approved=(df["Predictions"]=="Approved").sum()
                rejected= total - approved
                approved_pct = (approved/total)*100
                
                
                col1,col2,col3,col4 =st.columns(4)
                col1.metric("Total Applications", total)
                col2.metric("Approved", approved)
                col3.metric("Rejected", rejected)
                col4.metric("Approval %", f"{approved_pct:.1f}%")
                
                df.to_sql("loan_predictions", engine, if_exists="append", index=False)
                st.toast("Saved to database!")
                
                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Results CSV",
                    data=csv_data,
                    file_name="LoanPrediction_result.csv",
                    mime="text/csv",
                    key="download_csv"
                    )
                



                report= f"""Loan Prediction Report
                -----------------------------------
                total Applications={total}
                Approved : {approved}
                Rejected : {rejected}
                Approval Rate :{approved_pct:.1f}%

                Risk Breakdown:
                 {df['Risk'].value_counts().to_string()}
                                
                """
                st.download_button(
                    "Download Summary Report",
                    data=report,
                    file_name="summary_report.txt",
                    mime="text/plain",
                    key="download_report"
                )
                
        
        
with tab3:
    st.title("Prediction History")
    st.caption("View past predictions and approval trends")
    history = pd.read_sql("""
    SELECT batch_id, 
           COUNT(*) AS total,
           SUM(CASE WHEN "Predictions"='Approved' THEN 1 ELSE 0 END) AS approved,
           ROUND(100.0 * SUM(CASE WHEN "Predictions"='Approved' THEN 1 ELSE 0 END) / COUNT(*), 1) AS approval_rate
    FROM loan_predictions
    GROUP BY batch_id
    ORDER BY batch_id DESC
    """, engine)
    if not history.empty:
        st.write(history)
        st.line_chart(history.sort_values("batch_id").set_index("batch_id")["approval_rate"])
    else:
        st.info("No prediction history yet. ")
        
    



       