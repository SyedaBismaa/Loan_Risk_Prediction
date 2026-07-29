#  Loan Risk Prediction System

An end-to-end Machine Learning project that predicts whether a loan application should be approved based on an applicant's financial and demographic information.

The project covers the complete ML pipeline—from data preprocessing and feature engineering to model comparison, hyperparameter tuning, explainable AI (SHAP), and deployment using Streamlit.

---

##  Features

- Data Cleaning & Preprocessing
- Feature Engineering
- VIF Analysis
- Weight of Evidence (WoE)
- Information Value (IV)
- Model Training & Comparison
- Hyperparameter Tuning using GridSearchCV
- Feature Importance Analysis
- Loan Approval Prediction
- Approval Probability Score
- Risk Assessment (Low / Medium / High)
- SHAP Explainable AI for prediction interpretation
- Interactive Streamlit Web Application

---

##  Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- ScorecardPy
- Matplotlib
- SHAP
- Streamlit
- Joblib

---

##  Models Trained

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 76.0% |
| Decision Tree | 90.6% |
| AdaBoost | 96.4% |
| XGBoost | 96.2% |

The final deployed model is **XGBoost**, selected based on its superior predictive performance.

---

##  Streamlit Application

The web application allows users to:

- Enter applicant details
- Predict loan approval
- View approval probability
- View applicant risk level
- Understand the prediction using SHAP Explainability

---

##  Application Preview

<img width="607" height="673" alt="image" src="https://github.com/user-attachments/assets/c9671be4-a87c-49b4-8215-8bf14777e755" />


---


