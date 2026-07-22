# Loan Risk Prediction

This project predicts whether a loan application will be approved using Machine Learning.

The project includes data cleaning, feature engineering, model training, hyperparameter tuning, feature importance analysis, and a Streamlit web application for prediction.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- XGBoost
- ScorecardPy
- Matplotlib
- Streamlit
- Joblib

## Models Used

The following machine learning models were trained and compared:

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 76.0% |
| Decision Tree | 90.6% |
| AdaBoost | 96.4% |
| XGBoost | 96.2% |

Hyperparameter tuning was performed using **GridSearchCV** for Decision Tree, AdaBoost, and XGBoost.

## Project Files

- `Loan_risk_datacleaning.py` – Data cleaning and preprocessing
- `Loan_risk_Analysis.py` – Feature engineering, model training and evaluation
- `app.py` – Streamlit application
- `loan_approval_model.pkl` – Trained model
- `loan_risk_cleaned.csv` – Cleaned dataset

## Features

- Data Cleaning
- VIF Analysis
- Weight of Evidence (WoE)
- Information Value (IV)
- Feature Importance
- Model Comparison
- Loan Approval Prediction using Streamlit

## Running the Project

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```
