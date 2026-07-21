# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 20:01:28 2026

@author: syeda
"""

import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv(r"C:\Users\syeda\OneDrive\Desktop\Loan_risk\loan_risk_prediction_dataset.csv")

print(df.head())
df.isna().sum()
print(df.dtypes)
   

#Null Vlas income-196, creditscore-194, education-198

print(df["Income"].describe())
print(df["CreditScore"].describe())
print(df["Education"].describe())


#neg vals -null vals 
df.loc[df["Income"] < 0, "Income"] = np.nan

#plots 
plt.boxplot(df["Income"].dropna())
plt.title("Income Boxplot")
plt.show()

plt.hist(df["CreditScore"], bins=30)
plt.title("Credit Score Distribution")
plt.show()

plt.boxplot(df["CreditScore"].dropna())
plt.title("Credit Score Boxplot")
plt.show()

#Null Val treatment
Q1 = df["Income"].quantile(0.25)
Q3 = df["Income"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
df["Income"] = df["Income"].clip(lower, upper)

df["Income"]=df["Income"].fillna(df["Income"].mean())
df["CreditScore"] = df["CreditScore"].fillna(df["CreditScore"].median())
df["Education"] = df["Education"].fillna("Unkown")


   
#Gender   Education  City EmploymentType    ----object

print(df["Gender"].unique()) #2 M F
print(df["Education"].unique())#5
print(df["City"].unique()) #4
print(df["EmploymentType"].unique()) #3

df=pd.get_dummies(df,columns=["Gender"],dtype=int)
df=pd.get_dummies(df,columns=["Education"],dtype=int)
df=pd.get_dummies(df,columns=["City"],dtype=int)
df=pd.get_dummies(df,columns=["EmploymentType"],dtype=int)

print(df.dtypes)


#Bucketing 
print(df["YearsExperience"].describe())

df["Age_Group"]=pd.cut(
   df["Age"],
   bins=[18,30,45,60,70],
   labels=["Young","Adult","Middle Age","Senior"],
   include_lowest=True
    )

df=pd.get_dummies(df,columns=["Age_Group"],dtype=int)

df["Income_Group"]=pd.cut(
    df["Income"],
    bins=[0,30000,50000,70000,100000],
    labels=["Low","Medium","High","Very High"]
    )

df=pd.get_dummies(df,columns=["Income_Group"],dtype=int)

df["Credit_Risk"] = pd.cut(
    df["CreditScore"],
    bins=[300,580,670,740,800,850],
    labels=["Poor","Fair","Good","Very Good","Excellent"],
    include_lowest=True
)
df=pd.get_dummies(df,columns=["Credit_Risk"],dtype=int)

#New features
df["Loan_Income_Ratio"] = df["LoanAmount"] / df["Income"]
df["Credit_Experience"] = (
    df["CreditScore"] * df["YearsExperience"]
)

df["Affordability"] = (
    df["Income"] / df["LoanAmount"]
)


print(df.dtypes)


# Save cleaned dataset
df.to_csv(
    r"C:\Users\syeda\OneDrive\Desktop\Loan_risk\loan_risk_cleaned.csv",
    index=False
)

print("Cleaned dataset saved successfully!")