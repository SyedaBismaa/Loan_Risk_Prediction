# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 12:57:01 2026

@author: syeda
"""

import pandas as pd


df=pd.read_csv(r"C:\Users\syeda\OneDrive\Desktop\Loan_risk\loan_risk_cleaned.csv")

print(df.head(5))
print(df.dtypes)
print(df.describe)