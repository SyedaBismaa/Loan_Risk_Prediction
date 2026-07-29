# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 12:57:01 2026

@author: syeda
"""

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scorecardpy  as sc
import matplotlib.pyplot as plt

#Models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import(
    accuracy_score,
    confusion_matrix,
    classification_report
    )


df=pd.read_csv(r"C:\Users\syeda\OneDrive\Documents\Loan_risk\loan_risk_cleaned.csv")

print(df.head(5))
print(df.dtypes)
print(df.describe)
print(df.columns)
df = df.drop(columns=["Credit_Experience","Loan_Income_Ratio"], axis=1)
X=df.drop(columns=["LoanApproved"])

vif=pd.DataFrame()

vif["Feature"]=X.columns

vif["VIF"]=[
    variance_inflation_factor(X.values,i)
    for i in range(X.shape[1])
    ]

print(vif.sort_values(by="VIF",ascending=False))




#woe
bins = sc.woebin(df, y="LoanApproved")
print(bins["Age"])

iv = sc.iv(df, y="LoanApproved")
print(iv.sort_values(by="info_value", ascending=False))

y=df["LoanApproved"]

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
    )

model = LogisticRegression(
    random_state=42
    )
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#Decision Tree

model = DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced"
)

param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 5, 7, 10],
    "min_samples_split": [20, 50, 100],
    "min_samples_leaf": [10, 20, 50],
    "max_features": ["sqrt", "log2"]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,                  # Faster than 5-fold
    scoring="f1",
    n_jobs=-1,             # Use all CPU cores
    verbose=2
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV F1:", grid_search.best_score_)

best_tree = grid_search.best_estimator_

# Prediction
y_pred = best_tree.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plot Tree
plt.figure(figsize=(18,10))

plot_tree(
    best_tree,
    feature_names=X.columns,
    class_names=["Not Fraud", "Fraud"],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=9
)

plt.show()


#adaboosting 


model = AdaBoostClassifier(random_state=42)

param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 1.0]
}

grid = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print(grid.best_params_)

y_pred = grid.best_estimator_.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))




#boosting 

model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 5, 7]
}

grid = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print(grid.best_params_)

y_pred = grid.best_estimator_.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



import joblib

best_model = grid.best_estimator_

joblib.dump(best_model, "loan_approval_model.pkl")

print("Model saved successfully!")


model = joblib.load("loan_approval_model.pkl")

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)

plt.figure(figsize=(12,6))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.xticks(rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance - AdaBoost")

plt.tight_layout()
plt.show()




