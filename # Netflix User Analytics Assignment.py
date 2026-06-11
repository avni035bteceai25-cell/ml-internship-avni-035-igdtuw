# Netflix User Analytics Assignment
# Works in VS Code with Python 3.x

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------------
# Part A: Dataset Understanding
# -------------------------------

# Load dataset (update path if needed)
df = pd.read_csv("Dataset 2 (2).csv")

# Q1: Display first five records
print("First five records:\n", df.head())

# Q2: Number of rows and columns
print("\nShape of dataset:", df.shape)

# Q3: Column names
print("\nColumn names:", df.columns.tolist())

# Q4: Numerical vs Categorical features
num_features = df.select_dtypes(include=[np.number]).columns.tolist()
cat_features = df.select_dtypes(exclude=[np.number]).columns.tolist()
print("\nNumerical features:", num_features)
print("Categorical features:", cat_features)

# Q5: Missing values
print("\nMissing values:\n", df.isnull().sum())

# -------------------------------
# Part B: Exploratory Data Analysis
# -------------------------------

print("\nAverage Age:", df["Age"].mean())
print("Average Watch Hours per Week:", df["WatchHoursPerWeek"].mean())
print("Average Monthly Spending:", df["MonthlySpend"].mean())

print("\nUsers per Subscription Type:\n", df["SubscriptionType"].value_counts())

renewed_pct = df["SubscriptionRenewed"].value_counts(normalize=True) * 100
print("\nSubscription Renewal Percentage:\n", renewed_pct)

# -------------------------------
# Part C: Data Preparation
# -------------------------------

# Encode categorical features
le = LabelEncoder()
for col in cat_features:
    df[col] = le.fit_transform(df[col])

# Define feature set (X) and target (y)
X = df.drop("SubscriptionRenewed", axis=1)
y = df["SubscriptionRenewed"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Part D: Decision Tree
# -------------------------------

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix (Decision Tree):\n", confusion_matrix(y_test, y_pred_dt))

# -------------------------------
# Part E: KNN
# -------------------------------

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

print("\nKNN Accuracy:", accuracy_score(y_test, y_pred_knn))

# -------------------------------
# Part F: Linear Regression
# -------------------------------

# Target: MonthlySpend
X_lr = df.drop("MonthlySpend", axis=1)
y_lr = df["MonthlySpend"]

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train_lr, y_train_lr)

# Predict for a new user (example values)
new_user = pd.DataFrame({
    "UserID": [2001],
    "Age": [30],
    "Gender": [1],  # encoded
    "SubscriptionType": [2],  # encoded
    "WatchHoursPerWeek": [15],
    "DevicesUsed": [2],
    "FavoriteGenre": [3],  # encoded
    "AdClicks": [20],
    "SubscriptionRenewed": [1]  # encoded
})

predicted_spend = lr_model.predict(new_user.drop("UserID", axis=1))
print("\nPredicted Monthly Spend for new user:", predicted_spend[0])
