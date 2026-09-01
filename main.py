# ==========================================================
# STEP 1: Import Libraries
# ==========================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================================
# STEP 2: Load Dataset
# ==========================================================

df = pd.read_csv("Housing.csv")

# ==========================================================
# STEP 3: Display Basic Information
# ==========================================================

print("="*60)
print("First 5 Rows")
print("="*60)
print(df.head())

print("\n")

print("="*60)
print("Dataset Shape")
print("="*60)
print(df.shape)

print("\n")

print("="*60)
print("Dataset Information")
print("="*60)
print(df.info())

print("\n")

print("="*60)
print("Statistical Summary")
print("="*60)
print(df.describe())

# ==========================================================
# STEP 4: Check Missing Values
# ==========================================================

print("\n")
print("="*60)
print("Missing Values")
print("="*60)
print(df.isnull().sum())

# ==========================================================
# STEP 5: Check Duplicate Rows
# ==========================================================

print("\n")
print("="*60)
print("Duplicate Rows")
print("="*60)
print(df.duplicated().sum())

# ==========================================================
# STEP 6: Data Types
# ==========================================================

print("\n")
print("="*60)
print("Data Types")
print("="*60)
print(df.dtypes)

# ==========================================================
# STEP 7: Convert Categorical Variables
# ==========================================================

df = pd.get_dummies(df, drop_first=True)

print("\n")
print("="*60)
print("After Encoding")
print("="*60)
print(df.head())

# ==========================================================
# STEP 8: Correlation Heatmap
# ==========================================================

plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ==========================================================
# STEP 9: Select Features and Target
# ==========================================================

X = df.drop("price", axis=1)
y = df["price"]

print("\n")
print("="*60)
print("Features Shape")
print("="*60)
print(X.shape)

print("\n")
print("="*60)
print("Target Shape")
print("="*60)
print(y.shape)

# ==========================================================
# STEP 10: Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n")
print("="*60)
print("Training Data Shape")
print("="*60)
print(X_train.shape)

print("\n")
print("="*60)
print("Testing Data Shape")
print("="*60)
print(X_test.shape)

# ==========================================================
# STEP 11: Build Linear Regression Model
# ==========================================================

model = LinearRegression()

# ==========================================================
# STEP 12: Train Model
# ==========================================================

model.fit(X_train, y_train)

print("\n")
print("="*60)
print("Model Training Completed")
print("="*60)

# ==========================================================
# STEP 13: Prediction
# ==========================================================

y_pred = model.predict(X_test)

# ==========================================================
# STEP 14: Evaluation Metrics
# ==========================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n")
print("="*60)
print("Model Evaluation")
print("="*60)

print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# ==========================================================
# STEP 15: Compare Actual vs Predicted
# ==========================================================

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\n")
print("="*60)
print("Actual vs Predicted")
print("="*60)
print(comparison.head(20))

# ==========================================================
# STEP 16: Scatter Plot
# ==========================================================

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")

plt.show()

# ==========================================================
# STEP 17: Residual Plot
# ==========================================================

residuals = y_test - y_pred

plt.figure(figsize=(8,6))

plt.scatter(y_pred, residuals)

plt.axhline(y=0, color='red')

plt.xlabel("Predicted Price")
plt.ylabel("Residuals")

plt.title("Residual Plot")

plt.show()

# ==========================================================
# STEP 18: Distribution of Residuals
# ==========================================================

plt.figure(figsize=(8,6))

sns.histplot(residuals, kde=True)

plt.title("Residual Distribution")

plt.show()

# ==========================================================
# STEP 19: Model Coefficients
# ==========================================================

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

coefficients = coefficients.sort_values(
    by="Coefficient",
    ascending=False
)

print("\n")
print("="*60)
print("Feature Importance (Coefficients)")
print("="*60)
print(coefficients)

# ==========================================================
# STEP 20: Intercept
# ==========================================================

print("\n")
print("="*60)
print("Model Intercept")
print("="*60)

print(model.intercept_)

# ==========================================================
# STEP 21: Predict New House Price
# ==========================================================

new_house = X.iloc[[0]]

prediction = model.predict(new_house)

print("\n")
print("="*60)
print("Prediction for First House")
print("="*60)

print(f"Predicted Price : {prediction[0]:,.2f}")
print(f"Actual Price    : {y.iloc[0]:,.2f}")

print("\n")
print("="*60)
print("Project Completed Successfully")
print("="*60)