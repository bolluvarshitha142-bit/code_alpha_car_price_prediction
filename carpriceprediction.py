# ==========================================
# CodeAlpha - Car Price Prediction
# Author: Bollu Varshitha
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load Dataset
data = pd.read_csv("car data.csv")

# Display dataset
print("\n========== FIRST 5 ROWS ==========")
print(data.head())

print("\n========== DATASET INFO ==========")
print(data.info())

print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())

# Convert categorical columns into numbers
data = pd.get_dummies(data, drop_first=True)

# Features and Target
X = data.drop("Selling_Price", axis=1)
y = data["Selling_Price"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("\n========== MODEL PERFORMANCE ==========")
print("R² Score :", round(r2_score(y_test, y_pred), 3))
print("MAE      :", round(mean_absolute_error(y_test, y_pred), 3))
print("RMSE     :", round(mean_squared_error(y_test, y_pred) ** 0.5, 3))

# Scatter Plot
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Price")
plt.grid(True)
plt.show()

# Sample Prediction
sample = X.iloc[[0]]
prediction = model.predict(sample)

print("\n========== SAMPLE PREDICTION ==========")
print("Predicted Selling Price:", round(prediction[0], 2))

print("\nProject Completed Successfully!")