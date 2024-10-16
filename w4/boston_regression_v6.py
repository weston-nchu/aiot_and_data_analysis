# Version: v6.0
# Author: Weston
# Date: 2024-10-14
# Description: Solving the Boston Housing Problem using Scikit-Learn and CRISP-DM
# Updated to evaluate models by R² Score and MSE.

import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Data Understanding
# Fetch the dataset from the URL
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
response = requests.get(url)

# Save the CSV content to a pandas DataFrame
data = pd.read_csv(pd.compat.StringIO(response.text))

# Display basic information about the dataset
print(data.info())
print(data.describe())

# Step 3: Data Preparation
# Check for missing values
print(data.isnull().sum())

# Split the data into features and target variable
X = data.drop('medv', axis=1)  # 'medv' is the target variable
y = data['medv']

# Step 4: Train-Test Split
# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data (optional, based on model choice)
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()

# Step 5: Modeling
# Choose models: Linear Regression, Lasso Regression, and Random Forest
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lasso_model = Lasso(alpha=0.1)  # You can adjust the alpha parameter
lasso_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predicting
y_pred_lr = lr_model.predict(X_test)
y_pred_lasso = lasso_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Step 6: Evaluation
# Evaluate Linear Regression model
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f'Linear Regression MAE: {mae_lr:.2f}, MSE: {mse_lr:.2f}, R²: {r2_lr:.2f}')

# Evaluate Lasso Regression model
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
print(f'Lasso Regression MAE: {mae_lasso:.2f}, MSE: {mse_lasso:.2f}, R²: {r2_lasso:.2f}')

# Evaluate Random Forest model
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f'Random Forest MAE: {mae_rf:.2f}, MSE: {mse_rf:.2f}, R²: {r2_rf:.2f}')

# Optional: Visualization of Predictions
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.scatterplot(x=y_test, y_pred_lr)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Linear Regression Predictions')

plt.subplot(1, 3, 2)
sns.scatterplot(x=y_test, y_pred_lasso)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Lasso Regression Predictions')

plt.subplot(1, 3, 3)
sns.scatterplot(x=y_test, y_pred_rf)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Random Forest Predictions')

plt.tight_layout()
plt.show()
