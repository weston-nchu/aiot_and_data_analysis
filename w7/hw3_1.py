# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Business Understanding
# Objective: Compare Logistic Regression and SVM classifiers on a 1D classification problem.

# Step 2: Data Understanding
# Generate a simple 1D dataset
np.random.seed(42)
X = np.random.rand(300, 1) * 1000  # 300 samples, single feature in range [0, 1000]
y = (X[:, 0] > 500).astype(int)    # Classify based on a threshold of 500

# Step 3: Data Preparation
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Modeling
# Train Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Train SVM model
svm = SVC(kernel='linear', probability=True)  # Use linear kernel for simplicity
svm.fit(X_train, y_train)

# Step 5: Evaluation
# Predict on the test set
y_pred_log_reg = log_reg.predict(X_test)
y_pred_svm = svm.predict(X_test)

# Compute accuracies
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Display results
print("Logistic Regression Accuracy:", accuracy_log_reg)
print("SVM Accuracy:", accuracy_svm)
print("\nClassification Report for Logistic Regression:\n", classification_report(y_test, y_pred_log_reg))
print("\nClassification Report for SVM:\n", classification_report(y_test, y_pred_svm))

# Step 6: Deployment
# Visualize the predictions
x_line = np.linspace(0, 1000, 300).reshape(-1, 1)  # 300 points in range [0, 1000]
y_pred_log_reg = log_reg.predict(x_line)
y_pred_svm = svm.predict(x_line)

# Plot Logistic Regression predictions
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], y, color='blue', label='True Data', edgecolor='k')
plt.scatter(x_line, y_pred_log_reg, color='red', label='Predictions', marker='x', s=10)
plt.axvline(500, color='black', linestyle=':', label='True Threshold')
plt.title('Logistic Regression Predictions')
plt.xlabel('Feature (0 to 1000)')
plt.ylabel('Predicted Class')
plt.legend()

# Plot SVM predictions
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], y, color='blue', label='True Data', edgecolor='k')
plt.scatter(x_line, y_pred_svm, color='green', label='Predictions', marker='x', s=10)
plt.axvline(500, color='black', linestyle=':', label='True Threshold')
plt.title('SVM Predictions')
plt.xlabel('Feature (0 to 1000)')
plt.ylabel('Predicted Class')
plt.legend()

plt.tight_layout()
plt.show()
