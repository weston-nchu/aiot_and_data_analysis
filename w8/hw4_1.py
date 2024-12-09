# Import necessary libraries
import pandas as pd
from pycaret.classification import *
import matplotlib.pyplot as plt

# Load Titanic dataset
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# Data Preparation
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Pclass'] = df['Pclass'].astype(str)
df['SibSp'] = df['SibSp'].astype(str)
df['Parch'] = df['Parch'].astype(str)

# PyCaret Setup
automl_setup = setup(
    data=df,
    target='Survived',
    session_id=123,  # Ensures reproducibility
    fold=5,          # Cross-validation folds
    verbose=False    # Suppress verbose output
)

# Compare Models
comparison_results = compare_models(n_select=16)  # Select top 16 models

# Visualization of Results
performance_metrics = pull()  # Retrieve model comparison results
performance_metrics = performance_metrics.sort_values(by="Accuracy", ascending=False)

# Plot Accuracy
plt.figure(figsize=(14, 8))
plt.barh(performance_metrics['Model'], performance_metrics['Accuracy'], color='skyblue')
plt.title("Accuracy of Top 16 Models", fontsize=16)
plt.xlabel("Accuracy", fontsize=14)
plt.ylabel("Model", fontsize=14)
plt.gca().invert_yaxis()  # Best model on top
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()
