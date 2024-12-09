# Import necessary libraries
import pandas as pd
from pycaret.classification import *

# Step 1: Load the Titanic dataset
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# Step 2: Data Preparation for PyCaret
# Selecting relevant columns
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# Filling missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Converting categorical features to strings for encoding
df['Pclass'] = df['Pclass'].astype(str)
df['SibSp'] = df['SibSp'].astype(str)
df['Parch'] = df['Parch'].astype(str)

# Step 3: Feature Engineering, Model Selection, and Hyperparameter Optimization
# Initialize PyCaret
automl_setup = setup(
    data=df,
    target='Survived',
    session_id=123,  # Ensures reproducibility
    normalize=True,  # Normalize features
    fold=5  # 5-fold cross-validation
)

# Step 4: Model Selection
# Compare multiple models and select the best one
best_model = compare_models()

# Step 5: Hyperparameter Optimization
# Fine-tune the selected model
tuned_model = tune_model(best_model)

# Step 6: Evaluate the Optimized Model
# Plot model evaluation metrics
evaluate_model(tuned_model)

# Step 7: Finalize and Save the Model
final_model = finalize_model(tuned_model)
save_model(final_model, 'w8//optimized_titanic_model')

# Step 8: Make Predictions on New Data
# Example: Generate predictions on the dataset
predictions = predict_model(final_model, data=df)
print(predictions.head())

# Optional: Feature Importance Plot
plot_model(tuned_model, plot='feature')  # Visualize feature importance
