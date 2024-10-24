# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import pickle
import optuna
import os

### PHASE 1: BUSINESS UNDERSTANDING ###
# Objective: Predict whether a passenger survived the Titanic disaster using the available features.

### PHASE 2: DATA UNDERSTANDING ###
# Load the Titanic datasets (train and test)
train_csv_file = 'w6/train.csv'
test_csv_file = 'w6/test.csv'

if not os.path.exists(train_csv_file) or not os.path.exists(test_csv_file):
    print(f"CSV file not found. Please ensure both 'train.csv' and 'test.csv' are in the current directory.")
else:
    # Load the data into DataFrames
    train_data = pd.read_csv(train_csv_file)
    test_data = pd.read_csv(test_csv_file)

    # Explore the dataset
    print(train_data.head())
    print(train_data.info())
    print(train_data.describe())
    print(test_data.info())  # Check missing data in the test set

### PHASE 3: DATA PREPARATION ###
    # Handle missing values
    train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
    test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
    
    train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
    test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace=True)
    
    test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

    # Drop irrelevant columns
    train_data.drop(columns=['Cabin'], inplace=True)
    test_data.drop(columns=['Cabin'], inplace=True)

    # Encode categorical variables
    train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
    test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
    
    train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True)
    test_data = pd.get_dummies(test_data, columns=['Embarked'], drop_first=True)

    # Drop unnecessary columns
    train_data.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)
    passenger_ids = test_data['PassengerId']  # Save PassengerId for final prediction output
    test_data.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

    # Feature scaling
    scaler = StandardScaler()
    train_data[['Age', 'Fare']] = scaler.fit_transform(train_data[['Age', 'Fare']])
    test_data[['Age', 'Fare']] = scaler.transform(test_data[['Age', 'Fare']])

    # Separate features and target variable
    X = train_data.drop(columns=['Survived'])
    y = train_data['Survived']

### PHASE 4: MODELING ###
    # Define a function for Optuna optimization
    def objective(trial):
        # Define a model
        model_name = trial.suggest_categorical('model', ['logistic_regression', 'random_forest', 'decision_tree'])

        if model_name == 'logistic_regression':
            model = LogisticRegression(max_iter=500)
        elif model_name == 'random_forest':
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            model = RandomForestClassifier(n_estimators=n_estimators)
        else:
            max_depth = trial.suggest_int('max_depth', 3, 12)
            model = DecisionTreeClassifier(max_depth=max_depth)

        # Feature selection with SelectKBest
        k = trial.suggest_int('k_best', 5, X.shape[1])
        selector = SelectKBest(f_classif, k=k)
        X_selected = selector.fit_transform(X, y)

        # Perform cross-validation
        score = cross_val_score(model, X_selected, y, cv=StratifiedKFold(n_splits=5), scoring='accuracy').mean()
        return score

    # Run Optuna optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # Output the best parameters
    print("Best trial:")
    trial = study.best_trial
    print(f"  Accuracy: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Train final model with best parameters
    best_model = trial.params['model']
    if best_model == 'logistic_regression':
        model = LogisticRegression(max_iter=500)
    elif best_model == 'random_forest':
        model = RandomForestClassifier(n_estimators=trial.params['n_estimators'])
    else:
        model = DecisionTreeClassifier(max_depth=trial.params['max_depth'])

    # Apply feature selection based on best 'k' features
    selector = SelectKBest(f_classif, k=trial.params['k_best'])
    X_selected = selector.fit_transform(X, y)
    X_test_selected = selector.transform(test_data)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

### PHASE 5: EVALUATION ###
    # Predict on validation set
    y_pred_val = model.predict(X_val)

    # Confusion matrix and accuracy
    cm = confusion_matrix(y_val, y_pred_val)
    print(f"Confusion Matrix:\n{cm}\n")

    acc = accuracy_score(y_val, y_pred_val)
    print(f"Accuracy: {acc:.4f}")

    # Additional metrics
    print(f"Precision: {precision_score(y_val, y_pred_val):.4f}")
    print(f"Recall: {recall_score(y_val, y_pred_val):.4f}")
    print(f"F1 Score: {f1_score(y_val, y_pred_val):.4f}")

### PHASE 6: DEPLOYMENT ###
    # Predict on test set
    y_test_pred = model.predict(X_test_selected)

    # Create DataFrame for the predictions
    prediction_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': y_test_pred
    })

    # Save predictions to CSV file
    prediction_df.to_csv('w6/predict.csv', index=False)
    print("Predictions saved to 'predict.csv'.")

    # Save the model
    with open('w6/titanic_optuna_model.pkl', 'wb') as file:
        pickle.dump(model, file)