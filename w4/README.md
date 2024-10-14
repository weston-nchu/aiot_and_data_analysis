# Boston Housing Problem Regression

This project demonstrates how to solve the Boston Housing Problem using Scikit-Learn while following the CRISP-DM methodology. The goal is to predict housing prices based on various features.

## Table of Contents
1. [Project Description](#project-description)
2. [Installation](#installation)
3. [Data Source](#data-source)
4. [Methodology](#methodology)
5. [Code](#code)
6. [Model Evaluation](#model-evaluation)
7. [Visualization](#visualization)
8. [Version Control](#version-control)

## Project Description
The Boston Housing Problem dataset consists of housing information from the Boston area. The aim is to build regression models to predict the median value of owner-occupied homes.

## Installation
Make sure to install the required libraries before running the code:

```bash
pip install pandas numpy requests scikit-learn matplotlib seaborn
```

## Data Source
The dataset is fetched from the following URL:
1. [Boston Housing Dataset](https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv)

## Methodology
This project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology:
1. **Data Understanding**: Understanding the data structure and key features.
2. **Data Preparation**: Cleaning and preparing the data for modeling, including handling missing values and splitting the dataset.
3. **Modeling**: Building different regression models (Linear Regression, Lasso Regression, and Random Forest) to predict housing prices.
4. **Evaluation**: Assessing the performance of the models using appropriate metrics.
5. **Deployment (optional)**: Integrating the model into a production environment (not covered in this project).

## Model Evaluation

### 1. Models Evaluated
1. **Linear Regression**: A simple and interpretable linear model that predicts the target variable as a linear combination of input features.
2. **Lasso Regression**: A linear model that incorporates L1 regularization to reduce the risk of overfitting by penalizing large coefficients.
3. **Random Forest**: An ensemble model that combines the predictions of multiple decision trees to enhance accuracy and robustness.

### 2. Performance Metrics Used
1. **Mean Absolute Error (MAE)**: Measures the average absolute difference between predicted and actual values, providing insight into prediction accuracy.
2. **Mean Squared Error (MSE)**: Calculates the average squared difference between predicted and actual values, emphasizing larger errors due to squaring.
3. **R² Score**: Represents the proportion of variance in the target variable that can be explained by the model, indicating how well the model fits the data.

## Visualization
Visualizations of predictions from each model are generated to compare actual vs. predicted values. Scatter plots are used to illustrate the relationship between actual housing prices and predicted prices for each model.

## Version Control
Version: 7.2
Author: Weston
Date: 2024-10-14

## ChatGPT Prompt Log
[ChatGpt Prompt Log](https://chatgpt.com/share/670cb1de-bbd8-8003-ad15-b0805fe3b705)