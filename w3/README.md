# Linear Regression Example with Noise

## Overview

This Streamlit app models the relationship between input variable \( X \) and output variable \( y \) using the equation:

\[
y = a \cdot X + 50 + c \cdot \text{noise}
\]

where:
- \( a \) is the coefficient that defines the slope of the line,
- \( c \) is a coefficient for the noise,
- noise is generated from a standard normal distribution.

The app allows users to interactively adjust parameters and visualize the linear regression model's performance.

## Features

- **Interactive Inputs**: Users can adjust the number of samples, coefficient \( a \), coefficient \( c \), and noise scale using sliders.
- **Real-time Visualization**: The app displays a scatter plot of true values versus predicted values, updating dynamically based on the input parameters.
- **Model Evaluation**: The app provides the estimated coefficient \( a \), intercept, and Mean Squared Error (MSE) of the model.

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- NumPy
- Plotly

## Installation

To run the app, you need to have Python installed on your machine. Follow these steps to set up the project:

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install required packages:**

   You can use pip to install the required libraries. It is recommended to create a virtual environment first.

   ```bash
   pip install streamlit scikit-learn plotly numpy
   ```

3. **Run the Streamlit app:**

   ```bash
   streamlit run linear_regression_app.py
   ```

## Usage

1. Adjust the parameters in the sidebar:
   - **Number of Samples**: Set the number of data points to generate.
   - **True Coefficient (a)**: Set the slope of the linear relationship.
   - **Noise Coefficient (c)**: Set the coefficient that influences the noise in the data.
   - **Noise Scale (c * noise)**: Adjust the scale of the noise added to the model.

2. The main area of the app will display:
   - The estimated coefficient \( a \) and intercept.
   - The Mean Squared Error (MSE) of the predictions.
   - A plot showing the true values and predicted values.

## Example Screenshot

*(You may want to insert an example screenshot of your app here)*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the Streamlit community for their continuous support and development of this interactive framework.
- Inspiration for the model design and structure came from various machine learning resources.
```

You can replace `<repository-url>` and `<repository-directory>` with the actual values for your project. If you need any additional sections or modifications, feel free to ask!