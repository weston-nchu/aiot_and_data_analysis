import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Title of the Streamlit App
st.title("Linear Regression Example with Noise")

# Create ChatGPT prompt history dialog
with open("w3/chatgpt_prompt_history.html", "r", encoding="utf-8") as f:
    html_content = f.read()
@st.dialog("ChatGpt prompt history", width="large")
def prompt_history():
    components.html(html_content, width=700, height=600, scrolling=True)

# CRISP-DM Step 1: Business Understanding
st.markdown("""
### Objective:
This app models the relationship between `X` and `y` where `y = a * X + 50 + c * noise` using Linear Regression.
The model estimates the coefficient `a`, the intercept, and evaluates the model performance.
""")

# Create a sidebar for inputs
st.sidebar.header("Parameters")

n_samples = st.sidebar.slider("Number of Samples", min_value=100, max_value=2000, value=1000, step=100)
a_true = st.sidebar.slider("True Coefficient (a)", min_value=0.0, max_value=100.0, value=2.0)
c_true = st.sidebar.slider("Noise Coefficient (c)", min_value=0.0, max_value=100.0, value=1.0)
noise_scale = st.sidebar.slider("Noise Scale (c * noise)", min_value=0.0, max_value=500.0, value=100.0)

# Create ChatGPT float button
gpt_prompt_btn = st.button("", on_click=prompt_history)
gpt_btn_style = """
    <style>
        button[kind="secondary"] {
            position: fixed;
            bottom: 60px;
            right: 40px;
            width: 70px;
            height: 70px;
            background-image: url("https://cdn-icons-png.flaticon.com/512/12222/12222588.png");
            background-size: cover;
            border-radius: 50%;
            border: none;
            cursor: pointer;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }

        button[kind="secondary"]:hover {
            border
        }
    </style>
"""
st.markdown(gpt_btn_style, unsafe_allow_html=True)

# CRISP-DM Step 2: Data Understanding - Generate synthetic data
np.random.seed(42)  # For reproducibility
X = np.random.rand(n_samples, 1) * 10  # Random X values between 0 and 10
noise = np.random.randn(n_samples, 1) * noise_scale  # Normally distributed noise scaled by noise_scale
y = a_true * X + 50 + c_true * noise

# CRISP-DM Step 3: Data Preparation - Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CRISP-DM Step 4: Modeling - Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# CRISP-DM Step 5: Evaluation - Predict on test set and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Display the model results
st.write(f"### Model Results:")
st.write(f"Estimated Coefficient (a): {model.coef_[0][0]:.2f}")
st.write(f"Intercept: {model.intercept_[0]:.2f}")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")

# CRISP-DM Step 5 (cont.): Visualize the predictions with Plotly
fig = go.Figure()

# Add scatter plot for actual test data
fig.add_trace(go.Scatter(x=X_test.flatten(), y=y_test.flatten(),
                         mode='markers', name='True Values', marker=dict(color='blue')))

# Add line plot for predicted data
fig.add_trace(go.Scatter(x=X_test.flatten(), y=y_pred.flatten(),
                         mode='lines', name='Predicted Values', line=dict(color='red', width=2)))

# Update the layout of the plot
fig.update_layout(title="Linear Regression Predictions",
                  xaxis_title="X",
                  yaxis_title="y",
                  legend_title="Legend")

# Display the plot
st.plotly_chart(fig)

# CRISP-DM Step 6: Deployment - In real-world cases, you might save the model for future use.
