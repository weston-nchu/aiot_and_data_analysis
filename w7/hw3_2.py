import streamlit as st
import numpy as np
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Business Understanding: Explain the purpose in the app
st.title("2D SVM Classification with Circular Data")
st.markdown("""
This app demonstrates an SVM classifier on synthetic circular data. 
You can adjust parameters to explore how the decision boundary changes.
""")

# 2. Data Understanding and 3. Data Preparation
noise = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1, 0.01)
X, y = make_circles(n_samples=500, factor=0.5, noise=noise, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Visualize data
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(X[y == 0, 0], X[y == 0, 1], color="blue", label="Class 0", alpha=0.6)
ax.scatter(X[y == 1, 0], X[y == 1, 1], color="red", label="Class 1", alpha=0.6)
ax.set_title("Circular Data Distribution")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()
st.pyplot(fig)

# 4. Modeling
c_value = st.sidebar.slider("Regularization Parameter (C)", 0.1, 100.0, 1.0, 0.1)
kernel = st.sidebar.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"], index=2)
model = SVC(kernel=kernel, C=c_value)
model.fit(X_scaled, y)

# Prepare grid for decision surface
xx, yy = np.meshgrid(np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 100),
                     np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 100))
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 5. Evaluation: 3D plot of the decision boundary
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1], y[y == 0], color="blue", alpha=0.6, label="Class 0")
ax.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1], y[y == 1], color="red", alpha=0.6, label="Class 1")
ax.plot_surface(xx, yy, Z, alpha=0.3, cmap="coolwarm")
ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors="black")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Decision Value")
ax.set_title("3D Decision Surface")
ax.legend()
st.pyplot(fig)

# 6. Deployment: Run the app
st.write("Adjust the sliders and dropdowns to see how the model responds!")
