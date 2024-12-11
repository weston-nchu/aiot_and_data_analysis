# Step 1: Business Understanding
# Goal: Classify the Iris dataset into three species (Setosa, Versicolour, Virginica) using PyTorch.

# Step 2: Data Understanding
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# Load the dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

print(data.describe())

# Step 3: Data Preparation
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# One-hot encode target labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert the data into PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Step 4: Modeling
# Define a simple neural network model in PyTorch
class IrisClassifier(nn.Module):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

# Instantiate the model
model = IrisClassifier()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Training and Evaluation
# Create DataLoader for training and validation data
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

valid_data = TensorDataset(X_test_tensor, y_test_tensor)
valid_loader = DataLoader(valid_data, batch_size=8, shuffle=False)

# Track loss and accuracy
train_losses = []
train_accuracies = []
valid_losses = []
valid_accuracies = []

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, torch.argmax(labels, dim=1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == torch.argmax(labels, dim=1)).sum().item()

    # Calculate average training loss and accuracy for the epoch
    epoch_train_loss = running_loss / len(train_loader)
    epoch_train_accuracy = 100 * correct / total

    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_accuracy)

    # Validation phase
    model.eval()
    running_valid_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in valid_loader:
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, torch.argmax(labels, dim=1))
            running_valid_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == torch.argmax(labels, dim=1)).sum().item()

    # Calculate average validation loss and accuracy for the epoch
    epoch_valid_loss = running_valid_loss / len(valid_loader)
    epoch_valid_accuracy = 100 * correct / total

    valid_losses.append(epoch_valid_loss)
    valid_accuracies.append(epoch_valid_accuracy)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.2f}%, Valid Loss: {epoch_valid_loss:.4f}, Valid Accuracy: {epoch_valid_accuracy:.2f}%')

# Step 6: Evaluation on Test Data
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    _, predicted_classes = torch.max(y_pred, 1)
    y_test_classes = torch.argmax(y_test_tensor, dim=1)
    accuracy = (predicted_classes == y_test_classes).sum().item() / y_test_classes.size(0)

print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Step 7: Deployment
# Example of predicting with new data
new_data = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example input
new_data_scaled = scaler.transform(new_data)
new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

model.eval()
with torch.no_grad():
    prediction = model(new_data_tensor)
    predicted_class = torch.argmax(prediction, dim=1).item()

# Inverse transform the predicted class (fix the issue here)
class_prediction = encoder.inverse_transform(np.eye(3)[predicted_class].reshape(1, -1))
print(f"Predicted Class: {class_prediction[0][0]}")

# Additional: Visualizing Performance
import matplotlib.pyplot as plt

# Plot training and validation loss and accuracy
plt.figure(figsize=(12, 6))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', color='red')
plt.plot(valid_losses, label='Validation Loss', color='orange')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy', color='blue')
plt.plot(valid_accuracies, label='Validation Accuracy', color='green')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()
