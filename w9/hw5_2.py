import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

# Step 1: Data Preparation
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN input
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Step 2: Model Building
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2)

# Step 3: Save the Model
model.save('w9/handwriting_recognition_model.h5')
print("Model saved as 'handwriting_recognition_model.h5'.")

# Step 4: Evaluation
# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Classification report
y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred))

# Step 5: Visualization of 10 Predictions
def plot_predictions(model, test_images, test_labels, start_index=0):
    plt.figure(figsize=(15, 10))
    
    for i in range(10):
        # Get the test image and true label
        index = start_index + i
        test_image = test_images[index]
        true_label = np.argmax(test_labels[index])
        
        # Predict probabilities for the image
        probabilities = model.predict(test_image[np.newaxis, ...])[0]
        predicted_label = np.argmax(probabilities)
        
        # Plot the image
        plt.subplot(2, 5, i + 1)
        plt.imshow(test_image.squeeze(), cmap='gray')
        color = 'green' if predicted_label == true_label else 'red'
        plt.title(f"Predicted: {predicted_label}\nTrue: {true_label}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Display 10 predictions starting from the first test image
plot_predictions(model, x_test, y_test, start_index=0)
