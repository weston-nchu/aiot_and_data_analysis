import tensorflow as tf
import requests
from io import BytesIO
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Business Understanding
# Goal: Train a model to classify medical mask images (mask vs no_mask)

# Step 2: Data Understanding
# Assume the dataset is organized into two folders: train and validation
# Each folder has subfolders: 'mask' and 'no_mask'

train_dir = '/path/to/train'
val_dir = '/path/to/val'

# Step 3: Data Preparation

# ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize images
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Match the input size of VGG16
    batch_size=32,
    class_mode='binary'  # 2 classes: mask, no_mask
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Step 4: Modeling

# Load VGG16 pretrained model
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model to retain its pretrained weights
base_model.trainable = False

# Add custom layers on top of VGG16
model = models.Sequential([
    base_model,  # Pretrained VGG16 base model
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')  # 2 classes: mask and no_mask
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Display model architecture
model.summary()

# Step 5: Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,  # Number of epochs can be adjusted
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Step 6: Evaluation
# Evaluate the model on validation data
val_loss, val_acc = model.evaluate(validation_generator)
print(f'Validation Accuracy: {val_acc * 100:.2f}%')

# Step 7: Deployment - Predict using HTTP input image
def load_and_predict_image(url):
    # Download the image
    response = requests.get(url)
    img = image.load_img(BytesIO(response.content), target_size=(224, 224))
    
    # Convert the image to an array
    img_array = image.img_to_array(img)
    
    # Rescale the image to match the model's input
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    
    # Make prediction
    predictions = model.predict(img_array)
    
    # Get class label (binary: mask = 0, no_mask = 1)
    class_idx = np.argmax(predictions, axis=1)
    class_label = train_generator.class_indices
    label = {v: k for k, v in class_label.items()}[class_idx[0]]
    
    return label

# Example usage: Input URL of the image to predict
image_url = 'http://example.com/path/to/image.jpg'
predicted_class = load_and_predict_image(image_url)
print(f'The predicted class for the image is: {predicted_class}')