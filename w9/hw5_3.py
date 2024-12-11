import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG19
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# 1. Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize images to the range [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. Load the VGG19 pre-trained model without the top layers
vgg_base = VGG19(weights="imagenet", include_top=False, input_shape=(32, 32, 3))

# Freeze the VGG19 base layers to avoid retraining them
vgg_base.trainable = False

# 3. Build the model
model = models.Sequential([
    vgg_base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# 4. Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# 5. Train the model
history = model.fit(x_train, y_train, 
                    validation_data=(x_test, y_test),
                    epochs=10, 
                    batch_size=64)

# 6. Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# 7. Save the model for deployment
model.save("cifar10_vgg19_model.h5")