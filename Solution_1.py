import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Set random seed
tf.random.set_seed(42)

# Directory paths
train_dir = "e:\\Deep Learning\\xrays\\train"
test_dir = "e:\\Deep Learning\\xrays\\test"

# Data Preprocessing
train_datagen_augmented = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen_augmented.flow_from_directory(
    directory=train_dir,
    batch_size=32,
    target_size=(224, 224),
    class_mode="binary",
    seed=42
)

valid_data = valid_datagen.flow_from_directory(
    directory=test_dir,
    batch_size=32,
    target_size=(224, 224),
    class_mode="binary",
    seed=42
)

# Transfer Learning with EfficientNetB0
base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model layers

# Custom Feature Extraction Layers
custom_model = Sequential([
    base_model,
    Conv2D(32, 3, activation="relu", padding="same"),
    BatchNormalization(),
    MaxPool2D(2),
    Conv2D(64, 3, activation="relu", padding="same"),
    BatchNormalization(),
    MaxPool2D(2),
    Conv2D(128, 3, activation="relu", padding="same"),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dropout(0.4),
    Dense(1, activation="sigmoid")  # Output layer for binary classification
])

# Compile the Model
custom_model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

# Early Stopping Callback
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# Train the Custom Model (Base Frozen)
history_1 = custom_model.fit(
    train_data,
    validation_data=valid_data,
    epochs=20,
    callbacks=[early_stopping]
)

# Unfreeze Base Model for Fine-Tuning
base_model.trainable = True

# Compile the Model for Fine-Tuning
custom_model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    metrics=["accuracy"]
)

# Fine-Tune the Model
history_fine = custom_model.fit(
    train_data,
    validation_data=valid_data,
    epochs=10,
    callbacks=[early_stopping]
)

# Save the Final Model
custom_model.save("xray_binary_classifier_refined.h5")

# Plotting Training Curves
def plot_training_curves(history, history_fine):
    combined_loss = history.history["loss"] + history_fine.history["loss"]
    combined_val_loss = history.history["val_loss"] + history_fine.history["val_loss"]
    combined_accuracy = history.history["accuracy"] + history_fine.history["accuracy"]
    combined_val_accuracy = history.history["val_accuracy"] + history_fine.history["val_accuracy"]

    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(combined_loss, label="Training Loss")
    plt.plot(combined_val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(combined_accuracy, label="Training Accuracy")
    plt.plot(combined_val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.show()

plot_training_curves(history_1, history_fine)

# Evaluate the Model on Test Data
test_data = valid_datagen.flow_from_directory(
    directory=test_dir,
    batch_size=32,
    target_size=(224, 224),
    class_mode="binary"
)

test_loss, test_accuracy = custom_model.evaluate(test_data)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Performance Check
if test_accuracy >= 0.9:
    print("The model has achieved the target validation accuracy.")
else:
    print("Consider further fine-tuning or adding more data for better results.")
