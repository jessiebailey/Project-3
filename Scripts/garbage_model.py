
# Import necessary libraries for data handling, model building, and visualization
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt

# ================================
# CONFIGURATION
# ================================
# Location of the folder containing subfolders of waste images
MAIN_FOLDER = r"C:\Users\kmr8mp\Downloads\RealWaste"

# Target image size for the neural network (MobileNetV2 requires 224×224)
IMG_SIZE = (224, 224)

# Number of images fed to the model at once
BATCH_SIZE = 32

# Maximum number of times the model will see the full dataset during training
EPOCHS = 50

# Random seed to ensure reproducible results
SEED = 42

# Set seeds for reproducibility
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ================================
# 1. DATA PREPARATION
# ================================

# The ImageDataGenerator automatically loads images from folders
# and optionally applies transformations to make the model more robust.
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize pixel values from 0–255 to 0–1
    rotation_range=20,           # Randomly rotate images
    width_shift_range=0.2,       # Random horizontal shifting
    height_shift_range=0.2,      # Random vertical shifting
    zoom_range=0.2,              # Random zooming
    horizontal_flip=True,        # Random flipping
    validation_split=0.2         # Reserve 20% of data for validation
)

# Validation data should NOT be augmented—only rescaled
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

print("Loading training data...")
# Load training data (80% of dataset)
train_generator = train_datagen.flow_from_directory(
    MAIN_FOLDER,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',    # Multi-class classification
    subset='training',
    seed=SEED,
    shuffle=True                 # Shuffle for better learning
)

print("Loading validation data...")
# Load validation data (20% of dataset)
val_generator = val_datagen.flow_from_directory(
    MAIN_FOLDER,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=SEED,
    shuffle=False                # No need to shuffle when evaluating
)

# Dictionary mapping class names to numeric labels (e.g., {"Plastic": 0})
class_indices = train_generator.class_indices

# Reverse dictionary mapping numeric labels → class names
idx_to_class = {v: k for k, v in class_indices.items()}

print("\nClass mapping (index → label):")
for idx, label in idx_to_class.items():
    print(f"  {idx}: {label}")

# ================================
# 2. MODEL 1: Custom CNN
# ================================
# This model is built from scratch using convolutional layers.
def build_custom_cnn(num_classes, input_shape=(224, 224, 3)):
    model = Sequential([
        # Convolution layers learn image features like edges and textures
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        # Flatten to convert 2D features into a 1D vector
        Flatten(),

        # Fully connected neural network layers
        Dense(512, activation='relu'),
        Dropout(0.5),

        # Output layer (one neuron per class)
        Dense(num_classes, activation='softmax')
    ])
    return model

# ================================
# 3. MODEL 2: MobileNetV2 Transfer Learning
# ================================
# Transfer learning reuses a model pre-trained on millions of images.
def build_transfer_model(num_classes, input_shape=(224, 224, 3)):
    # Load MobileNetV2 but remove its classification head
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze to keep pre-trained features

    model = Sequential([
        base_model,                # Pre-trained feature extractor
        GlobalAveragePooling2D(),  # Reduces feature maps into a vector
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Final classification
    ])
    return model

# ================================
# Waste Category Mapping
# ================================
# Converts fine-grained classes into broad categories: Trash, Compost, Recycling.
WASTE_MAP = {
    "Textile Trash": "Trash",
    "Plastic": "Trash",
    "Miscellaneous Trash": "Trash",

    "Vegetation": "Compost",
    "Food Organics": "Compost",

    "Paper": "Recycling",
    "Metal": "Recycling",
    "Cardboard": "Recycling",
    "Glass": "Recycling"
}

def map_to_waste_category(predicted_label):
    """Convert a detailed class label into a broad waste category."""
    return WASTE_MAP.get(predicted_label, "Unknown")

# ================================
# 4. TRAINING FUNCTION + PERFORMANCE PLOTS
# ================================
def train_and_evaluate(model, model_name, save_path):

    # Compile the model with optimizer, loss function, and accuracy metric
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks help improve training stability
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    ]

    print(f"\nTraining {model_name}...")
    # Train the model using training and validation data
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )

    # Save the trained model to a file
    model.save(save_path)
    print(f"Model saved to: {save_path}")

    # ================================
    # Plot training and validation accuracy/loss
    # ================================
    plt.figure(figsize=(14,5))

    # Accuracy plot
    plt.subplot(1,2,1)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{model_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss plot
    plt.subplot(1,2,2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{model_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ================================
    # Final evaluation on validation set
    # ================================
    val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
    
    print(f"\n===== FINAL EVALUATION: {model_name} =====")
    print(f"Validation Loss:     {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

    return history, val_loss, val_accuracy

# ================================
# 5. MAIN EXECUTION (Program Entry Point)
# ================================
if __name__ == "__main__":

    # Determine how many classes exist based on folder names
    num_classes = len(train_generator.class_indices)
    print(f"Detected {num_classes} classes.")

    # Build both the custom CNN and the transfer learning model
    custom_cnn = build_custom_cnn(num_classes)
    transfer_model = build_transfer_model(num_classes)

    # Train the custom CNN
    print("\n" + "="*60)
    hist1, loss1, acc1 = train_and_evaluate(
        custom_cnn,
        "Custom CNN",
        "custom_cnn_garbage.h5"
    )

    # Train the transfer learning model
    print("\n" + "="*60)
    hist2, loss2, acc2 = train_and_evaluate(
        transfer_model,
        "MobileNetV2 (Transfer)",
        "mobilenetv2_garbage.h5"
    )

    # Display final mapping from class index → label → waste type
    print("\n" + "="*60)
    print("CLASS INDEX → GARBAGE CATEGORY MAPPING")
    print("="*60)
    for idx, label in sorted(idx_to_class.items()):
        print(f"Index {idx} → {label} → {map_to_waste_category(label)}")

    print("\nModels and evaluation complete!")
