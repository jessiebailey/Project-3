# evaluate_garbage_model.py
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
MAIN_FOLDER = "/Users/arianaelahi/Desktop/realwaste-main/RealWaste"
IMG_SIZE = (224, 224)  # MobileNetV2 expects 224x224
BATCH_SIZE = 32
EPOCHS = 50
SEED = 42

# Ensure reproducibility
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ================================
# 1. DATA PREPARATION
# ================================
# Data generator with augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% train, 20% validation
)

# Only rescaling for validation
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

print("Loading training data...")
train_generator = train_datagen.flow_from_directory(
    MAIN_FOLDER,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=SEED,
    shuffle=True
)

print("Loading validation data...")
val_generator = val_datagen.flow_from_directory(
    MAIN_FOLDER,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=SEED,
    shuffle=False
)

# Get class labels and indices
class_indices = train_generator.class_indices
idx_to_class = {v: k for k, v in class_indices.items()}
print("\nClass mapping (index → label):")
for idx, label in idx_to_class.items():
    print(f"  {idx}: {label}")

# ================================
# 2. MODEL 1: Custom CNN from Scratch
# ================================
def build_custom_cnn(num_classes, input_shape=(224, 224, 3)):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# ================================
# 3. MODEL 2: Transfer Learning with MobileNetV2
# ================================
def build_transfer_model(num_classes, input_shape=(224, 224, 3)):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# ================================
# 4. TRAINING FUNCTION
# ================================
def train_and_evaluate(model, model_name, save_path):
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    ]

    print(f"\nTraining {model_name}...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )

    # Save model
    model.save(save_path)
    print(f"Model saved to: {save_path}")

    # ================================
    # 5. EVALUATION METRICS
    # ================================
    print(f"\nEvaluating {model_name} on validation set...")
    val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
    
    print(f"\n===== FINAL EVALUATION: {model_name} =====")
    print(f"Validation Loss:     {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

    return history, val_loss, val_accuracy

# ================================
# 6. MAIN EXECUTION
# ================================
if __name__ == "__main__":
    num_classes = len(train_generator.class_indices)
    print(f"Detected {num_classes} classes.")

    # Build models
    custom_cnn = build_custom_cnn(num_classes)
    transfer_model = build_transfer_model(num_classes)

    # Train & evaluate both
    print("\n" + "="*60)
    hist1, loss1, acc1 = train_and_evaluate(
        custom_cnn,
        "Custom CNN",
        "custom_cnn_garbage.h5"
    )

    print("\n" + "="*60)
    hist2, loss2, acc2 = train_and_evaluate(
        transfer_model,
        "MobileNetV2 (Transfer)",
        "mobilenetv2_garbage.h5"
    )

    # ================================
    # 7. FINAL CLASS MAPPING (for inference)
    # ================================
    print("\n" + "="*60)
    print("CLASS INDEX → GARBAGE CATEGORY MAPPING")
    print("="*60)
    for idx, label in sorted(idx_to_class.items()):
        print(f"Index {idx} → {label}")

    print("\nModels and evaluation complete!")
    print("Saved models: custom_cnn_garbage.h5, mobilenetv2_garbage.h5")