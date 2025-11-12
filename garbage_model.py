import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# Path to dataset
base_dir = r"C:\Users\kmr8mp\Downloads\Garbage classification" 

# Data preprocessing and augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255, 
    validation_split=0.2,  # 80/20 train-validation split
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# Create train and validation sets
train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Build the CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(200,200,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(6, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=25
)

# Save the model
model.save("garbage_classifier_model.h5")

# Class labels
class_indices = train_generator.class_indices
print("Class indices:", class_indices)

loss, acc = model.evaluate(val_generator)
print(f"Validation accuracy: {acc*100:.2f}%")

#%%
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("garbage_classifier_model.h5")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    class_name = list(class_indices.keys())[list(class_indices.values()).index(class_idx)]

    recyclable = class_name != "trash"
    print(f"Predicted class: {class_name}")
    print(f"Recyclable: {'Yes' if recyclable else 'No'}")

# Example:
predict_image("trash9.jpg")

#%%

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import os

# --- 1️⃣ Define base directory ---
base_dir = r"C:\Users\kmr8mp\Downloads\Garbage classification"

# --- 2️⃣ Data preprocessing and augmentation ---
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 80% train / 20% validation
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# --- 3️⃣ Load MobileNetV2 base model ---
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,   # exclude the classification head
    weights='imagenet'   # use pretrained ImageNet weights
)
base_model.trainable = False  # freeze convolutional layers

# --- 4️⃣ Build the model ---
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(6, activation='softmax')  # 6 categories
])

# --- 5️⃣ Compile model ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- 6️⃣ Training callbacks ---
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.3, patience=2, min_lr=1e-6)
]

# --- 7️⃣ Train the model ---
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=callbacks
)

# --- 8️⃣ Save the model ---
model.save("mobilenet_garbage_classifier.h5")

# --- 9️⃣ Show class indices ---
class_indices = train_generator.class_indices
print("Class indices:", class_indices)

#%%
loss, acc = model.evaluate(val_generator)
print(f"Validation accuracy: {acc*100:.2f}%")
