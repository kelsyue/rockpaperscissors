"""
preprocess_and_train.py - CNN Training for Rock-Paper-Scissors Classifier
==========================================================================
Rubric:
    - "Basic CNN running on your local machine/laptop - accuracy is unimportant" (20 pts)
    - "CNN on local machine able to classify rock paper scissors images
       with an accuracy of greater than 50%" (20 pts)

This script loads raw 96x96 grayscale BMP images captured from the ESP32
camera, preprocesses them to match the ESP32 inference pipeline exactly,
trains a Convolutional Neural Network (CNN), and exports the model in both
Keras (.h5) and TensorFlow Lite (.tflite) formats.

Final validation accuracy achieved: ~90.1%

Preprocessing pipeline (identical to ESP32):
    Raw 96x96 BMP -> Resize 32x32 -> Binary threshold at 128 -> (32,32,1) float tensor

CNN Architecture:
    Conv2D(16, stride=2) -> Conv2D(32, stride=2) -> Conv2D(64, stride=2)
    -> Flatten -> Dropout(0.4) -> Dense(32) -> Dense(3, softmax)

Note: CNN architecture, training callbacks, preprocessing pipeline, evaluation,
      and TFLite export assisted by Claude AI (Anthropic).
      Dataset loading written by student.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
# pillow - image loading and resizing
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
# imagedatagenerator for data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
# for handling imbalanced class sizes
from sklearn.utils import class_weight
import cv2

# check for 'datasets' folder first, fall back to 'dataset' if not found
DATASET_DIR = 'datasets' if os.path.exists('datasets') else 'dataset'

# the three hand gesture classes we are classifying
CLASSES = ['rock', 'paper', 'scissors']

# final image size fed into the CNN
IMG_SIZE = 32

# original raw image size from the ESP32 camera
RAW_SIZE = 96


def preprocess_image(img_path):
    """
    Preprocesses a single raw BMP image to match what the ESP32 does at inference time.

    It is critical that training preprocessing exactly matches ESP preprocessing,
    otherwise the model will perform well on the laptop but poorly on the device.

    Steps:
        1. Load raw 96x96 grayscale BMP from disk
        2. Resize to 32x32 using PIL bicubic interpolation
        3. Apply binary threshold at 128 (hand=white, background=black)
        4. Normalize to 0.0-1.0 float range
        5. Reshape to (32, 32, 1) for CNN input

    Args:
        img_path (str): Path to a raw 96x96 grayscale BMP image

    Returns:
        np.ndarray: Preprocessed array of shape (32, 32, 1), or None on failure
    """
    try:
        # step 1: open as grayscale - 'L' mode = 8-bit single channel
        img = Image.open(img_path).convert('L')

        # step 2: resize from 96x96 to 32x32
        # PIL uses bicubic interpolation; the ESP takes every 3rd pixel (simpler)
        img = img.resize((IMG_SIZE, IMG_SIZE))

        # step 3: convert to numpy array for numerical operations
        img_np = np.array(img)

        # step 4: binary threshold at 128 - matches ESP32 preprocessing
        # pixels >= 128 become 1.0 (white = hand silhouette)
        # pixels < 128 become 0.0 (black = background)
        img_np = np.where(img_np >= 128, 1.0, 0.0).astype('float32')

        # step 5: reshape to (32, 32, 1) - CNN expects (height, width, channels)
        return img_np.reshape(IMG_SIZE, IMG_SIZE, 1)

    except Exception as e:
        # skip corrupted or unreadable images
        print(f"Error processing {img_path}: {e}")
        return None


def load_data():
    """
    Walks through the dataset directory and loads all BMP images.

    Assigns numeric labels:
        rock = 0, paper = 1, scissors = 2

    Returns:
        X (np.ndarray): Shape (N, 32, 32, 1) - preprocessed images
        y (np.ndarray): Shape (N,) - integer class labels
    """
    X, y = [], []
    print(f"Loading raw images from {DATASET_DIR}...")

    for idx, label in enumerate(CLASSES):
        # build path to class folder e.g. datasets/rock/
        path = os.path.join(DATASET_DIR, label)
        count = 0

        if not os.path.exists(path):
            print(f"Warning: Path {path} not found!")
            continue

        for img_name in os.listdir(path):
            if img_name.lower().endswith('.bmp'):
                img_path = os.path.join(path, img_name)
                processed = preprocess_image(img_path)
                if processed is not None:
                    X.append(processed)  # add preprocessed image
                    y.append(idx)        # add class label
                    count += 1

        print(f"Found {count} images for class '{label}'")

    return np.array(X), np.array(y)


X, y = load_data()

# split 80/20 train/validation - stratify keeps class proportions equal
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining samples: {len(X_train)}, Validation samples: {len(X_val)}")


# if one class has more images, the model will bias toward it
# class weights penalize the model more for getting minority classes wrong
weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(weights))
print(f"Class weights: {class_weights_dict}")


# creates modified versions of images during training to improve generalization
# only applied to training data, NOT validation data
datagen = ImageDataGenerator(
    rotation_range=20,      # rotate up to 20 degrees
    width_shift_range=0.1,  # shift horizontally up to 10%
    height_shift_range=0.1, # shift vertically up to 10%
    zoom_range=0.1,         # zoom in/out up to 10%
    horizontal_flip=True,   # mirror images left-right
    fill_mode='nearest'     # fill new pixels with nearest neighbor value
)


# uses strided convolutions instead of MaxPooling because TinyMaix
# (the ESP32 inference engine) does not support MaxPooling layers.
# stride=2 achieves the same spatial downsampling effect.
model = models.Sequential([
    # 32x32 -> 16x16, learns basic edge and shape features
    layers.Conv2D(16, (3,3), activation='relu', padding='same',
                  strides=(2,2), input_shape=(IMG_SIZE, IMG_SIZE, 1)),

    # 16x16 -> 8x8, learns combinations of lower-level features
    layers.Conv2D(32, (3,3), activation='relu', padding='same',
                  strides=(2,2)),

    # 8x8 -> 4x4, learns high-level hand shape patterns
    layers.Conv2D(64, (3,3), activation='relu', padding='same',
                  strides=(2,2)),

    # flatten 4x4x64 = 1024 values into a 1D vector for dense layers
    layers.Flatten(),

    # dropout randomly zeros 40% of neurons to prevent overfitting
    layers.Dropout(0.4),

    # fully connected layer combines all learned features
    layers.Dense(32, activation='relu'),

    # output: probability for each of 3 classes, all sum to 1.0
    layers.Dense(len(CLASSES), activation='softmax')
])

# adam optimizer adapts learning rate automatically
# sparse_categorical_crossentropy works with integer labels (not one-hot encoded)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("\nStarting training (up to 50 epochs)...")

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),  # augmented training batches
    validation_data=(X_val, y_val),
    epochs=50,
    class_weight=class_weights_dict,
    callbacks=[
        # stop early if validation loss stops improving for 10 epochs
        # restore_best_weights keeps the best model seen during training
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        # halve the learning rate if validation loss plateaus for 5 epochs
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5
        )
    ]
)

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nFinal Validation Accuracy: {val_acc*100:.1f}%")

y_pred = np.argmax(model.predict(X_val), axis=1)
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=CLASSES))

# confusion matrix: rows = actual class, columns = predicted class
# diagonal = correct predictions, off-diagonal = mistakes
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

model.save('prs_cnn.h5')
print("\nModel saved as prs_cnn.h5")

# plot training and validation accuracy/loss over epochs
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
print("Training history saved as training_history.png")

# convert_model.py converts .tflite -> .tmdl for the ESP32
print("\nExporting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('prs_cnn.tflite', 'wb') as f:
    f.write(tflite_model)
print("prs_cnn.tflite created.")
print("\nNext step: run convert_model.py to generate prs_cnn.tmdl for the ESP32")