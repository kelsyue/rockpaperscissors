"""
test_model.py - Single Image Classifier Test (Laptop)
======================================================
Rubric: "CNN on local machine able to classify rock paper scissors images
        with an accuracy of greater than 50%" (20 pts)

This script tests the trained prs_cnn.h5 model on a single BMP image
on the laptop. It applies the EXACT same preprocessing pipeline used
during training and on the ESP32, ensuring consistent results.

Usage:
    python3 test_model.py path/to/image.bmp

Example:
    python3 test_model.py datasets/rock/0000.bmp

Note: Preprocessing pipeline and model loading code assisted by Claude AI (Anthropic). 
      Predictions and evaluation written by student.
"""

import sys
import os
import numpy as np
# pillow - for loading and resizing BMP images
from PIL import Image 
# tensorFlow/Keras - for loading the trained model         
import tensorflow as tf        


# class labels 
CLASSES = ['rock', 'paper', 'scissors']

# final image size 
IMG_SIZE = 32


def preprocess_image(image_path):
    """
    Applies the exact same preprocessing pipeline used during training
    and on the ESP32 at inference time.

    This is critical - if preprocessing differs between training and
    inference, the model will perform poorly even if it was well trained.

    Pipeline:
        1. Load image as grayscale (single channel)
        2. Resize from 96x96 to 32x32 using PIL bicubic interpolation
        3. Apply binary threshold at 128 (hand=white, background=black)
        4. Normalize to 0.0-1.0 float range
        5. Reshape to (1, 32, 32, 1) for CNN input (batch of 1)

    Args:
        image_path (str): Path to a raw 96x96 grayscale BMP image

    Returns:
        np.ndarray: Preprocessed image array of shape (1, 32, 32, 1)
                    or None if loading/processing fails
    """
    try:
        # step 1: load as grayscale
        img = Image.open(image_path).convert('L')

        # step 2: resize to 32x32
        img = img.resize((IMG_SIZE, IMG_SIZE))

        # step 3: convert to numpy for numerical operations
        img_np = np.array(img)

        # step 4: binary threshold at 128 - matches ESP32 preprocessing
        # Pixels >= 128 become 1.0 (white = hand silhouette)
        # Pixels < 128 become 0.0 (black = background)
        img_np = np.where(img_np >= 128, 1.0, 0.0).astype('float32')

        # step 5: reshape to (1, 32, 32, 1) 
        return img_np.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def test_single_image(image_path):
    """
    Loads the trained model and classifies a single image.

    Prints the confidence score for each class and the final prediction.

    Args:
        image_path (str): Path to the BMP image to classify
    """
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    # load the trained keras model from disk
    try:
        model = tf.keras.models.load_model('prs_cnn.h5')
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # preprocess the image using the same pipeline as training
    input_data = preprocess_image(image_path)
    if input_data is None:
        return

    # run inference - model outputs probability for each of 3 classes
    preds = model.predict(input_data, verbose=0)[0]

    # find index of highest probability class
    idx = np.argmax(preds)

    # display results in a readable format
    print(f"\nResults for: {image_path}")
    print("-" * 30)
    for i, label in enumerate(CLASSES):
        # show each class with its confidence percentage
        print(f"{label:10}: {preds[i]*100:6.2f}%")
    print("-" * 30)
    print(f"Final Prediction: {CLASSES[idx].upper()} (Confidence: {preds[idx]:.2f})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_model.py path/to/image.bmp")
    else:
        test_single_image(sys.argv[1])