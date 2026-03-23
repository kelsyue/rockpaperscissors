"""
classify.py - Real-Time Rock-Paper-Scissors Classifier (ESP32-S3)
=================================================================
Rubric:
    - "CNN running on the ESP" (20 pts)
    - "CNN running on the ESP able to classify video images in real(ish)
       time with sufficient accuracy, printing classification result on
       the host/PC" (40 pts)

This script runs on the ESP32-S3 Sense microcontroller. It continuously
captures 96x96 grayscale images from the onboard camera, preprocesses
them, runs them through the trained CNN model, and prints the result to
the laptop over USB serial.

Every 3rd frame is also sent as a base64-encoded image so the laptop-side
viewer.py can display a live camera preview.

Model pipeline on ESP32:
    Camera capture -> resize 96x96 to 32x32 -> binary threshold at 128
    -> strip BMP header -> emlearn_cnn_fp32.new().run() -> print prediction

Note: Camera initialization parameters and emlearn inference structure
      assisted by Claude AI (Anthropic) and course-provided examples.
      Classification logic and serial output protocol written by student.
"""

import array       
import gc          # garbage collector, frees memory on the ESP32
import time      
import ubinascii   # encodes image bytes as base64 for serial transmission

from camera import Camera, PixelFormat, FrameSize
from image_preprocessing import resize_96x96_to_32x32_and_threshold
from image_preprocessing import strip_bmp_header
# emlearn_cnn_fp32 is built into the custom ESP32 firmware provided by the course
import emlearn_cnn_fp32 as emlearn_cnn

# path to the tinymaix binary model file on the ESP32 filesystem
MODEL = 'prs_cnn_v2.tmdl'

# class labels 
CLASSES = ['rock', 'paper', 'scissors']

# binary threshold value 
THRESHOLD = 128

# minimum confidence required to print a prediction (85%)
CONFIDENCE = 0.85

# ESP32-S3 Sense OV2640 camera pin configuration
# these pin numbers are fixed by the hardware and must not be changed
CAMERA_PARAMETERS = {
    "data_pins": [15, 17, 18, 16, 14, 12, 11, 48],
    "vsync_pin": 38,
    "href_pin": 47,
    "sda_pin": 40,
    "scl_pin": 39,
    "pclk_pin": 13,
    "xclk_pin": 10,
    "xclk_freq": 20000000,   
    "powerdown_pin": -1,   
    "reset_pin": -1,    
    "frame_size": FrameSize.R96X96,     
    "pixel_format": PixelFormat.GRAYSCALE  
}

cam = Camera(**CAMERA_PARAMETERS)
cam.init()
cam.set_bmp_out(True)   

# load the tinymaix model binary from the ESP32 filesystem into RAM
with open(MODEL, 'rb') as f:
    model_data = array.array('B', f.read()) # 'B' = unsigned byte array

# free unused memory 
gc.collect()

# initialize the emlearn CNN inference engine with the model weights
model = emlearn_cnn.new(model_data)

# signal to the laptop that the model is loaded and ready
print("READY")

current = None   
frame_count = 0  

while True:
    # step 1: capture a raw 96x96 grayscale BMP from the camera
    img = cam.capture()

    # step 2: send a preview image to the laptop every 3rd frame
    # base64 encoding converts binary bytes to safe ASCII text for serial
    frame_count += 1
    if frame_count % 3 == 0:
        encoded = ubinascii.b2a_base64(img).decode('utf-8').strip()
        print(f"IMG:{encoded}")

    # step 3: resize 96x96 -> 32x32 and apply binary threshold at 128
    # this matches exactly what was done during training on the laptop
    processed = resize_96x96_to_32x32_and_threshold(img, THRESHOLD)

    # step 4: strip the BMP header to get raw pixel bytes
    # the model expects only the 32*32 = 1024 pixel values, not the header
    pixels = strip_bmp_header(processed)

    # step 5: wrap pixels in a typed byte array for emlearn
    input_data = array.array('B', pixels)

    # step 6: run inference - fills probs with confidence for each class
    probs = array.array('f', [0.0] * 3)  # 'f' = float32 array, one per class
    model.run(input_data, probs)

    # step 7: find the class with the highest probability (argmax)
    # MicroPython array does not have .index() so we find max manually
    best_idx = 0
    for i in range(1, 3):
        if probs[i] > probs[best_idx]:
            best_idx = i
    best_conf = probs[best_idx]

    # step 8: only print if confidence >= 85% and prediction changed
    if best_conf >= CONFIDENCE:
        pred = CLASSES[best_idx]
        if pred != current:
            current = pred
            # format: PRED:classname:confidence - parsed by viewer.py on laptop
            print(f"PRED:{pred}:{best_conf:.2f}")
    else:
        # confidence too low - reset so next valid prediction will print
        current = None

    # free memory after each frame to prevent memory buildup
    gc.collect()
    time.sleep(0.1)   