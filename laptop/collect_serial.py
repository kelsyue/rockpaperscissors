"""
collect_serial.py - Live Image Collection from ESP32 Camera via USB Serial
===========================================================================
Rubric: "Code running on the ESP able to collect images from the camera
        in real time / as a stream and store them" (20 pts)

This script runs on the LAPTOP and connects to the ESP32 over USB serial.
It receives raw 96x96 grayscale BMP images streamed from the ESP32 camera
in real time, displays them in a live matplotlib preview window, and saves
them to a labeled folder for use as training data.

The live preview lets the user see exactly what the camera sees during
collection, ensuring correct hand positioning and image quality.

Usage:
    1. Set CLASS to 'rock', 'paper', or 'scissors'
    2. Run stream_server.py on the ESP32 in Thonny first
    3. Run: python3 collect_serial.py

Note: Serial reading loop and matplotlib live preview structure assisted by
      Claude AI (Anthropic). Dataset folder logic and BMP saving written by student.
"""

import serial          
import os            
import matplotlib.pyplot as plt  
import numpy as np     
t
# find serial port with: ls /dev/cu.usbmodem*
PORT = '/dev/cu.usbmodem3101'
BAUD = 115200           # must match ESP32 serial baud rate

CLASS = 'paper'         # options: 'rock', 'paper', 'scissors'

# folder where collected images will be saved
SAVE_DIR = f'dataset/{CLASS}'
os.makedirs(SAVE_DIR, exist_ok=True)

# BMP header size for 96x96 8-bit grayscale with 256-color palette
# 14 (BMP header) + 40 (DIB header) + 1024 (palette) = 1078 bytes
HEADER_SIZE = 1078

# image dimensions from the ESP32 camera
IMG_W, IMG_H = 96, 96

# how many images to collect this session
TARGET = 50

# save every Nth frame to reduce redundancy in fast streams
SAVE_EVERY = 2

# continue numbering from existing images to avoid overwriting
existing = len([f for f in os.listdir(SAVE_DIR) if f.endswith('.bmp')])
print(f"Found {existing} existing images in {SAVE_DIR}/")
print(f"Will collect {TARGET} more (saving every {SAVE_EVERY} frames), then stop.\n")

ser = serial.Serial(PORT, BAUD, timeout=2)

count = 0   # images saved this session
frame = 0   # total frames received this session

# set up live preview window
fig, ax = plt.subplots(figsize=(4, 4))
placeholder = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
im = ax.imshow(placeholder, cmap='gray', vmin=0, vmax=255)
ax.axis('off')
plt.ion()  
plt.show()

try:
    while count < TARGET:
        # read one line from the ESP32 serial output
        line = ser.readline().decode('utf-8', errors='ignore').strip()

        if not line.startswith("IMG:"):
            if line:
                print(f"[ESP]: {line}")
            continue

        # parse hex-encoded image data sent after 'IMG:' prefix
        try:
            img_bytes = bytes.fromhex(line[4:])
        except ValueError:
            print("Bad hex frame, skipping")
            continue

        frame += 1

        # extract raw pixel data by skipping the BMP header
        pixel_data = img_bytes[HEADER_SIZE:]

        # update live preview if we have enough pixel data
        if len(pixel_data) >= IMG_W * IMG_H:
            # reshape flat bytes into 2D image array for display
            img_array = np.frombuffer(
                pixel_data[:IMG_W * IMG_H], dtype=np.uint8
            ).reshape((IMG_H, IMG_W))

            im.set_data(img_array)
            ax.set_title(
                f"Class: {CLASS} | Saved: {count}/{TARGET} | Total: {existing + count}",
                fontsize=10
            )
            plt.pause(0.001)  

        # only save every SAVE_EVERY frames to avoid redundant data
        if frame % SAVE_EVERY != 0:
            continue

        # save full BMP including header to the class folder
        filepath = f'{SAVE_DIR}/{existing + count:04d}.bmp'
        with open(filepath, 'wb') as f:
            f.write(img_bytes)
        count += 1

        print(f"Collecting: {count}/{TARGET} (total: {existing + count})", end='\r')

except KeyboardInterrupt:
    print(f"\nStopped early. Collected {count} images.")

finally:
    print(f"\nDone! {SAVE_DIR}/ now has {existing + count} images total.")
    ser.close()
    plt.close()