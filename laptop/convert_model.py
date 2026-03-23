"""
convert_model.py - Convert Keras Model to TinyMaix .tmdl Format
================================================================
Rubric: "CNN running on the ESP" (20 pts)

This script converts the trained Keras model through two steps:
    prs_cnn.h5 -> prs_cnn.tflite (done in training script)
    prs_cnn.tflite -> prs_cnn_v2.tmdl (done here)

The .tmdl file is a compact binary format loaded by the emlearn_cnn_fp32
module on the ESP32. It contains the model weights packed in TinyMaix format.

Conversion parameters:
    fp32  - 32-bit float weights, matches emlearn_cnn_fp32.mpy on ESP32
    1     - enable output dequantization
    32,32,1 - input shape (height, width, channels)
    3     - number of output classes (rock, paper, scissors)

Usage:
    python3 convert_model.py

Note: Conversion command and TinyMaix parameter format assisted by
      Claude AI (Anthropic).
      Integration and debugging done by student.
"""

import os
import subprocess
import sys

# input: tflite model produced by the training script
TFLITE_MODEL = 'prs_cnn.tflite'

# output: tinymaix binary model to be uploaded to the ESP32
TMDL_MODEL = 'prs_cnn_v2.tmdl'

# path to the tinymaix converter inside the cloned emlearn-micropython repo
CONVERTER_SCRIPT = 'emlearn-micropython/dependencies/TinyMaix/tools/tflite2tmdl.py'


def convert():
    """
    Converts prs_cnn.tflite to prs_cnn_v2.tmdl using the TinyMaix converter.

    The TinyMaix .tmdl format packs model weights into a binary file that
    the emlearn_cnn_fp32 MicroPython module can load into ESP32 RAM and run
    for inference without needing the full TensorFlow runtime.

    Raises:
        SystemExit if required files are missing or conversion fails
    """
    # check that the input tflite model exists
    if not os.path.exists(TFLITE_MODEL):
        print(f"Error: {TFLITE_MODEL} not found.")
        print("Run preprocess_and_train.py first.")
        return

    # check that the tinymaix converter script exists
    if not os.path.exists(CONVERTER_SCRIPT):
        print(f"Error: converter not found at {CONVERTER_SCRIPT}")
        print("Clone emlearn-micropython and run: git submodule update --init")
        return

    print(f"Converting {TFLITE_MODEL} -> {TMDL_MODEL}...")

    # build the conversion command
    # in_dims uses comma-separated format required by this version of tinymaix
    cmd = [
        sys.executable,    
        CONVERTER_SCRIPT,
        TFLITE_MODEL,      # input tflite model
        TMDL_MODEL,        # output binary model
        'fp32',       
        '1',               
        '32,32,1',         # input shape: 32x32 grayscale image
        '3',               # output: 3 classes (rock, paper, scissors)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)

        if result.returncode != 0:
            print("Conversion failed:")
            print(result.stderr)
            return

    except Exception as e:
        print(f"Failed to run converter: {e}")
        return

    if os.path.exists(TMDL_MODEL):
        size = os.path.getsize(TMDL_MODEL)
        print(f"\nSUCCESS: {TMDL_MODEL} created ({size:,} bytes / {size//1024}KB)")
        print(f"Upload {TMDL_MODEL} to the ESP32 via Thonny to deploy.")
    else:
        print("FAILED: no output file was created.")


if __name__ == "__main__":
    convert()