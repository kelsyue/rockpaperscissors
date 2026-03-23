# Rock, Paper, Scissors — ESP32-S3 Sense CNN Classifier

**ENMGT 5400 · Project 4 · Kelly Yue**

A real-time hand gesture classifier running on the Seeed Xiao ESP32-S3 Sense. A CNN is trained on the laptop and deployed to the microcontroller, which captures live camera frames, preprocesses them, runs inference, and prints the classification result over USB serial.

---

## Rubric Checklist

| Requirement | Where |
|---|---|
| Flash Xiao + blink sketch | `esp/blink.py` |
| Connect to Wi-Fi | `esp/wifi.py` (provided by professor) |
| View still image from camera | `esp/camera_test.py` |
| Real-time image capture/stream from ESP | `esp/classify.py` streams frames → `laptop/collect_serial.py` saves them |
| Basic CNN on local machine | `laptop/preprocess_and_train.py` |
| CNN classifying RPS > 50% accuracy | `laptop/preprocess_and_train.py` — final val accuracy: **90.1%** |
| CNN running on the ESP | `esp/classify.py` |
| Dataset ≥ 300 images | `dataset/` — 905 images total (rock: 351, paper: 351, scissors: 203) |
| Real-time classification + demo | `esp/classify.py` + `laptop/viewer.py` |
| Submitted code with comments | All files commented |
| Documentation | `ENMGT5400_Project4_Report.pdf` |

---

## Repository Structure

```
.
├── esp/
│   ├── blink.py                  # Blink sketch — verifies firmware and LED
│   ├── wifi.py                   # Wi-Fi connection (provided by professor)
│   ├── camera_test.py            # Capture still image, verify camera pins
│   ├── image_preprocessing.py    # Preprocessing helpers (provided by professor)
│   └── classify.py               # Main ESP32 script: capture → preprocess → infer → stream
│
├── laptop/
│   ├── collect_serial.py         # Receive frames over USB serial, save to dataset/
│   ├── preprocess_and_train.py   # Data loading, augmentation, CNN training
│   ├── convert_model.py          # Convert Keras .h5 → TinyMaix .tmdl
│   ├── test_model.py             # Offline accuracy testing
│   └── viewer.py                 # Display live classification results from serial
│
├── dataset/
│   ├── rock/                     # 351 training images
│   ├── paper/                    # 351 training images
│   └── scissors/                 # 203 training images
│
└── model/
    └── prs_cnn_v2.tmdl           # Converted TinyMaix model (deployed to ESP32)
```

---

## Hardware

- **Board:** Seeed Xiao ESP32-S3 Sense
- **Connection:** USB-C (power + serial)

---

## Setup & Usage

### 1. Flash Firmware
Flash the provided MicroPython firmware onto the ESP32 using Thonny. Verify with `esp/blink.py`.

### 2. Connect to Wi-Fi
Edit `esp/wifi.py` with your SSID and password. A phone hotspot is recommended — iPhone hotspots and institutional networks with AP Isolation will block peer-to-peer traffic between the ESP32 and laptop.

### 3. View a Still Image
Run `esp/camera_test.py` on the device to capture a test frame and confirm the camera is working. Copy the image to your laptop via Thonny to inspect it.

### 4. Collect Training Data
Run `esp/classify.py` on the ESP32 first to start streaming frames, then on your laptop:
```bash
python3 laptop/collect_serial.py
```
Set the `CLASS` variable in `collect_serial.py` to `'rock'`, `'paper'`, or `'scissors'` before each session. Images are saved to the corresponding `dataset/` subfolder with a live preview window. Review and delete blurry or partial frames after collection.

### 5. Train the Model
```bash
python3 laptop/preprocess_and_train.py
```
Trains the CNN on your laptop with data augmentation and class weights. Final validation accuracy: **90.1%**.

### 6. Convert Model for ESP32
```bash
python3 laptop/convert_model.py
```
Converts the trained Keras model → `prs_cnn_v2.tmdl` using the emlearn TinyMaix converter. Copy the `.tmdl` file to the ESP32 filesystem via Thonny.

### 7. Run Real-Time Classification
Run `esp/classify.py` on the ESP32. On your laptop:
```bash
python3 laptop/viewer.py
```
The predicted gesture and confidence score print to the terminal at roughly 1–2 fps.

---

## CNN Architecture

| Layer | Output Shape | Purpose |
|---|---|---|
| Conv2D(16, 3×3, stride=2) | 16×16×16 | Detect edges and shapes |
| Conv2D(32, 3×3, stride=2) | 8×8×32 | Learn feature combinations |
| Conv2D(64, 3×3, stride=2) | 4×4×64 | Learn high-level hand patterns |
| Flatten | 1024 | Convert feature maps to 1D |
| Dropout(0.4) | 1024 | Prevent overfitting |
| Dense(32, ReLU) | 32 | Combine all features |
| Dense(3, Softmax) | 3 | Output class probabilities |

MaxPooling is not used because the TinyMaix inference engine on the ESP32 does not support it. Strided convolutions (stride=2) achieve the same spatial downsampling.

---

## Preprocessing Pipeline

Both the laptop (training) and ESP32 (inference) apply the same steps in order:

1. Capture 96×96 grayscale BMP from camera
2. Resize to 32×32 using nearest-neighbor downsampling (take every 3rd pixel)
3. Apply binary threshold at 128 (≥128 → white/hand, <128 → black/background)
4. Strip the 1078-byte BMP file header
5. Pass the remaining 1023 raw pixel bytes to the model as `array.array('B')`