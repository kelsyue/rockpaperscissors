"""
viewer.py - Live Camera Stream Viewer for Rock-Paper-Scissors Classifier
=========================================================================
Rubric: "CNN running on the ESP able to classify video images in real(ish)
        time with sufficient accuracy, printing classification result on
        the host/PC" (40 pts)

This script runs on the LAPTOP and connects to the ESP32 over USB serial.
It reads the live camera stream sent by classify.py on the ESP32 and
displays it in a matplotlib window. When the CNN makes a prediction with
>= 85% confidence, the result is shown in large colored text on screen.

Usage:
    1. Run classify.py on the ESP32 in Thonny first
    2. Close Thonny completely (it blocks the serial port)
    3. Run: python3 viewer.py

Note: Code structure and matplotlib live-update pattern assisted by Claude AI (Anthropic).
      Serial communication protocol designed by student.
"""

# imports
import serial      
import base64     
import glob         
import sys    
import io      
import numpy as np                    
import matplotlib                      
matplotlib.use('MacOSX')       
import matplotlib.pyplot as plt    
from PIL import Image         


def find_port():
    """
    Automatically finds the ESP32's serial port on macOS.

    The ESP32 shows up as /dev/cu.usbmodem* when connected via USB.
    Returns the first matching port found, or exits if none found.

    Returns:
        str: Path to the serial port e.g. '/dev/cu.usbmodem3101'
    """
    ports = glob.glob('/dev/cu.usbmodem*')
    if ports:
        return ports[0]
    print("No ESP32 found!")
    sys.exit(1)


# serial connection 
PORT = find_port()
print(f"Connecting to {PORT}...")

try:
    # Open serial connection at 115200 baud - must match ESP32 setting
    # timeout=0.5 means readline() waits up to 0.5 seconds for data
    ser = serial.Serial(PORT, 115200, timeout=0.5)
except Exception as e:
    print(f"Could not open port: {e}")
    print("Close Thonny first!")
    sys.exit(1)


# matplot setup stream 
# Create a 6x6 inch figure with dark background to match camera preview
fig, ax = plt.subplots(figsize=(6, 6))
ax.axis('off')                      
fig.patch.set_facecolor('#1a1a1a')     
ax.set_facecolor('#1a1a1a')        

dummy = np.zeros((96, 96))
im = ax.imshow(dummy, cmap='gray', vmin=0, vmax=255,
               interpolation='bilinear', aspect='equal')

title = ax.set_title('Waiting...', fontsize=22,
                     fontweight='bold', color='white', pad=15)

plt.tight_layout()
plt.ion()     
plt.show()


# prediction displays
# each class gets a distinct color
PRED_COLORS = {
    'rock':     '#FF6B6B',  
    'paper':    '#4ECDC4',  
    'scissors': '#FFE66D', 
}

# emoji + label shown in the title when a prediction is made
PRED_EMOJI = {
    'rock':     '✊  ROCK',
    'paper':    '✋  PAPER',
    'scissors': '✌️  SCISSORS',
}

print("Viewer running!")
print("Press Ctrl+C to stop.\n")


# continuously read lines from the ESP32 serial output.
# the ESP32 sends two types of messages:
#   IMG:<base64>  - a raw 96x96 camera frame encoded in base64
#   PRED:<class>:<confidence> - a classification result
#   READY         - sent once when the model finishes loading
try:
    while True:
        # read one line from serial (blocks up to timeout=0.5s)
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line:
            continue

        if line.startswith('IMG:'):
            # decode base64 image and display it in the window
            try:
                b64 = line[4:]                            
                img_bytes = base64.b64decode(b64)     
                img = Image.open(io.BytesIO(img_bytes)).convert('L') 
                arr = np.array(img)                     
                im.set_data(arr)                        
                fig.canvas.draw()                         
                fig.canvas.flush_events()                 
            except:
                pass  # skip malformed frames silently

        elif line.startswith('PRED:'):
            # Parse prediction and update the title with colored text
            parts = line.split(':')
            if len(parts) >= 3:
                pred = parts[1]                        
                conf = float(parts[2])            
                color = PRED_COLORS.get(pred, 'white')
                label = PRED_EMOJI.get(pred, pred.upper())
                title.set_text(f"{label}  {conf*100:.0f}%") 
                title.set_color(color)
                fig.canvas.draw()
                fig.canvas.flush_events()
                print(f"→ {label} {conf*100:.0f}%")  

        elif line == 'READY':
            # ESP32 finished loading the model and is ready to classify
            title.set_text('Show your hand!')
            title.set_color('white')
            fig.canvas.draw()
            fig.canvas.flush_events()
            print("ESP32 ready!")

except KeyboardInterrupt:
    print("\nStopped.")
    ser.close()
    plt.close()