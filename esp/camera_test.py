"""
camera_test.py - ESP32-S3 Camera Initialization and Still Image Test
=====================================================================
Rubric: "View a still image from the camera" (10 pts)

This script verifies that the ESP32-S3 Sense camera is working correctly.
It initializes the camera with the correct hardware pin configuration,
captures a single still image, saves it to the ESP32 filesystem as
'test.bmp', and prints all available camera settings.

After running this script, download 'test.bmp' from the ESP32 via
Thonny (right-click -> Download to computer) to view the image.

Usage:
    Run directly in Thonny on the ESP32.

Source: Provided by course instructor (Professor, ENMGT 5400).
        Pin configuration and camera API usage from Seeed Studio documentation.
        Student verified camera operation and removed lens cover.
"""

from camera import Camera, GrabMode, PixelFormat, FrameSize, GainCeiling

# these values are fixed by the PCB layout and must not be changed
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
}

cam = Camera(**CAMERA_PARAMETERS)
cam.init()

cam.set_bmp_out(True)

get_methods = [method for method in dir(cam)
               if callable(getattr(cam, method)) and method.startswith("get")]

results = {}
for method in get_methods:
    try:
        result = getattr(cam, method)()
        results[method] = result
    except Exception as e:
        results[method] = f"Error: {e}"


print("Camera 'get' Method Outputs:")
for method, result in results.items():
    print(f"{method}: {result}")

cap = cam.capture()
print(f"Captured Image of size: {len(cap)}")

# save image to ESP32 filesystem
# download via Thonny
with open('test.bmp', 'wb') as f:
    f.write(cap)
print("Saved test.bmp - download via Thonny to view")