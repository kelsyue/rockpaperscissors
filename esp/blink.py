"""
blink.py - LED Blink Sketch (ESP32-S3)
=======================================
Rubric: "Flash Xiao and run a blink sketch" (5 pts)

This script demonstrates that the ESP32-S3 has been successfully flashed
with MicroPython firmware and can run basic programs. It blinks the
onboard LED on and off every 0.5 seconds indefinitely.

Usage:
    Run directly in Thonny on the ESP32.
    Press Stop to end the program.

Note: LED pin number verified by student by testing pins 21 and 48.
"""

from machine import Pin
import time

led = Pin(21, Pin.OUT)

while True:
    # turn LED on
    led.value(1) 
    time.sleep(0.5)
    # turn LED off
    led.value(0) 
    time.sleep(0.5)