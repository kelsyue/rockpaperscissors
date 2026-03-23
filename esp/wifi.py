"""
Wifi.py - Wi-Fi Station Connection Manager (ESP32-S3)
=====================================================
Rubric: "Connect to Wi-Fi" (5 pts)

This module provides a simple Wi-Fi connection class for MicroPython on
the ESP32-S3. It connects the ESP32 to a Wi-Fi access point (station mode)
and provides methods to check connection status, wait for connection, and
scan for available networks.

Used by classify.py and stream_server.py to establish Wi-Fi before
starting the camera stream or connecting to the laptop.

Source: Provided by course instructor (Professor, ENMGT 5400).
        Original author: Sharil Tumin (MIT License).
        SSID and password configured by student for personal hotspot.

Usage:
    from Wifi import Sta
    wif = Sta()
    wif.connect()
    wif.wait()    # blocks until connected, prints IP address
"""

from time import sleep
import network


class Sta:
    AP = "Jelly iPhone"     # your Wi-Fi network name (SSID)
    PWD = ""                # your Wi-Fi password, i not giving it 2 u

    def __init__(my, ap='', pwd=''):
        """
        Initializes the Wi-Fi station interface.

        Disables the access point mode (AP_IF) since we only need
        station mode (STA_IF) to connect to an existing network.

        Args:
            ap (str): optional SSID override. uses class default if empty.
            pwd (str): optional password override. uses class default if empty.
        """
        network.WLAN(network.AP_IF).active(False)
        sleep(1)

        my.wlan = network.WLAN(network.STA_IF)
        my.wlan.active(True)

        if ap == '':
            my.ap = Sta.AP
            my.pwd = Sta.PWD
        else:
            my.ap = ap
            my.pwd = pwd

    def connect(my, ap='', pwd=''):
        """
        Initiates a Wi-Fi connection attempt.

        Does nothing if already connected. Non-blocking - use wait()
        to block until connection is established.

        Args:
            ap (str): optional SSID override
            pwd (str): optional password override
        """
        if ap != '':
            my.ap = ap
            my.pwd = pwd
        if not my.wlan.isconnected():
            my.wlan.connect(my.ap, my.pwd)

    def status(my):
        """
        Returns the current network configuration if connected.

        Returns:
            tuple: (ip, subnet, gateway, dns) if connected, empty tuple if not
        """
        if my.wlan.isconnected():
            return my.wlan.ifconfig()
        else:
            return ()

    def wait(my):
        """
        Blocks until connected or timeout (30 seconds).

        Prints 'Waiting...' every 5 seconds until connected, then prints
        the IP address. Used to confirm Wi-Fi is working before proceeding.
        """
        cnt = 30
        while cnt > 0:
            print("Waiting...")
            if my.wlan.isconnected():
                print("Connected to %s" % my.ap)
                print('network config:', my.wlan.ifconfig())
                cnt = 0  
            else:
                sleep(5)
                cnt -= 5

    def scan(my):
        """
        Scans for available Wi-Fi networks.

        Returns:
            list: list of tuples (ssid, bssid, channel, rssi, authmode, hidden)
        """
        return my.wlan.scan()