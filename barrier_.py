import serial
import sys
import time

ser = serial.Serial('/dev/ttyUSB1', 9600)

while True:
    c = input("> ")
    print('c', c)
    if c == '1':
        # open
        print('hiu')
        ser.write(b'o')
        time.sleep(0.5)   # Delays for 5 seconds. You can also use a float value.
        ser.write(b'c')