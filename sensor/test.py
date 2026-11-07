# pip install pyserial
# Live-streams Grove loudness sensor readings from the Uno Q. Ctrl+C to stop.
import serial, time, sys

PORT = 'COM5'
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)  # the board resets when the port opens
ser.reset_input_buffer()

vals = []
last = time.time()
print(f"Listening on {PORT}... (Ctrl+C to stop)")
try:
    while True:
        line = ser.readline().decode(errors='ignore').strip()
        if not line:
            if time.time() - last > 3:
                print("No data for 3s - is the loudness sketch uploaded and Serial Monitor closed?")
                last = time.time()
            continue
        try:
            v = int(line)
        except ValueError:
            print("?", line)
            continue
        last = time.time()
        vals.append(v)
        vals = vals[-50:]
        bar = '#' * (v * 50 // 1023)
        print(f"{v:4d} (avg {sum(vals)//len(vals):4d}) |{bar}")
except KeyboardInterrupt:
    pass
finally:
    ser.close()
