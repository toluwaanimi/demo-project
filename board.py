import serial

# Define the serial port and baud rate
serial_port = '/dev/ttyUSB0'  # Change this to match your Arduino's serial port
baud_rate = 115200

try:
    # Open the serial port
    ser = serial.Serial(serial_port, baud_rate)

    while True:
        # Read a line of data from the Arduino
        data = ser.readline().decode().strip()

        # Check if the received data contains "Heart rate"
        if "Heart rate" in data:
            # Split the data to extract the heart rate value
            _, heart_rate_value = data.split(": ")
            heart_rate = int(heart_rate_value)
            
            # Display the heart rate value
            print(f"Heart rate: {heart_rate} bpm")

except serial.SerialException as e:
    print(f"Error: {e}")

finally:
    if ser.is_open:
        ser.close()
