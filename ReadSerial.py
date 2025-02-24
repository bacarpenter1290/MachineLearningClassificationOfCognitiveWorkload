##############
## Script listens to serial port and writes contents into a file
##############
## requires pySerial to be installed 
import serial  # sudo pip install pyserial should work

serial_port = 'COM6';
baud_rate = 9600; #In arduino, Serial.begin(baud_rate)
write_to_file_path = "output2.txt";


ser = serial.Serial(serial_port, baud_rate)
try:
    while True:
        output_file = open(write_to_file_path, "w");
        line = ser.readline();
        line = line.decode("utf-8") #ser.readline returns a binary, convert to string
        print(line);
        output_file.write(line);
except KeyboardInterrupt:
    ser.close();
    pass