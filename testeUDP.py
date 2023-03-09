
#? UDP Sender Script:

import socket

UDP_IP = "127.0.0.1"  # Replace with the IP address of the receiver
UDP_PORT = 5005      # Replace with a port number of your choice

# Create a socket object for sending data
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Send data over UDP
message = "Hello, receiver!"
sock.sendto(message.encode(), (UDP_IP, UDP_PORT))

# Close the socket
sock.close()


#? UDP Receiver Script:

import socket

UDP_IP = "127.0.0.1"  # IP address of this machine
UDP_PORT = 5005      # Port number to listen on

# Create a socket object for receiving data
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to the port
sock.bind((UDP_IP, UDP_PORT))

# Receive data over UDP
data, addr = sock.recvfrom(1024)
print("Received message: ", data.decode())

# Close the socket
sock.close()