"""
Test if we can access the loopback address.
"""
import socket

try:
    # Create a socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Connect to the loopback address
    s.connect(('127.0.0.1', 8080))
    
    print("Successfully connected to 127.0.0.1:8080")
    
    # Close the socket
    s.close()
except Exception as e:
    print(f"Error: {str(e)}")