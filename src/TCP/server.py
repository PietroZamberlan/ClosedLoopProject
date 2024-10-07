import socket
import ssl
import time

# MacBookPro ip adress: 172.17.12.112
def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(( '0.0.0.0', 65432))  # Bind to all interfaces on port 65432
    # server_socket.bind(( '172.17.12.112', 65432))  # Bind to all interfaces on port 65432

    server_socket.listen(1)
    print("Server is listening on port 65432...")

    while True:
        client_socket, addr = server_socket.accept()
        server_time = time.time()
        print(f"Connection from {addr}")
        data = client_socket.recv(1024)
        if not data:
            break

        # Process the received array (convert bytes to list of integers)
        array = list(map(int, data.decode('utf-8').split(',')))
        print(f"Received array: {array}")
       # Perform computations (for demonstration, we'll just sum the array)
        result = sum(array)

        # if the process receives a float
        # received_float = float(data.decode('utf-8'))
        # result = received_float - server_time

        print(f"Computed result: {result} seconds of difference between the two machines.")

        # Send the result back to Machine 1p
        # client_socket.sendall(str(result).encode('utf-8'))
        client_socket.sendall(str(result).encode('utf-8'))
        client_socket.close()

if __name__ == "__main__":
    start_server()