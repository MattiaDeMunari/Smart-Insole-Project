import socket
import csv
import datetime
import joblib


# Server settings
HOST = "0.0.0.0"  # Listens on all available interfaces
PORT = 5001  # Must match Arduino's serverPort

# Create a socket (IPv4, TCP)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"Listening for connections on port {PORT}...")

# Accept connection
client_socket, client_address = server_socket.accept()
print(f"Connected to {client_address}")
# touch
label = "not_pressed"
# Open CSV file for writing data
csv_filename = "./saving_data/training_sensor_data_{}.csv".format(label)
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Sensor Value", "label"])  # Write header

    try:
        while True:
            data = client_socket.recv(1024).decode("utf-8").strip()
            if data:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                data = data.split(',')[0]
                print(f"Received: {data}")

                # Save to CSV
                writer.writerow([timestamp, data, label])
                file.flush()  # Ensure data is written immediately
    except KeyboardInterrupt:
        print("Server stopped.")
    finally:
        client_socket.close()
        server_socket.close()
