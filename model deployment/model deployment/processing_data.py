import socket
import joblib
import numpy as np
import datetime
from collections import deque

# Server settings
HOST = "0.0.0.0"  # Listens on all available interfaces
PORT = 5001  # Must match Arduino's serverPort

# Load the trained SVM model
svm_model = joblib.load("svm_model.pkl")

# Define window parameters
WINDOW_LENGTH = 20
CLASSIFICATION_INTERVAL = 20
window_buffer = deque(maxlen=WINDOW_LENGTH)  # Circular buffer to store latest sensor values
received_count = 0

# Create a socket (IPv4, TCP)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"Listening for connections on port {PORT}...")

# Accept connection
client_socket, client_address = server_socket.accept()
print(f"Connected to {client_address}")

try:
    while True:
        data = client_socket.recv(1024).decode("utf-8").strip()
        if data:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data = data.split(',')[0]
            print(f"Received: {data}")

            try:
                # Convert received data to float
                sensor_value = float(data)

                # Append to the window buffer
                window_buffer.append(sensor_value)
                received_count += 1

                # Make a prediction every 20 received values if we have enough data
                if received_count % CLASSIFICATION_INTERVAL == 0 and len(window_buffer) == WINDOW_LENGTH:
                    window_array = np.array(window_buffer)

                    # Extract features
                    # mean_val = np.mean(window_array)
                    # var_val = np.var(window_array)
                    # min_val = np.min(window_array)
                    # max_val = np.max(window_array)
                    mean = np.mean(window_array)
                    std = np.std(window_array)
                    min = np.min(window_array)
                    max = np.max(window_array)
                    ptp = np.ptp(window_array)
                    rms = np.sqrt(np.mean(window_array ** 2))

                    feature_vector = np.array([mean, std, min, max, ptp, rms]).reshape(1, -1)

                    # input_data = np.array(window_buffer).reshape(1, -1)  # Reshape for SVM
                    prediction = svm_model.predict(feature_vector)[0]

                    label = "PRESSED" if prediction == 1 else "NOT PRESSED"
                    print(f"[{timestamp}] Prediction: {label}")

            except ValueError:
                print("Invalid sensor value received.")
except KeyboardInterrupt:
    print("Server stopped.")
finally:
    client_socket.close()
    server_socket.close()
