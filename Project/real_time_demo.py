import socket   
import csv 
import datetime 
import joblib
import numpy as np
import time  # Import the time module

# Load model and scaler
model = joblib.load('knn_model_filtered.pkl')  
scaler = joblib.load('scaler_knn_filtered.pkl')

# Dictionary for label mapping
label_map = {
    0: "Normal Standing",
    1: "Heel Tapping",
    2: "Toe Tapping",
    3: "Standing in one foot",
    4: "Sitting",
    5: "Standing on the other foot",
    6: "Walking",
    7: "Jumping",
    8: "Movement not classified"
}

# Server setup
HOST = "0.0.0.0"  
PORT = 5001  

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  
server_socket.bind((HOST, PORT))  
server_socket.listen(1)

print(f"Listening for connections on port {PORT}...")  

client_socket, client_address = server_socket.accept()
print(f"Connected to {client_address}")

# Open CSV file for writing
csv_filename = "C:\\Users\\Mattia\\Desktop\\Smart Werables\\Project\\data_collection\\Demo\\data_predicted.csv"
with open(csv_filename, mode="w", newline="") as file:  
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Sensor1", "Sensor2", "Predicted Activity"])  

    last_time = time.time()  # Save the last time data was processed

    try:
        while True:
            data = client_socket.recv(2048).decode("utf-8").strip()
            current_time = time.time()  # Get the current time

            # Process data every second
            if current_time - last_time >= 1:  
                if data:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"Received: {data}")  

                    # Data cleaning
                    clean_data = data.replace("\n", "").replace("\r", "").strip()
                    values = clean_data.split(",")

                    # Check if there are at least 2  valid values
                    if len(values) < 2:
                        print("Incomplete data received, ignored.")
                        continue

                    try:
                        sensor_values = np.array([[float(values[0]), float(values[1])]])  # Converti in float
                    except ValueError:
                        print(f"Error in data conversion: {values}, ignored.")
                        continue  
                    
                    # Normalizes the data
                    sensor_values = scaler.transform(sensor_values)
                    
                    # Predicts the activity and converts the result to an integer
                    prediction = model.predict(sensor_values)[0]

                    # if the prediction is a string, convert it to the corresponding integer
                    if isinstance(prediction, str):
                        prediction = list(label_map.keys())[list(label_map.values()).index(prediction)]
                    else:
                        prediction = int(prediction)

                    # get the corresponding label from the dictionary
                    predicted_label = label_map.get(prediction, "Unknown Activity")
                    
                    print(f"Predicted Activity: {predicted_label}")  

                    # Save to CSV
                    writer.writerow([timestamp] + values[:2] + [predicted_label])  
                    file.flush()  

                last_time = current_time  # Update the last time data was processed

    except KeyboardInterrupt:  
        print("Server stopped.")  
    finally:  
        client_socket.close()  
        server_socket.close()
