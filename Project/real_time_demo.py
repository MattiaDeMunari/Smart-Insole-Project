import socket   
import csv 
import datetime 
import joblib
import numpy as np
import time  # Importa il modulo time

# Carica il modello e lo scaler
model = joblib.load('knn_model_filtered.pkl')  
scaler = joblib.load('scaler_knn_filtered.pkl')

# Dizionario per la decodifica delle etichette
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

# Impostazioni server
HOST = "0.0.0.0"  
PORT = 5001  

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  
server_socket.bind((HOST, PORT))  
server_socket.listen(1)

print(f"Listening for connections on port {PORT}...")  

client_socket, client_address = server_socket.accept()
print(f"Connected to {client_address}")

# Apri il file CSV per salvare i dati con predizione
csv_filename = "C:\\Users\\Mattia\\Desktop\\Smart Werables\\Project\\data_collection\\Demo\\data_predicted.csv"
with open(csv_filename, mode="w", newline="") as file:  
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Sensor1", "Sensor2", "Predicted Activity"])  

    last_time = time.time()  # Salva il tempo dell'ultima elaborazione

    try:
        while True:
            data = client_socket.recv(2048).decode("utf-8").strip()
            current_time = time.time()  # Tempo corrente

            # Elabora i dati solo se è passato almeno 1 secondo
            if current_time - last_time >= 1:  # Intervallo di 1 secondo
                if data:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"Received: {data}")  

                    # Pulizia dei dati: rimuove spazi, a capo e caratteri non numerici
                    clean_data = data.replace("\n", "").replace("\r", "").strip()
                    values = clean_data.split(",")

                    # Controllo se ci sono almeno 2 valori validi
                    if len(values) < 2:
                        print("⚠️ Dati incompleti ricevuti, ignorati.")
                        continue

                    try:
                        sensor_values = np.array([[float(values[0]), float(values[1])]])  # Converti in float
                    except ValueError:
                        print(f"⚠️ Errore nella conversione dei dati: {values}, ignorati.")
                        continue  # Ignora i dati errati
                    
                    # Normalizza i dati
                    sensor_values = scaler.transform(sensor_values)
                    
                    # Predice l'attività e converte in intero
                    prediction = model.predict(sensor_values)[0]

                    # Se il modello restituisce una stringa invece di un intero, convertilo
                    if isinstance(prediction, str):
                        prediction = list(label_map.keys())[list(label_map.values()).index(prediction)]
                    else:
                        prediction = int(prediction)

                    # Ottieni la label dal dizionario
                    predicted_label = label_map.get(prediction, "Unknown Activity")
                    
                    print(f"Predicted Activity: {predicted_label}")  

                    # Salva nel CSV
                    writer.writerow([timestamp] + values[:2] + [predicted_label])  
                    file.flush()  

                last_time = current_time  # Aggiorna il tempo dell'ultima elaborazione

    except KeyboardInterrupt:  
        print("Server stopped.")  
    finally:  
        client_socket.close()  
        server_socket.close()
