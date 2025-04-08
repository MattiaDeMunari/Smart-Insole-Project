#include <SPI.h>
#include <WiFiNINA.h>

// WiFi credentials
const char* ssid = "zyyyyy";     // Replace with your WiFi SSID
const char* password = "zy561526"; // Replace with your WiFi password

// Server details (your computer's IP and port)
const char* serverIP = "172.20.10.2";  // Replace with your computer's local IP
const int serverPort = 5001;  // Must match the Python script

int sensorPin1 = A2;
int sensorValue1 = 0;

WiFiClient client;

void setup() {
    Serial.begin(115200);
    while (!Serial);

    // Connect to WiFi
    Serial.print("Connecting to WiFi...");
    while (WiFi.begin(ssid, password) != WL_CONNECTED) {
        Serial.print(".");
        delay(1000);
    }
    Serial.println("Connected!");

    // Connect to the server
    Serial.print("Connecting to server...");
    while (!client.connect(serverIP, serverPort)) {
        Serial.print(".");
        delay(1000);
    }
    Serial.println("Connected to server!");
}

void loop() {
    if (client.connected()) {
        sensorValue1 = analogRead(sensorPin1);  // Read from a sensor (adjust as needed)
        String data = String(sensorValue1 )+ ",";  // Convert to string
        client.print(data);  // Send data
        Serial.print("Sent: ");
        Serial.println(data);
        delay(100);  // Adjust sending rate
    } else {
        Serial.println("Disconnected, reconnecting...");
        client.stop();
        delay(2000);
        client.connect(serverIP, serverPort);
    }
}