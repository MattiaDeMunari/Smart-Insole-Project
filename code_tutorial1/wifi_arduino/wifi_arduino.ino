#include <SPI.h>
#include <WiFiNINA.h>

// WiFi credentials
const char* ssid = "iPhone di Mattia";     // Replace with your WiFi SSID
const char* password = "Mattidemu"; // Replace with your WiFi password

// Server details (your computer's IP and port)
const char* serverIP = "172.20.10.3";  // Replace with your computer's local IP/ the IP after your computer connect to the hotspot
const int serverPort = 5001;  // Must match the server Python script

int sensorPin1 = A1;
int sensorValue1 = 0;
int sensorPin2 = A2;
int sensorValue2 = 0;
//const int buttonPin = 2;

WiFiClient client;

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 5000);  // Aspetta max 5 secondi, poi continua
    //while (!Serial);

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
        sensorValue2 = analogRead(sensorPin2); 
        // int test_value = digitalRead(buttonPin);
        //int label = 7;  //activity gesture that we want to detect; Remember to change every time before start the recognition of a different activity 
        unsigned long timestamp = millis();
        String data = String(sensorValue1) + "," + String(sensorValue2) + "\n";
        //String data = String(sensorValue1) + "," + String(sensorValue2) + "," + String(label) + "\n";  // Convert to string，typo in previous version code
        
        client.print(data);  // Send data
        Serial.print("Sent: ");
        Serial.print(data);
        
        delay(150);  // Adjust sending rate/frequency -> campionamento a 50Hz (1000ms / 6.66 = 150ms) 1 sample every 150ms
    } else {
        Serial.println("Disconnected, reconnecting...");
        client.stop();
        delay(2000);
        client.connect(serverIP, serverPort);
    }
}
