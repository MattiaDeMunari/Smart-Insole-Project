int sensorPin1 = A2;    // Select the input pin for the pressure sensor (Analog pins)
int sensorValue1 = 0;  // Variable to store the value coming from the sensor

int sensorPin2 = A3;    // 
int sensorValue2 = 0;  //


void setup() {
  Serial.begin(9600); // Connection to USB. Sets the data rate in bits per second (baud) for serial data transmission.
}


void loop() {
  
  sensorValue1 = analogRead(sensorPin1); //Reads the sensor, its is analog range (0 to 1023)
  sensorValue2 = analogRead(sensorPin2); // Uncomment this to add more sensors


  Serial.print(sensorValue1); // Prints the value on the serial monitor
  Serial.print(",");
  Serial.print(sensorValue2); // Uncomment this to add more sensors
  Serial.println(",");

  delay(100); // 100 miliseconds delay in the loop

  if (sensorValue1 > 800) {
    Serial.println("Pressed");
  } else {
    Serial.println("Not pressed");
  }
}
