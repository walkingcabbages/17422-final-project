#include <SparkFunMMA8452Q.h>
#include "ICM_20948.h"
#define SERIAL_PORT Serial
#define WIRE_PORT Wire
#define AD0_VAL_1 1  
TCPClient client; // TCP Client
byte server[] = {172,26,23,208}; // TODO: Server IP, separated by commas instead of periods
char buffer[256]; // rx buffer
int framesToRead = 0; // samples remaining
String msg = ""; // tx buffer

ICM_20948_I2C myICM1; // Instance for first sensor
MMA8452Q accel; // MMA8452 QWIIC Accelerometer
int led = D7; // on-board LED


void setup() {
  SERIAL_PORT.begin(115200);
  while (!SERIAL_PORT);

  WIRE_PORT.begin();
  WIRE_PORT.setClock(400000);

  myICM1.begin(WIRE_PORT, AD0_VAL_1);
  SERIAL_PORT.print("Initialization of sensor 1 returned: ");
  SERIAL_PORT.println(myICM1.statusString());
  pinMode(led, OUTPUT);

    if (!accel.begin(SCALE_2G, ODR_50)) {
          SERIAL_PORT.println("Failed to find IMU");
     }
    if (client.connect(server, 8080))
     {
        Serial.println("connected");
      }
     else
      {
        Serial.println("connection failed");
    }
}

void loop() {
  if (framesToRead > 0) {
   digitalWrite(led, HIGH);
    if (myICM1.dataReady() && accel.available() ) {
        myICM1.getAGMT();
        accel.read();
        msg += String(accel.cx*1000) + " " + String(accel.cy*1000) + " " + String(accel.cz*1000) + " " + String(myICM1.accX()) 
        + " " + String(myICM1.accY()) + " " + String(myICM1.accZ()) + " " + String(myICM1.gyrX()) +  " " + String(myICM1.gyrY()) +  " " + String(myICM1.gyrZ())
        +  " " + String(myICM1.magX()) +  " " + String(myICM1.magY()) +  " " + String(myICM1.magZ()) + "\n";
        framesToRead --;
        
        // if no frames to read left, send reply.
        if (framesToRead == 0) {
            Serial.println("Sending samples.");
            msg += "EOD\n";
            client.println(msg);
            client.flush();
            msg = "";
        }
    }
  } else if (client.status() && client.available()) // if available recv, recv next command.
  {
    // clear buf and receive message
    memset(buffer, 0, 256 * sizeof(char));
    int bytesRead = client.read((uint8_t *)buffer, 256);
    String buf = String(buffer);
    // if (buf == "trigger_function") {
    //   triggerFunction();
    // }
    // else {
      framesToRead = buf.toInt();
      Serial.println("Measuring " + String(framesToRead) + " samples...");
      // 1
      digitalWrite(led, HIGH);
      delay(500);
      digitalWrite(led, LOW);
      delay(500);
      
      // 2
      digitalWrite(led, HIGH);
      delay(500);
      digitalWrite(led, LOW);
      delay(500);
    //}
  } else { // if no available recv and no frames to read, inactive.
      digitalWrite(led, LOW);
  }
}

void triggerFunction() {
    if (Particle.connected()) {
        Particle.publish("push-notification", "it has activated");
    }
    else{
        Serial.print("could not push");
    }
}



void readAndPrintData(){
    if (myICM1.dataReady() && accel.available()) {
    myICM1.getAGMT();
    accel.read();
    SERIAL_PORT.print(myICM1.accX(), 3);
    SERIAL_PORT.print(",");
    SERIAL_PORT.print(accel.cx*1000, 3);
    SERIAL_PORT.print(",");
    SERIAL_PORT.print(myICM1.accY(), 3);
    SERIAL_PORT.print(",");
    SERIAL_PORT.print(accel.cy*1000, 3);
    SERIAL_PORT.print(",");
    SERIAL_PORT.print(myICM1.accZ(), 3);
    SERIAL_PORT.print(",");
    SERIAL_PORT.print(accel.cz*1000, 3);
    SERIAL_PORT.print(",");
    SERIAL_PORT.print(myICM1.gyrX());
    SERIAL_PORT.print(",");
    SERIAL_PORT.print(myICM1.gyrY());
    SERIAL_PORT.print(",");
    SERIAL_PORT.print(myICM1.gyrZ());
    SERIAL_PORT.print(",");
    SERIAL_PORT.print(myICM1.magX());
    SERIAL_PORT.print(",");
    SERIAL_PORT.print(myICM1.magY());
    SERIAL_PORT.print(",");
    SERIAL_PORT.print(myICM1.magZ());
    SERIAL_PORT.print(",");
    SERIAL_PORT.print(myICM1.temp());
    Serial.println();
  }
  else{
      SERIAL_PORT.println("Waiting for data");
  }
}
