#include <SparkFunMMA8452Q.h>
#include "ICM_20948.h"
#define SERIAL_PORT Serial
#define WIRE_PORT Wire
#define AD0_VAL_1 0 
#define AD0_VAL_2 1  

TCPClient client; // TCP Client
byte server[] = {10,0,0,10}; // TODO: Server IP, separated by commas instead of periods
char buffer[256]; // rx buffer
int framesToRead = 0; // samples remaining
String msg = ""; // tx buffer

ICM_20948_I2C myICM1; // Instance for first sensor
ICM_20948_I2C myICM2; // Instance for first sensor

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
  
  
  myICM2.begin(WIRE_PORT, AD0_VAL_2);
  SERIAL_PORT.print("Initialization of sensor 2 returned: ");
  SERIAL_PORT.println(myICM2.statusString());


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

void restartSensors(){
  SERIAL_PORT.begin(115200);
  while (!SERIAL_PORT);

  WIRE_PORT.begin();
  WIRE_PORT.setClock(400000);

  myICM2.begin(WIRE_PORT, AD0_VAL_2);


}

void triggerFunction(String prediction) {
    if (Particle.connected()) {
        Particle.publish("push-notification", prediction);
    }
    else{
        Serial.print("could not push");
    }
}

void loop() {
  if (framesToRead > 0) {
    digitalWrite(led, HIGH);
    if (myICM1.dataReady() && myICM2.dataReady() && accel.available()) {
        myICM1.getAGMT();
        myICM2.getAGMT();
        accel.read();
        msg += String(accel.cx*1000) + " " + String(accel.cy*1000) + " " + String(accel.cz*1000) 
        + " " + String(myICM1.accX())  + " " + String(myICM1.accY()) + " " + String(myICM1.accZ()) + " " + String(myICM1.gyrX()) +  " " + String(myICM1.gyrY()) +  " " + String(myICM1.gyrZ())
        +  " " + String(myICM1.magX()) +  " " + String(myICM1.magY()) +  " " + String(myICM1.magZ()) + 
        + " " + String(myICM2.accX()) + " " + String(myICM2.accY()) + " " + String(myICM2.accZ()) + " " + String(myICM2.gyrX()) +  " " + String(myICM2.gyrY()) +  " " + String(myICM2.gyrZ())
        +  " " + String(myICM2.magX()) +  " " + String(myICM2.magY()) +  " " + String(myICM2.magZ()) + "\n";
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
    else if(myICM2.statusString() == "Data Underflow"){
        Serial.println("Resetting");
        restartSensors();
        Serial.println("Done");
    }else{
        Serial.println(framesToRead);
        Serial.println(myICM1.dataReady()); 
        Serial.println(myICM2.statusString()); 
        Serial.println(accel.available()); 
    }
  }
  else if (client.status() && client.available()) // if available recv, recv next command.
  {
    // clear buf and receive message
    memset(buffer, 0, 256 * sizeof(char));
    int bytesRead = client.read((uint8_t *)buffer, 256);
    String buf = String(buffer);
    if (buf.startsWith("trigger_function")) {
      triggerFunction(buf.substring(16)); /* message format: "trigger_function <prediction>"*/
    } else {
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
    }
  } else { // if no available recv and no frames to read, inactive.
      digitalWrite(led, LOW);
  }
}




void readAndPrintData(){
    if (myICM1.dataReady() && accel.available() && myICM2.dataReady()) {
    myICM1.getAGMT();
    myICM2.getAGMT();
    accel.read();
    SERIAL_PORT.print(accel.cx*1000, 3);
    SERIAL_PORT.print(",");
    SERIAL_PORT.print(accel.cy*1000, 3);
    SERIAL_PORT.print(",");
    SERIAL_PORT.print(accel.cz*1000, 3);
    SERIAL_PORT.print(",");
    SERIAL_PORT.print(myICM1.accX(), 3);
    SERIAL_PORT.print(",");
    SERIAL_PORT.print(myICM1.accY(), 3);
    SERIAL_PORT.print(",");
    SERIAL_PORT.print(myICM1.accZ(), 3);
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
    SERIAL_PORT.print(",");
    SERIAL_PORT.print(myICM2.accX(), 3);
    SERIAL_PORT.print(",");
    SERIAL_PORT.print(myICM2.accY(), 3);
    SERIAL_PORT.print(",");
    SERIAL_PORT.print(myICM2.accZ(), 3);
    SERIAL_PORT.print(",");
    SERIAL_PORT.print(myICM2.gyrX());
    SERIAL_PORT.print(",");
    SERIAL_PORT.print(myICM2.gyrY());
    SERIAL_PORT.print(",");
    SERIAL_PORT.print(myICM2.gyrZ());
    SERIAL_PORT.print(",");
    SERIAL_PORT.print(myICM2.magX());
    SERIAL_PORT.print(",");
    SERIAL_PORT.print(myICM2.magY());
    SERIAL_PORT.print(",");
    SERIAL_PORT.print(myICM2.magZ());
    SERIAL_PORT.print(",");
    SERIAL_PORT.print(myICM2.temp());
    Serial.println();
  }
  else{
    SERIAL_PORT.println("Waiting for data");
    restartSensors();

  }
}
