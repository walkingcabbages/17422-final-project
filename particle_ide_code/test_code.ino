// This #include statement was automatically added by the Particle IDE.
#include "ICM_20948.h"
#define SERIAL_PORT Serial
#define WIRE_PORT Wire
#define AD0_VAL_1 0
#define AD0_VAL_2 1 

TCPClient client; // TCP Client
byte server[] = {172,26,23,208}; // TODO: Server IP, separated by commas instead of periods
char buffer[256]; // rx buffer
int framesToRead = 0; // samples remaining
String msg = ""; // tx buffer

ICM_20948_I2C myICM1; // Instance for first sensor
ICM_20948_I2C myICM2; // Instance for first sensor


void setup() {
  SERIAL_PORT.begin(115200);
  while (!SERIAL_PORT);

  WIRE_PORT.begin();
  WIRE_PORT.setClock(400000);

  myICM1.begin(WIRE_PORT, AD0_VAL_1);
  myICM2.begin(WIRE_PORT, AD0_VAL_2);

  SERIAL_PORT.print("Initialization of sensor 1 returned: ");
  SERIAL_PORT.println(myICM1.statusString());
  
  SERIAL_PORT.print("Initialization of sensor 2 returned: ");
  SERIAL_PORT.println(myICM2.statusString());

}

void loop() {
 readAndPrintData();
 delay(300);
}

// void triggerFunction() {
//     if (Particle.connected()) {
//         Particle.publish("push-notification", "it has activated");
//     }
//     else{
//         Serial.print("could not push");
//     }
// }



void readAndPrintData(){
    SERIAL_PORT.print(myICM1.dataReady());
    SERIAL_PORT.print(myICM2.dataReady());
    if (myICM1.dataReady() && myICM2.dataReady()) {
    myICM1.getAGMT();
    myICM2.getAGMT();
    
    SERIAL_PORT.print(myICM1.accX(), 3);
    SERIAL_PORT.print(",");
    SERIAL_PORT.print(myICM2.accX(), 3);
    // SERIAL_PORT.print(",");
    
    // SERIAL_PORT.print(myICM1.accY(), 3);
    // SERIAL_PORT.print(",");
    // SERIAL_PORT.print(myICM1.accZ(), 3);
    // SERIAL_PORT.print(",");
    // SERIAL_PORT.print(myICM1.gyrX());
    // SERIAL_PORT.print(",");
    // SERIAL_PORT.print(myICM1.gyrY());
    // SERIAL_PORT.print(",");
    // SERIAL_PORT.print(myICM1.gyrZ());
    // SERIAL_PORT.print(",");
    // SERIAL_PORT.print(myICM1.magX());
    // SERIAL_PORT.print(",");
    // SERIAL_PORT.print(myICM1.magY());
    // SERIAL_PORT.print(",");
    // SERIAL_PORT.print(myICM1.magZ());
    // SERIAL_PORT.print(",");
    // SERIAL_PORT.print(myICM1.temp());
    Serial.println();
  }
  else{
      SERIAL_PORT.println("Waiting for data");
      setup();
  }
}



