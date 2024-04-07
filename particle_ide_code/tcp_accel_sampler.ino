// This #include statement was automatically added by the Particle IDE.
#include <SparkFunMMA8452Q.h>

MMA8452Q accel; // MMA8452 QWIIC Accelerometer
TCPClient client; // TCP Client
byte server[] = {172, 26, 1, 41}; // TODO: Server IP, separated by commas instead of periods
char buffer[256]; // rx buffer
int framesToRead = 0; // samples remaining
String msg = ""; // tx buffer

int led = D7; // on-board LED

void setup()
{
  // Make sure your Serial Terminal app is closed before powering your device
  Serial.begin(9600);
  pinMode(led, OUTPUT);
  
  // enable accelerometer at 50Hz
  if (!accel.begin(SCALE_2G, ODR_50)) {
      Serial.println("Failed to find IMU");
  }
  
  // connect to socket
  Serial.println("connecting...");
  
  if (client.connect(server, 8080))
  {
    Serial.println("connected");
  }
  else
  {
    Serial.println("connection failed");
  }
}

void loop()
{
  // if frames remaining, add accelerometer data.
  if (framesToRead > 0) {
    digitalWrite(led, HIGH);
    if (accel.available()) {
        accel.read();
        msg += String(accel.cx) + " " + String(accel.cy) + " " + String(accel.cz) + "\n";
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
  } else { // if no available recv and no frames to read, inactive.
      digitalWrite(led, LOW);
  }
}