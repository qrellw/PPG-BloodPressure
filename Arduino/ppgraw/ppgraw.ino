#include <Wire.h>
#include "MAX30105.h"

MAX30105 particleSensor;

// Khai báo chân I2C cho ESP32
const int I2C_SDA = 16; 
const int I2C_SCL = 17;

// Cài đặt thông số thời gian cho Sample Rate 100Hz
unsigned long previousMicros = 0;
const unsigned long intervalMicros = 10000; // 10,000 micro-giây = 10 mili-giây = 100Hz

void setup() {
  Serial.begin(115200);
  Wire.begin(I2C_SDA, I2C_SCL);

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("Loi ket noi MAX30102! Kiem tra lai day I2C.");
    while (1);
  }

  // Cấu hình MAX30102: Sample Rate 100Hz
  particleSensor.setup(0x3F, 1, 2, 100, 411, 4096); 
  
  // In ra Header cho file CSV
  Serial.println("Raw_IR"); 
}

void loop() {
  unsigned long currentMicros = micros();

  // Chỉ thực thi khi đã trôi qua đúng 10,000 micro-giây
  if (currentMicros - previousMicros >= intervalMicros) {
    // Cập nhật lại mốc thời gian. 
    // Cộng dồn intervalMicros giúp bù trừ sai số thời gian thực thi lệnh
    previousMicros += intervalMicros; 

    long irValue = particleSensor.getIR(); // Lấy tín hiệu thô

    if (irValue > 50000) { // Có ngón tay
      Serial.println(irValue);
    } else { // Không có ngón tay
      Serial.println(0); 
    }
  }
  
}