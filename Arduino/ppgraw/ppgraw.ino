#include <Wire.h>
#include "MAX30105.h"

MAX30105 particleSensor;

// Khai báo chân I2C cho ESP32
const int I2C_SDA = 16; 
const int I2C_SCL = 17;

void setup() {
  Serial.begin(115200);
  Wire.begin(I2C_SDA, I2C_SCL);

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("Loi ket noi MAX30102! Kiem tra lai day I2C.");
    while (1);
  }

  // Cấu hình Sample Rate 100Hz
  particleSensor.setup(0x3F, 1, 2, 100, 411, 4096); 
  
  // In ra Header cho file CSV (Chỉ in 1 cột duy nhất để dễ phân tích)
  Serial.println("Raw_IR"); 
}

void loop() {
  long irValue = particleSensor.getIR(); // Lấy tín hiệu thô từ đèn Hồng ngoại

  if (irValue > 50000) { // Cảm biến nhận diện có ngón tay
    
    // In thẳng giá trị thô ra Serial. 
    // Serial Plotter sẽ tự động scale (co giãn) trục Y để hiển thị dao động.
    Serial.println(irValue);
    
  } else {
    // Không có ngón tay thì in ra 0
    Serial.println(0); 
  }

  // Delay 10ms để duy trì tốc độ 100Hz
  delay(10); 
}