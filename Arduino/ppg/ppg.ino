#include <Wire.h>
#include "MAX30105.h"
#include "PpgFilter.h" // Nhúng "bảo vật" của ông vào đây

MAX30105 particleSensor;
PpgFilter myFilter; // Khởi tạo bộ lọc FIR 91 Taps

// Khai báo chân I2C cho ESP32 (Tùy mạch ông đang cắm nhé, đây là ví dụ)
const int I2C_SDA = 16; 
const int I2C_SCL = 17;

void setup() {
  Serial.begin(115200);
  Wire.begin(I2C_SDA, I2C_SCL);

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("Lỗi kết nối MAX30102! Kiểm tra lại dây I2C.");
    while (1);
  }

  // Cấu hình SIÊU QUAN TRỌNG: 
  // Số 100 ở giữa chính là Sample Rate (100Hz) bắt buộc phải khớp với bộ lọc
  particleSensor.setup(0x3F, 1, 2, 100, 411, 4096); 
}

void loop() {
  long irValue = particleSensor.getIR(); // Lấy tín hiệu thô từ đèn Hồng ngoại

  if (irValue > 50000) { // Cảm biến nhận diện có ngón tay
    
    // Đưa tín hiệu thô đi qua bộ lọc 91 tầng
    float clean_wave = myFilter.process((float)irValue);

    // BƯỚC 1: Khóa Zoom (In 2 đường giới hạn)
    Serial.print(300);    // Đường TRẦN (Cố định ở mức 300)
    Serial.print(",");
    Serial.print(-300);   // Đường SÀN (Cố định ở mức -300)
    Serial.print(",");

    // BƯỚC 2: In sóng PPG và trục 0
    Serial.print(clean_wave); // Đường sóng nhịp tim (Đống X đang dao động)
    Serial.print(",");
    Serial.println(0);        // Trục hoành mốc 0 nằm chính giữa
    
  } else {
    // Không có ngón tay thì vẫn phải in đủ 4 biến để Plotter không bị lỗi format
    Serial.print(300);    // Giữ nguyên trần
    Serial.print(",");
    Serial.print(-300);   // Giữ nguyên sàn
    Serial.print(",");
    Serial.print(0);      // Sóng nằm im ở số 0
    Serial.print(",");
    Serial.println(0);    // Trục hoành mốc 0
  }

  // Delay 10ms để đảm bảo đúng tốc độ 100 vòng/giây (100Hz)
  delay(10); 
}