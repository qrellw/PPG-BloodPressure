import serial
import time
import csv

# ⚠️ QUAN TRỌNG: Sửa COM port cho đúng với mạch ESP32 của ông
# Kiểm tra trong Arduino IDE xem ESP đang nhận COM mấy (ví dụ: 'COM3', 'COM5'...)
SERIAL_PORT = 'COM5' 
BAUD_RATE = 115200

# Khởi tạo cổng Serial
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Đã kết nối thành công với {SERIAL_PORT}")
except Exception as e:
    print(f"Lỗi cổng COM: {e}")
    exit()

# Tên file xuất ra (thêm giờ cho khỏi bị trùng nếu đo nhiều lần)
filename = time.strftime("ppg_data_%Y%m%d_%H%M%S.csv")

# Mở file CSV để ghi
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Tran', 'San', 'Gia_tri_PPG', 'Truc_0']) # Tạo tên cột

    print("Đợi 5 giây cho sóng ổn định...")
    time.time() # Xóa bộ đệm cũ
    ser.reset_input_buffer() 
    time.sleep(5) # Bỏ qua 5 giây đầu nhiễu

    print(f"BẮT ĐẦU GHI 2500 MẪU (khoảng 25 giây)...")
    
    samples_collected = 0
    while samples_collected < 2500:
        if ser.in_waiting > 0:
            try:
                # Đọc 1 dòng từ ESP32
                line = ser.readline().decode('utf-8').strip()
                
                # Tách 4 con số bằng dấu phẩy
                data = line.split(',')
                
                # Kiểm tra nếu đúng là có 4 cột thì mới lưu
                if len(data) == 4:
                    writer.writerow(data)
                    samples_collected += 1
                    
                    # In tiến độ cho vui
                    if samples_collected % 100 == 0:
                        print(f"Đã thu: {samples_collected}/2500 mẫu")
                        
            except Exception as e:
                pass # Bỏ qua nếu có lỗi giải mã (rác data)

print(f"\n HOÀN THÀNH! Dữ liệu đã lưu vào file: {filename}")
ser.close()