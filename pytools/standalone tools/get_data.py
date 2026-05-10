import serial
import serial.tools.list_ports
import os
import csv
from datetime import datetime

def find_ch340_port():
    """Tự động tìm cổng COM của chip nạp CH340"""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # Kiểm tra chuỗi CH340 trong mô tả thiết bị
        if "CH340" in port.description.upper():
            print(f"Đã tìm thấy ESP32 tại: {port.device}")
            return port.device
    return None

# --- XỬ LÝ ĐƯỜNG DẪN THƯ MỤC ---
# 1. Tìm thư mục chứa file script hiện tại (Code/pytools)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Nhảy lên 1 cấp (Code/) rồi chui vào 'ppgdata'
# Kết quả sẽ là: D:/.../Research/BME 1/Code/ppgdata
target_dir = os.path.abspath(os.path.join(current_dir, "..", "ppgdata"))

# 3. Tạo thư mục ppgdata nếu nó chưa tồn tại
os.makedirs(target_dir, exist_ok=True)

# --- KẾT NỐI SERIAL ---
port = find_ch340_port()
if not port:
    print("Lỗi: Không tìm thấy chip ")
    exit()

try:
    ser = serial.Serial(port, 115200, timeout=1)
    
    # Tạo tên file theo thời gian thực
    filename = f"ppg_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    save_path = os.path.join(target_dir, filename)
    
    print(f" Đang lưu dữ liệu vào: {save_path}")
    print(" Nhấn Ctrl + C để dừng ghi.")

    with open(save_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Ghi header cho file CSV
        writer.writerow(["Tran", "San", "Gia_tri_PPG", "Truc_0"])
        
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    data = line.split(',')
                    if len(data) == 4: # Chỉ ghi nếu đủ 4 cột
                        writer.writerow(data)
                        print(f"Data: {data[2]}") # In cột PPG ra xem cho vui

except KeyboardInterrupt:
    print("\n Dừng ghi. File đã được lưu an toàn!")
except Exception as e:
    print(f" Lỗi: {e}")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()