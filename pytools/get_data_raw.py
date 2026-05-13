import serial
import serial.tools.list_ports
import os
import csv
import sys
from datetime import datetime

def find_ch340_port():
    """Tự động tìm cổng COM của chip nạp CH340"""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "CH340" in port.description.upper():
            print(f"Đã tìm thấy ESP32 tại: {port.device}")
            return port.device
    return None

def print_progress(iteration, total, length=40):
    """Hàm vẽ thanh tiến độ trên Terminal"""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    # \r để in đè lên dòng cũ, tạo cảm giác thanh tiến độ đang chạy
    sys.stdout.write(f'\r Tiến độ: |{bar}| {percent}% ({iteration}/{total} mẫu)')
    sys.stdout.flush()

# --- XỬ LÝ ĐƯỜNG DẪN THƯ MỤC ---
current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(current_dir, "..", "ppgdata"))
os.makedirs(target_dir, exist_ok=True)

# --- KẾT NỐI SERIAL ---
port = find_ch340_port()
if not port:
    print(" Lỗi: Không tìm thấy chip CH340")
    exit()

# Thiết lập số lượng mẫu cần thu (9000 mẫu ở 100Hz = 90 giây)
TARGET_SAMPLES = 9000
sample_count = 0

try:
    # Mở cổng Serial
    ser = serial.Serial(port, 115200, timeout=1)
    
    # Đổi tên file có chữ 'raw' để dễ phân biệt
    filename = f"raw_ppg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    save_path = os.path.join(target_dir, filename)
    
    print(f"\n Đang thu thập {TARGET_SAMPLES} mẫu dữ liệu thô (RAW)...")
    print(f" File lưu tại: {save_path}\n")

    with open(save_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Raw_IR"]) # Chỉ ghi 1 cột duy nhất
        
        # Khởi tạo thanh tiến độ ở mức 0
        print_progress(0, TARGET_SAMPLES)
        
        while sample_count < TARGET_SAMPLES:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    # Bỏ qua dòng Header "Raw_IR" do Arduino in ra (nếu có)
                    if line == "Raw_IR":
                        continue
                    
                    try:
                        # Đảm bảo dữ liệu nhận được là số nguyên (giá trị thô thường rất lớn)
                        value = int(line) 
                        writer.writerow([value])
                        sample_count += 1
                        
                        # Cập nhật thanh tiến độ (cứ 10 mẫu update 1 lần cho đỡ lag Terminal)
                        if sample_count % 10 == 0 or sample_count == TARGET_SAMPLES:
                            print_progress(sample_count, TARGET_SAMPLES)
                            
                    except ValueError:
                        # Nếu dính dòng rác do nhiễu cổng COM thì bỏ qua
                        pass 

except Exception as e:
    print(f"\n Lỗi: {e}")
finally:
    # Đoạn này sẽ chạy ngay khi thu đủ 9000 mẫu
    print("\n\n Thu thập hoàn tất! File đã được đóng an toàn.")
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print(" Đã ngắt kết nối cổng COM")