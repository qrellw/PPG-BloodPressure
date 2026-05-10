import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import csv
from datetime import datetime
from collections import deque

# --- CẤU HÌNH ---
MAX_SAMPLES = 1000  # Số lượng mẫu hiển thị trên màn hình (độ rộng khung hình)
BAUD_RATE = 115200

def find_ch340():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "CH340" in port.description.upper():
            return port.device
    return None

# --- XỬ LÝ THƯ MỤC ---
current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(current_dir, "..", "ppgdata"))
os.makedirs(target_dir, exist_ok=True)

# --- KHỞI TẠO DỮ LIỆU ---
data_ppg = deque(maxlen=MAX_SAMPLES)
data_tran = deque(maxlen=MAX_SAMPLES)
data_san = deque(maxlen=MAX_SAMPLES)
data_zero = deque(maxlen=MAX_SAMPLES)

port = find_ch340()
if not port:
    print(" Không thấy CH340!")
    exit()

ser = serial.Serial(port, BAUD_RATE, timeout=0.1)

# File lưu trữ
filename = f"live_ppg_{datetime.now().strftime('%H%M%S')}.csv"
save_path = os.path.join(target_dir, filename)
f = open(save_path, mode='w', newline='')
writer = csv.writer(f)
writer.writerow(["Tran", "San", "Gia_tri_PPG", "Truc_0"])

# --- THIẾT LẬP ĐỒ THỊ (EXCEL STYLE) ---
plt.style.use('bmh') # Style nhìn cho sạch sẽ
fig, ax = plt.subplots(figsize=(10, 6))
line_ppg, = ax.plot([], [], color='teal', linewidth=1.5, label='Sóng PPG')
line_tran, = ax.plot([], [], color='blue', linewidth=1, linestyle='--', label='Trần')
line_san, = ax.plot([], [], color='orange', linewidth=1, linestyle='--', label='Sàn')
line_zero, = ax.plot([], [], color='red', linewidth=1, alpha=0.5)

ax.set_title("Real-time PPG Monitor - Project 1 (BME)", fontsize=14)
ax.set_ylim(-6000, 4000) # Khóa khung nhìn y-axis giống cái notch V của ông
ax.set_xlim(0, MAX_SAMPLES)
ax.legend(loc='upper right')

def update(frame):
    while ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                parts = line.split(',')
                if len(parts) == 4:
                    t, s, p, z = map(float, parts)
                    
                    # Lưu vào file
                    writer.writerow([t, s, p, z])
                    
                    # Đưa vào hàng đợi để vẽ
                    data_tran.append(t)
                    data_san.append(s)
                    data_ppg.append(p)
                    data_zero.append(z)
        except:
            continue

    # Cập nhật dữ liệu lên đồ thị
    line_ppg.set_data(range(len(data_ppg)), list(data_ppg))
    line_tran.set_data(range(len(data_tran)), list(data_tran))
    line_san.set_data(range(len(data_san)), list(data_san))
    line_zero.set_data(range(len(data_zero)), list(data_zero))
    
    return line_ppg, line_tran, line_san, line_zero

ani = FuncAnimation(fig, update, interval=20, blit=True)

print(f" Đang vẽ sóng và lưu tại: {save_path}")
plt.show()

# Đóng file khi tắt cửa sổ đồ thị
f.close()
ser.close()