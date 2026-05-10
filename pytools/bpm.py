import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# Đọc file
df = pd.read_csv('ppg_data_20260503_124146.csv')
signal = df['Gia_tri_PPG'].values

# Vì đỉnh của ông đang là giá trị lớn nhất (ví dụ ~800) 
# Ta dùng find_peaks với ngưỡng để bắt đỉnh tâm thu
# distance=70 vì ở 100Hz, nhịp tim không thể nhanh hơn 1.4 nhịp/s (85 BPM) 
# nên các đỉnh cách nhau ít nhất 70 mẫu là an toàn
peaks, _ = find_peaks(signal, distance=70, height=-500)

# Tính khoảng cách giữa các đỉnh (đơn vị: mẫu)
diffs = np.diff(peaks)

# Chuyển từ mẫu sang giây (Fs = 100Hz nên 1 mẫu = 0.01s)
intervals_sec = diffs * 0.01

# Tính BPM cho từng khoảng và lấy trung bình
bpms = 60 / intervals_sec
avg_bpm = np.mean(bpms)

print(f"Số lượng nhịp tim đếm được: {len(peaks)}")
print(f"Nhịp tim trung bình (BPM): {avg_bpm:.2f}")