import sys
import serial
import serial.tools.list_ports
import os
import csv
from datetime import datetime
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QComboBox, QLabel, QTabWidget, 
                             QFileDialog, QMessageBox, QSplitter, QSpinBox, QCheckBox)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt
import pyqtgraph as pg
import pyqtgraph.exporters
import pandas as pd
import scipy.fftpack
from scipy.signal import find_peaks

# --- SERIAL THREAD ---
class SerialThread(QThread):
    new_data = pyqtSignal(float, float, float, float)
    error = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.port = ""
        self.baudrate = 115200
        self.is_running = False
        self.ser = None
        self.save_file = None
        self.csv_writer = None

    def run(self):
        self.is_running = True
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            
            # Tạo thư mục và file lưu trữ
            current_dir = os.path.dirname(os.path.abspath(__file__))
            target_dir = os.path.abspath(os.path.join(current_dir, "..", "ppgdata"))
            os.makedirs(target_dir, exist_ok=True)
            filename = f"live_ppg_gui_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            save_path = os.path.join(target_dir, filename)
            
            self.save_file = open(save_path, mode='w', newline='')
            self.csv_writer = csv.writer(self.save_file)
            self.csv_writer.writerow(["Tran", "San", "Gia_tri_PPG", "Truc_0"])
            
            while self.is_running:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        parts = line.split(',')
                        if len(parts) == 4:
                            try:
                                t, s, p, z = map(float, parts)
                                self.csv_writer.writerow([t, s, p, z])
                                self.new_data.emit(t, s, p, z)
                            except ValueError:
                                pass 
        except Exception as e:
            self.error.emit(str(e))
        finally:
            if self.save_file:
                self.save_file.close()
            if self.ser and self.ser.is_open:
                self.ser.close()

    def stop(self):
        self.is_running = False
        self.wait()

# --- MAIN WINDOW ---
class PPGAnalyzerSuite(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PPG Analyzer Suite (BME) - Advanced")
        self.resize(1200, 800)
        
        # Thiết lập style PyQtGraph
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOptions(antialias=True)
        
        self.serial_thread = SerialThread()
        self.serial_thread.new_data.connect(self.on_new_data)
        self.serial_thread.error.connect(self.on_serial_error)
        
        # Buffers cho Live Plotting (1000 mẫu = 10s tại 100Hz)
        self.max_samples = 1000
        self.data_p = np.zeros(self.max_samples)
        self.data_t = np.zeros(self.max_samples)
        self.data_s = np.zeros(self.max_samples)
        
        # Offline data buffers
        self.offline_signal = None
        self.offline_filename = ""
        
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        self.tab_live = QWidget()
        self.tab_offline = QWidget()
        self.tabs.addTab(self.tab_live, "1. Live Plotting")
        self.tabs.addTab(self.tab_offline, "2. Offline Analysis")
        
        self.setup_live_tab()
        self.setup_offline_tab()
        
    def setup_live_tab(self):
        layout = QVBoxLayout(self.tab_live)
        
        # --- BẢNG ĐIỀU KHIỂN ---
        controls = QHBoxLayout()
        self.cb_ports = QComboBox()
        self.update_ports()
        
        self.btn_refresh = QPushButton("Làm mới Cổng")
        self.btn_refresh.clicked.connect(self.update_ports)
        
        self.btn_connect = QPushButton("Kết nối & Bắt đầu thu")
        self.btn_connect.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_connect.clicked.connect(self.toggle_connection)
        
        self.spin_window = QSpinBox()
        self.spin_window.setRange(100, 10000)
        self.spin_window.setValue(1000)
        self.spin_window.setSingleStep(100)
        self.spin_window.setSuffix(" mẫu")
        self.spin_window.valueChanged.connect(self.change_window_size)
        
        controls.addWidget(QLabel("Cổng COM:"))
        controls.addWidget(self.cb_ports)
        controls.addWidget(self.btn_refresh)
        controls.addWidget(QLabel(" | Khung hiển thị:"))
        controls.addWidget(self.spin_window)
        controls.addWidget(self.btn_connect)
        controls.addStretch()
        layout.addLayout(controls)
        
        # --- ĐỒ THỊ LIVE ---
        self.plot_widget = pg.PlotWidget(title="Tín hiệu PPG Thời gian thực")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()
        self.plot_widget.setLabel('bottom', "Mẫu (Samples)")
        self.plot_widget.setLabel('left', "Biên độ (Amplitude)")
        # Bật tính năng tự động co giãn trục Y để sóng không bị bẹp thành đường thẳng
        self.plot_widget.enableAutoRange(axis='y') 
        
        # Tạo các đường vẽ
        self.curve_p = self.plot_widget.plot(pen=pg.mkPen('purple', width=2), name="Sóng PPG")
        self.curve_t = self.plot_widget.plot(pen=pg.mkPen('r', width=1, style=pg.QtCore.Qt.PenStyle.DashLine), name="Trần")
        self.curve_s = self.plot_widget.plot(pen=pg.mkPen('g', width=1, style=pg.QtCore.Qt.PenStyle.DashLine), name="Sàn")
        
        layout.addWidget(self.plot_widget)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(20) # 20ms
        
    def setup_offline_tab(self):
        layout = QVBoxLayout(self.tab_offline)
        
        # Các nút điều khiển chung
        controls = QHBoxLayout()
        self.btn_load_csv = QPushButton("1. Chọn file CSV")
        self.btn_load_csv.clicked.connect(self.load_csv)
        
        self.btn_analyze = QPushButton("2. Bắt đầu Phân tích")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self.analyze_data)
        
        self.btn_export_fft = QPushButton("Lưu ảnh Tab FFT")
        self.btn_export_fft.clicked.connect(self.export_fft)
        
        self.btn_export_time = QPushButton("Lưu ảnh Tab Đạo hàm")
        self.btn_export_time.clicked.connect(self.export_time_domain)
        
        controls.addWidget(self.btn_load_csv)
        controls.addWidget(self.btn_analyze)
        controls.addStretch()
        controls.addWidget(self.btn_export_fft)
        controls.addWidget(self.btn_export_time)
        layout.addLayout(controls)
        
        # === TẠO SUB-TABS ===
        self.offline_tabs = QTabWidget()
        
        # --- TAB 1: FOURIER ---
        self.sub_tab_fft = QWidget()
        layout_fft = QVBoxLayout(self.sub_tab_fft)
        
        splitter_fft = QSplitter(Qt.Orientation.Vertical)
        
        self.fft_ppg_plot = pg.PlotWidget(title="Tín hiệu PPG Thô (Time Domain)")
        self.fft_ppg_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fft_ppg_plot.setLabel('bottom', "Mẫu (Samples)")
        self.fft_ppg_plot.setLabel('left', "Biên độ")
        splitter_fft.addWidget(self.fft_ppg_plot)
        
        self.fft_widget = pg.PlotWidget(title="Phân tích Phổ Tần Số (FFT) - Tìm nguồn nhiễu")
        self.fft_widget.showGrid(x=True, y=True, alpha=0.3)
        self.fft_widget.setLabel('bottom', "Tần số (Hz)")
        self.fft_widget.setLabel('left', "Biên độ (Magnitude)")
        self.fft_widget.addLegend()
        
        # Các dải màu
        self.lr_baseline = pg.LinearRegionItem([0, 0.5], brush=(255, 0, 0, 30), movable=False)
        self.lr_hr = pg.LinearRegionItem([0.8, 3.0], brush=(0, 255, 0, 30), movable=False)
        self.lr_notch = pg.LinearRegionItem([3.0, 10.0], brush=(255, 165, 0, 30), movable=False)
        self.fft_widget.addItem(self.lr_baseline)
        self.fft_widget.addItem(self.lr_hr)
        self.fft_widget.addItem(self.lr_notch)
        
        self.fft_widget.plot([], [], pen=pg.mkPen(color=(255,100,100), width=3), name="Nhiễu Baseline (<0.5Hz)")
        self.fft_widget.plot([], [], pen=pg.mkPen(color=(100,255,100), width=3), name="Dải Nhịp Tim (0.8-3Hz)")
        self.fft_widget.plot([], [], pen=pg.mkPen(color=(255,200,100), width=3), name="Đặc trưng nhịp (>3Hz)")
        
        # Biến chứa curve FFT để update an toàn
        self.fft_curve = self.fft_widget.plot([], [], pen=pg.mkPen('b', width=1.5))
        
        splitter_fft.addWidget(self.fft_widget)
        splitter_fft.setSizes([400, 600])
        layout_fft.addWidget(splitter_fft)
        
        # --- TAB 2: ĐẠO HÀM (VPG/APG) ---
        self.sub_tab_deriv = QWidget()
        layout_deriv = QVBoxLayout(self.sub_tab_deriv)
        
        # Cụm Checkbox Ẩn/Hiện
        chk_layout = QHBoxLayout()
        self.cb_show_vpg = QCheckBox("Hiển thị VPG (Bậc 1)")
        self.cb_show_vpg.setChecked(True)
        self.cb_show_vpg.stateChanged.connect(self.toggle_vpg)
        
        self.cb_show_apg = QCheckBox("Hiển thị APG (Bậc 2)")
        self.cb_show_apg.setChecked(True)
        self.cb_show_apg.stateChanged.connect(self.toggle_apg)
        
        chk_layout.addWidget(self.cb_show_vpg)
        chk_layout.addWidget(self.cb_show_apg)
        chk_layout.addStretch()
        layout_deriv.addLayout(chk_layout)
        
        self.deriv_time_widget = pg.GraphicsLayoutWidget()
        
        self.deriv_ppg_plot = self.deriv_time_widget.addPlot(title="Tín hiệu PPG Thô (Time Domain)")
        self.deriv_ppg_plot.showGrid(x=True, y=True, alpha=0.3)
        self.deriv_ppg_plot.setLabel('left', "Biên độ")
        
        self.deriv_time_widget.nextRow()
        self.vpg_plot = self.deriv_time_widget.addPlot(title="VPG (Đạo hàm Bậc 1)")
        self.vpg_plot.showGrid(x=True, y=True, alpha=0.3)
        self.vpg_plot.setLabel('left', "VPG")
        self.vpg_plot.setXLink(self.deriv_ppg_plot) # Đồng bộ trục X
        
        self.deriv_time_widget.nextRow()
        self.apg_plot = self.deriv_time_widget.addPlot(title="APG (Đạo hàm Bậc 2)")
        self.apg_plot.showGrid(x=True, y=True, alpha=0.3)
        self.apg_plot.setLabel('bottom', "Mẫu (Samples)")
        self.apg_plot.setLabel('left', "APG")
        self.apg_plot.setXLink(self.deriv_ppg_plot) # Đồng bộ trục X
        
        layout_deriv.addWidget(self.deriv_time_widget)
        
        self.offline_tabs.addTab(self.sub_tab_fft, "Trang 1: Time Domain & Fourier")
        self.offline_tabs.addTab(self.sub_tab_deriv, "Trang 2: Time Domain & Đạo hàm")
        
        layout.addWidget(self.offline_tabs)
        
    def update_ports(self):
        self.cb_ports.clear()
        ports = serial.tools.list_ports.comports()
        for p in ports:
            self.cb_ports.addItem(f"{p.device} ({p.description})")
            
    def toggle_connection(self):
        if self.serial_thread.is_running:
            self.serial_thread.stop()
            self.btn_connect.setText("Kết nối & Bắt đầu thu")
            self.btn_connect.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
            self.cb_ports.setEnabled(True)
        else:
            port_text = self.cb_ports.currentText()
            if not port_text:
                return
            port_name = port_text.split(" ")[0]
            self.serial_thread.port = port_name
            self.serial_thread.start()
            self.btn_connect.setText("Dừng thu (Disconnect)")
            self.btn_connect.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
            self.cb_ports.setEnabled(False)
            
    def change_window_size(self, size):
        self.max_samples = size
        # Khởi tạo lại mảng khi đổi kích thước cửa sổ (xóa dữ liệu cũ trên màn hình)
        self.data_p = np.zeros(self.max_samples)
        self.data_t = np.zeros(self.max_samples)
        self.data_s = np.zeros(self.max_samples)
            
    def on_new_data(self, t, s, p, z):
        self.data_p[:-1] = self.data_p[1:]
        self.data_p[-1] = p
        self.data_t[:-1] = self.data_t[1:]
        self.data_t[-1] = t
        self.data_s[:-1] = self.data_s[1:]
        self.data_s[-1] = s

    def update_plot(self):
        if self.serial_thread.is_running:
            self.curve_p.setData(self.data_p)
            self.curve_t.setData(self.data_t)
            self.curve_s.setData(self.data_s)

    def on_serial_error(self, err_msg):
        self.toggle_connection()
        QMessageBox.critical(self, "Lỗi Serial", f"Lỗi: {err_msg}")
        
    def load_csv(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.abspath(os.path.join(current_dir, "..", "ppgdata"))
        
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn file CSV", target_dir, "CSV Files (*.csv)")
        if not file_path:
            return
            
        try:
            df = pd.read_csv(file_path)
            if 'Gia_tri_PPG' in df.columns:
                signal = df['Gia_tri_PPG'].values
            elif 'Raw_IR' in df.columns:
                signal = df['Raw_IR'].values
            else:
                QMessageBox.warning(self, "Lỗi file", "Không tìm thấy cột 'Gia_tri_PPG' hoặc 'Raw_IR'.")
                return
                
            self.offline_signal = signal
            self.offline_filename = os.path.basename(file_path)
            
            # Reset đồ thị
            self.fft_ppg_plot.clear()
            self.deriv_ppg_plot.clear()
            self.vpg_plot.clear()
            self.apg_plot.clear()
            self.fft_curve.setData([], []) # Xóa phổ FFT cũ
            
            # Chỉ vẽ tín hiệu gốc lên 2 Plot Time Domain
            self.fft_ppg_plot.plot(self.offline_signal, pen=pg.mkPen('purple', width=1.5))
            self.fft_ppg_plot.setTitle(f"Tín hiệu PPG Thô - {self.offline_filename} ({len(signal)} mẫu)")
            
            self.deriv_ppg_plot.plot(self.offline_signal, pen=pg.mkPen('purple', width=1.5))
            self.deriv_ppg_plot.setTitle(f"Tín hiệu PPG Thô - {self.offline_filename} ({len(signal)} mẫu)")
            
            self.btn_analyze.setEnabled(True)
            self.btn_analyze.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
            
            QMessageBox.information(self, "Đã tải file", "Đã tải dữ liệu thành công. Nhấn 'Bắt đầu Phân tích' để xem FFT và Đạo hàm.")
            
        except Exception as e:
            QMessageBox.critical(self, "Lỗi đọc file", str(e))

    def toggle_vpg(self, state):
        if state == 2: # Qt.CheckState.Checked = 2 in PyQt6
            self.vpg_plot.show()
        else:
            self.vpg_plot.hide()
            
    def toggle_apg(self, state):
        if state == 2:
            self.apg_plot.show()
        else:
            self.apg_plot.hide()

    def analyze_data(self):
        if self.offline_signal is None:
            return
            
        signal = self.offline_signal
        signal_no_dc = signal - np.mean(signal)
        
        # 1. Tính toán VPG và APG
        vpg = np.gradient(signal_no_dc)
        apg = np.gradient(vpg)
        
        self.vpg_plot.clear()
        self.apg_plot.clear()
        
        self.vpg_plot.plot(vpg, pen=pg.mkPen('g', width=1.5))
        self.apg_plot.plot(apg, pen=pg.mkPen('orange', width=1.5))
        
        # 2. Tìm đỉnh trên PPG gốc
        peaks, _ = find_peaks(signal_no_dc, distance=50, prominence=np.max(signal_no_dc)*0.1)
        scatter1 = pg.ScatterPlotItem(
            x=peaks, y=signal[peaks], 
            size=8, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 255)
        )
        scatter2 = pg.ScatterPlotItem(
            x=peaks, y=signal[peaks], 
            size=8, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 255)
        )
        # Thêm chấm đỏ vào cả 2 tab
        self.fft_ppg_plot.addItem(scatter1)
        self.deriv_ppg_plot.addItem(scatter2)
        
        # Tính BPM
        bpm = 0
        if len(peaks) > 1:
            intervals = np.diff(peaks) * 0.01 # Fs = 100Hz
            bpm = 60.0 / np.mean(intervals)
        
        title_str = f"Tín hiệu PPG Thô - {self.offline_filename} | Nhịp tim: {bpm:.1f} BPM"
        self.fft_ppg_plot.setTitle(title_str)
        self.deriv_ppg_plot.setTitle(title_str)
        
        # 3. Tính toán và vẽ FFT
        Fs = 100
        N = len(signal_no_dc)
        T = 1.0 / Fs
        
        yf = scipy.fftpack.fft(signal_no_dc)
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
        yf_abs = 2.0/N * np.abs(yf[:N//2])
        
        # Cập nhật curve FFT thay vì xóa đi tạo lại
        self.fft_curve.setData(xf, yf_abs)
        self.fft_widget.setXRange(0, 10) # Zoom mặc định từ 0 - 10Hz
        
        QMessageBox.information(self, "Hoàn tất", "Đã phân tích xong Phổ tần số và Đạo hàm.")

    def export_fft(self):
        # Lưu hình ảnh của toàn bộ sub_tab_fft
        pixmap = self.sub_tab_fft.grab()
        path, _ = QFileDialog.getSaveFileName(self, "Lưu ảnh Trang FFT", "fft_page.png", "PNG Files (*.png);;JPEG Files (*.jpg)")
        if path:
            pixmap.save(path)
            QMessageBox.information(self, "Thành công", f"Đã lưu: {path}")

    def export_time_domain(self):
        # Lưu hình ảnh của toàn bộ sub_tab_deriv
        pixmap = self.sub_tab_deriv.grab()
        path, _ = QFileDialog.getSaveFileName(self, "Lưu ảnh Trang Đạo hàm", "deriv_page.png", "PNG Files (*.png);;JPEG Files (*.jpg)")
        if path:
            pixmap.save(path)
            QMessageBox.information(self, "Thành công", f"Đã lưu: {path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PPGAnalyzerSuite()
    window.show()
    sys.exit(app.exec())
