import sys
import serial
import serial.tools.list_ports
import os
import csv
from datetime import datetime
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QComboBox, QLabel, QTabWidget, 
                             QFileDialog, QMessageBox, QSplitter, QSpinBox, QCheckBox,
                             QListWidget, QListWidgetItem)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt6.QtGui import QShortcut, QKeySequence
import pyqtgraph as pg
import pyqtgraph.exporters
import pandas as pd
import scipy.fftpack
from scipy.signal import find_peaks, butter, firwin, freqz, filtfilt, sosfreqz, sosfiltfilt, savgol_filter
from scipy.ndimage import median_filter

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
                        elif len(parts) == 1: # Raw signal từ ppgraw.ino (1 cột)
                            try:
                                p = float(parts[0])
                                # Ghi NaN cho các cột không có
                                self.csv_writer.writerow([np.nan, np.nan, p, np.nan])
                                self.new_data.emit(np.nan, np.nan, p, np.nan)
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
        
        # Filter coefficients
        self.filter_b = None
        self.filter_a = None
        
        # Pipeline State
        self.filter_pipeline = []
        self.current_filter_idx = -1
        self._is_populating = False
        self.pure_offline_signal = None
        
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
        self.tab_filter = QWidget()
        self.tabs.addTab(self.tab_live, "1. Live Plotting")
        self.tabs.addTab(self.tab_offline, "2. Offline Analysis")
        self.tabs.addTab(self.tab_filter, "3. Phát triển Bộ lọc (Filter)")
        
        self.setup_live_tab()
        self.setup_offline_tab()
        self.setup_filter_tab()
        
        # Phím tắt Hoàn tác / Làm lại cho thước tạm
        self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.shortcut_undo.activated.connect(self.undo_temp_ruler)
        
        self.shortcut_redo = QShortcut(QKeySequence("Ctrl+Y"), self)
        self.shortcut_redo.activated.connect(self.redo_temp_ruler)
        
        # Phím tắt chuyển đổi chế độ chấm đỏ
        self.shortcut_dot = QShortcut(QKeySequence("Ctrl+D"), self)
        self.shortcut_dot.activated.connect(self.toggle_dot_mode)
        
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
        
        self.cb_invert_live = QCheckBox("Lật ngược")
        
        controls.addWidget(QLabel("Cổng COM:"))
        controls.addWidget(self.cb_ports)
        controls.addWidget(self.btn_refresh)
        controls.addWidget(QLabel(" | Khung hiển thị:"))
        controls.addWidget(self.spin_window)
        controls.addWidget(self.cb_invert_live)
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
        
        # Thêm Crosshair
        self.vLine_live = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('gray', style=Qt.PenStyle.DashLine))
        self.plot_widget.addItem(self.vLine_live, ignoreBounds=True)
        self.label_live = pg.TextItem(anchor=(0, 1), color='k', fill=pg.mkBrush(255, 255, 255, 200))
        self.plot_widget.addItem(self.label_live, ignoreBounds=True)
        self.proxy_live = pg.SignalProxy(self.plot_widget.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved_live)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(20) # 20ms
        
    def setup_offline_tab(self):
        layout = QVBoxLayout(self.tab_offline)
        
        # Các nút điều khiển chung
        controls = QHBoxLayout()
        self.btn_load_csv = QPushButton("1. Chọn file CSV")
        self.btn_load_csv.clicked.connect(self.load_csv)
        
        self.btn_analyze = QPushButton("2. Phân tích (FFT & Đạo hàm)")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self.analyze_data)
        
        self.btn_find_peaks = QPushButton("3. Tìm Đỉnh & Nhịp tim")
        self.btn_find_peaks.setEnabled(False)
        self.btn_find_peaks.clicked.connect(self.find_peaks_bpm)
        self.btn_find_peaks.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        
        self.cb_invert_offline = QCheckBox("Lật ngược tín hiệu")
        self.cb_invert_offline.stateChanged.connect(self.on_invert_changed)
        
        self.btn_export_fft = QPushButton("Lưu ảnh Tab FFT")
        self.btn_export_fft.clicked.connect(self.export_fft)
        
        self.btn_export_time = QPushButton("Lưu ảnh Tab Đạo hàm")
        self.btn_export_time.clicked.connect(self.export_time_domain)
        
        controls.addWidget(self.btn_load_csv)
        controls.addWidget(self.btn_analyze)
        controls.addWidget(self.btn_find_peaks)
        controls.addWidget(self.cb_invert_offline)
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
        
        self.fft_widget = pg.PlotWidget(title="Phân tích Phổ Tần Số (FFT)")
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
        
        # --- Crosshair cho FFT Tab ---
        self.vLine_fft_ppg = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('gray', style=Qt.PenStyle.DashLine))
        self.vLine_fft = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('gray', style=Qt.PenStyle.DashLine))
        self.fft_ppg_plot.addItem(self.vLine_fft_ppg, ignoreBounds=True)
        self.fft_widget.addItem(self.vLine_fft, ignoreBounds=True)
        
        self.label_fft_ppg = pg.TextItem(anchor=(0, 1), color='k', fill=pg.mkBrush(255, 255, 255, 200))
        self.label_fft = pg.TextItem(anchor=(0, 1), color='k', fill=pg.mkBrush(255, 255, 255, 200))
        self.fft_ppg_plot.addItem(self.label_fft_ppg, ignoreBounds=True)
        self.fft_widget.addItem(self.label_fft, ignoreBounds=True)
        
        self.proxy_fft_ppg = pg.SignalProxy(self.fft_ppg_plot.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved_fft_ppg)
        self.proxy_fft_widget = pg.SignalProxy(self.fft_widget.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved_fft_widget)
        
        # --- TAB 2: ĐẠO HÀM (VPG/APG) ---
        self.sub_tab_deriv = QWidget()
        layout_deriv = QVBoxLayout(self.sub_tab_deriv)
        
        # Cụm Checkbox Ẩn/Hiện và nút xóa thước
        chk_layout = QHBoxLayout()
        self.cb_show_vpg = QCheckBox("Hiển thị VPG (Bậc 1)")
        self.cb_show_vpg.setChecked(True)
        self.cb_show_vpg.stateChanged.connect(self.toggle_vpg)
        
        self.cb_show_apg = QCheckBox("Hiển thị APG (Bậc 2)")
        self.cb_show_apg.setChecked(True)
        self.cb_show_apg.stateChanged.connect(self.toggle_apg)
        
        self.cb_dot_mode = QCheckBox("Chấm đỏ (Ctrl+D)")
        self.cb_dot_mode.setChecked(False)
        
        self.btn_clear_rulers = QPushButton("Xóa các đánh dấu tạm")
        self.btn_clear_rulers.clicked.connect(self.clear_temp_rulers)
        
        chk_layout.addWidget(self.cb_show_vpg)
        chk_layout.addWidget(self.cb_show_apg)
        chk_layout.addWidget(self.cb_dot_mode)
        chk_layout.addStretch()
        chk_layout.addWidget(self.btn_clear_rulers)
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
        
        # --- Crosshair cho Đạo hàm Tab ---
        self.vLine_deriv_ppg = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('gray', style=Qt.PenStyle.DashLine))
        self.vLine_vpg = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('gray', style=Qt.PenStyle.DashLine))
        self.vLine_apg = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('gray', style=Qt.PenStyle.DashLine))
        
        self.deriv_ppg_plot.addItem(self.vLine_deriv_ppg, ignoreBounds=True)
        self.vpg_plot.addItem(self.vLine_vpg, ignoreBounds=True)
        self.apg_plot.addItem(self.vLine_apg, ignoreBounds=True)
        
        self.label_deriv_ppg = pg.TextItem(anchor=(0, 1), color='k', fill=pg.mkBrush(255, 255, 255, 200))
        self.label_vpg = pg.TextItem(anchor=(0, 1), color='k', fill=pg.mkBrush(255, 255, 255, 200))
        self.label_apg = pg.TextItem(anchor=(0, 1), color='k', fill=pg.mkBrush(255, 255, 255, 200))
        
        self.deriv_ppg_plot.addItem(self.label_deriv_ppg, ignoreBounds=True)
        self.vpg_plot.addItem(self.label_vpg, ignoreBounds=True)
        self.apg_plot.addItem(self.label_apg, ignoreBounds=True)
        
        self.proxy_deriv = pg.SignalProxy(self.deriv_time_widget.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved_deriv)
        self.deriv_time_widget.scene().sigMouseClicked.connect(self.mouseClicked_deriv)
        
        self.offline_tabs.addTab(self.sub_tab_fft, "Trang 1: Time Domain & Fourier")
        self.offline_tabs.addTab(self.sub_tab_deriv, "Trang 2: Time Domain & Đạo hàm")
        
        layout.addWidget(self.offline_tabs)

    def setup_filter_tab(self):
        from PyQt6.QtWidgets import QDoubleSpinBox, QFormLayout, QGroupBox, QGridLayout, QInputDialog, QTextEdit, QDialog
        layout = QHBoxLayout(self.tab_filter)
        
        # Panel Điều khiển (Trái)
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(300)
        
        group_pipeline = QGroupBox("Danh sách Bộ lọc (Pipeline)")
        pipe_layout = QVBoxLayout(group_pipeline)
        
        self.list_pipeline = QListWidget()
        self.list_pipeline.currentRowChanged.connect(self.on_pipeline_selection_changed)
        self.list_pipeline.itemChanged.connect(self.on_pipeline_item_changed)
        
        btn_pipe_layout = QHBoxLayout()
        self.btn_add_filter = QPushButton("+ Thêm")
        self.btn_add_filter.clicked.connect(self.add_pipeline_filter)
        self.btn_remove_filter = QPushButton("- Xóa")
        self.btn_remove_filter.clicked.connect(self.remove_pipeline_filter)
        self.btn_up_filter = QPushButton("▲ Lên")
        self.btn_up_filter.clicked.connect(self.move_pipeline_filter_up)
        self.btn_down_filter = QPushButton("▼ Xuống")
        self.btn_down_filter.clicked.connect(self.move_pipeline_filter_down)
        
        btn_pipe_layout.addWidget(self.btn_add_filter)
        btn_pipe_layout.addWidget(self.btn_remove_filter)
        btn_pipe_layout.addWidget(self.btn_up_filter)
        btn_pipe_layout.addWidget(self.btn_down_filter)
        
        pipe_layout.addWidget(self.list_pipeline)
        pipe_layout.addLayout(btn_pipe_layout)
        control_layout.addWidget(group_pipeline)
        
        group_design = QGroupBox("Thông số Bộ lọc đang chọn")
        form_layout = QFormLayout(group_design)
        
        self.cb_filter_impl = QComboBox()
        self.cb_filter_impl.addItems(["IIR (Butterworth)", "FIR", "Kalman (1D)", "Hampel", "Savitzky-Golay"])
        self.cb_filter_impl.currentIndexChanged.connect(self.on_filter_impl_changed)
        
        self.cb_filter_type = QComboBox()
        self.cb_filter_type.addItems(["Lowpass", "Highpass", "Bandpass", "Bandstop"])
        self.cb_filter_type.currentIndexChanged.connect(self.on_filter_type_changed)
        
        # UI for FIR/IIR
        self.spin_order = QSpinBox()
        self.spin_order.setRange(1, 1000)
        self.spin_order.setValue(4)
        
        self.spin_cutoff1 = QDoubleSpinBox()
        self.spin_cutoff1.setRange(0.01, 50.0) # max Fs/2 = 50Hz
        self.spin_cutoff1.setValue(5.0)
        self.spin_cutoff1.setDecimals(2)
        self.spin_cutoff1.setSuffix(" Hz")
        
        self.spin_cutoff2 = QDoubleSpinBox()
        self.spin_cutoff2.setRange(0.01, 50.0)
        self.spin_cutoff2.setValue(10.0)
        self.spin_cutoff2.setDecimals(2)
        self.spin_cutoff2.setSuffix(" Hz")
        self.spin_cutoff2.hide()
        self.label_cutoff2 = QLabel("Tần số cắt 2:")
        self.label_cutoff2.hide()
        
        # UI for Hampel
        self.spin_hampel_window = QSpinBox()
        self.spin_hampel_window.setRange(3, 1000)
        self.spin_hampel_window.setValue(10)
        self.spin_hampel_window.hide()
        self.label_hampel_window = QLabel("Window Size:")
        self.label_hampel_window.hide()
        
        self.spin_hampel_sigma = QDoubleSpinBox()
        self.spin_hampel_sigma.setRange(0.1, 10.0)
        self.spin_hampel_sigma.setValue(3.0)
        self.spin_hampel_sigma.setDecimals(1)
        self.spin_hampel_sigma.hide()
        self.label_hampel_sigma = QLabel("Sigma:")
        self.label_hampel_sigma.hide()
        
        # UI for Kalman
        self.spin_kalman_r = QDoubleSpinBox()
        self.spin_kalman_r.setRange(0.0001, 1000.0)
        self.spin_kalman_r.setValue(1.0)
        self.spin_kalman_r.setDecimals(4)
        self.spin_kalman_r.hide()
        self.label_kalman_r = QLabel("R (Meas Noise):")
        self.label_kalman_r.hide()
        
        self.spin_kalman_q = QDoubleSpinBox()
        self.spin_kalman_q.setRange(0.000001, 10.0)
        self.spin_kalman_q.setValue(1e-4)
        self.spin_kalman_q.setDecimals(6)
        self.spin_kalman_q.hide()
        self.label_kalman_q = QLabel("Q (Proc Noise):")
        self.label_kalman_q.hide()
        
        form_layout.addRow("Phương pháp:", self.cb_filter_impl)
        self.label_f_type = QLabel("Dạng lọc:")
        form_layout.addRow(self.label_f_type, self.cb_filter_type)
        self.label_order = QLabel("Bậc / Taps:")
        form_layout.addRow(self.label_order, self.spin_order)
        self.label_cutoff1 = QLabel("Tần số cắt 1:")
        form_layout.addRow(self.label_cutoff1, self.spin_cutoff1)
        form_layout.addRow(self.label_cutoff2, self.spin_cutoff2)
        
        form_layout.addRow(self.label_hampel_window, self.spin_hampel_window)
        form_layout.addRow(self.label_hampel_sigma, self.spin_hampel_sigma)
        
        form_layout.addRow(self.label_kalman_r, self.spin_kalman_r)
        form_layout.addRow(self.label_kalman_q, self.spin_kalman_q)
        
        self.spin_sg_window = QSpinBox()
        self.spin_sg_window.setRange(3, 999)
        self.spin_sg_window.setSingleStep(2)
        self.spin_sg_window.setValue(11)
        self.spin_sg_window.hide()
        self.label_sg_window = QLabel("Window Length:")
        self.label_sg_window.hide()
        
        self.spin_sg_poly = QSpinBox()
        self.spin_sg_poly.setRange(1, 10)
        self.spin_sg_poly.setValue(3)
        self.spin_sg_poly.hide()
        self.label_sg_poly = QLabel("Poly Order:")
        self.label_sg_poly.hide()
        
        form_layout.addRow(self.label_sg_window, self.spin_sg_window)
        form_layout.addRow(self.label_sg_poly, self.spin_sg_poly)
        
        # Auto Preview
        self.cb_filter_invert = QCheckBox("Lật ngược tín hiệu (Invert)")
        self.cb_auto_preview = QCheckBox("Tự động xem trước (Auto-Preview)")
        self.cb_auto_preview.setChecked(True)
        self.cb_show_response = QCheckBox("Hiển thị Đồ thị Đáp ứng (Freq/Phase)")
        self.cb_show_response.setChecked(True)
        control_layout.addWidget(group_design)
        control_layout.addWidget(self.cb_filter_invert)
        control_layout.addWidget(self.cb_auto_preview)
        control_layout.addWidget(self.cb_show_response)
        
        self.btn_apply_filter = QPushButton("Áp dụng Lọc & Xem đáp ứng")
        self.btn_apply_filter.clicked.connect(self.apply_filter)
        self.btn_apply_filter.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        
        group_export = QGroupBox("Xuất kết quả")
        export_layout = QVBoxLayout(group_export)
        self.btn_export_c = QPushButton("Xuất Code C/C++ (Hệ số)")
        self.btn_export_c.clicked.connect(self.export_c_code)
        self.btn_export_csv = QPushButton("Xuất dữ liệu sau lọc (CSV)")
        self.btn_export_csv.clicked.connect(self.export_filtered_csv)
        export_layout.addWidget(self.btn_export_c)
        export_layout.addWidget(self.btn_export_csv)
        
        control_layout.addWidget(self.btn_apply_filter)
        control_layout.addWidget(group_export)
        control_layout.addStretch()
        
        # Connect signals for auto-preview
        for widget in [self.cb_filter_impl, self.cb_filter_type, self.spin_order, self.spin_cutoff1, 
                       self.spin_cutoff2, self.spin_hampel_window, self.spin_hampel_sigma, 
                       self.spin_kalman_r, self.spin_kalman_q, self.spin_sg_window, self.spin_sg_poly]:
            if isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self.on_param_changed)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(self.on_param_changed)
                
        self.cb_auto_preview.stateChanged.connect(self.on_param_changed)
        self.cb_filter_invert.stateChanged.connect(self.on_param_changed)
        self.cb_show_response.stateChanged.connect(self.on_show_response_changed)
        
        # Khu vực đồ thị (Phải)
        plots_splitter = QSplitter(Qt.Orientation.Vertical)
        
        self.freq_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self.freq_plot = pg.PlotWidget(title="Đáp ứng Biên độ (Magnitude)")
        self.freq_plot.showGrid(x=True, y=True, alpha=0.3)
        self.freq_plot.setLabel('bottom', "Tần số (Hz)")
        self.freq_plot.setLabel('left', "Biên độ (dB)")
        self.freq_curve = self.freq_plot.plot([], [], pen=pg.mkPen('b', width=2))
        
        self.phase_plot = pg.PlotWidget(title="Đáp ứng Pha (Phase)")
        self.phase_plot.showGrid(x=True, y=True, alpha=0.3)
        self.phase_plot.setLabel('bottom', "Tần số (Hz)")
        self.phase_plot.setLabel('left', "Pha (Radians)")
        self.phase_curve = self.phase_plot.plot([], [], pen=pg.mkPen('r', width=2))
        
        self.freq_splitter.addWidget(self.freq_plot)
        self.freq_splitter.addWidget(self.phase_plot)
        
        time_splitter = QSplitter(Qt.Orientation.Vertical)
        
        self.orig_time_plot = pg.PlotWidget(title="Tín hiệu Gốc (Chưa lọc)")
        self.orig_time_plot.showGrid(x=True, y=True, alpha=0.3)
        self.orig_time_plot.setLabel('left', "Biên độ")
        self.orig_curve = self.orig_time_plot.plot([], [], pen=pg.mkPen((150, 150, 150), width=1.5))
        
        self.filt_time_plot = pg.PlotWidget(title="Tín hiệu Đã lọc")
        self.filt_time_plot.showGrid(x=True, y=True, alpha=0.3)
        self.filt_time_plot.setLabel('bottom', "Mẫu (Samples)")
        self.filt_time_plot.setLabel('left', "Biên độ")
        self.filt_time_plot.setXLink(self.orig_time_plot)
        self.filt_curve = self.filt_time_plot.plot([], [], pen=pg.mkPen('purple', width=1.5))
        
        time_splitter.addWidget(self.orig_time_plot)
        time_splitter.addWidget(self.filt_time_plot)
        
        plots_splitter.addWidget(self.freq_splitter)
        plots_splitter.addWidget(time_splitter)
        plots_splitter.setSizes([300, 400])
        
        layout.addWidget(control_panel)
        layout.addWidget(plots_splitter)

    
    def create_default_filter_layer(self, impl="IIR (Butterworth)"):
        return {
            'impl': impl,
            'type': "Lowpass",
            'order': 4 if "IIR" in impl else 101,
            'cutoff1': 5.0,
            'cutoff2': 10.0,
            'hampel_window': 10,
            'hampel_sigma': 3.0,
            'kalman_r': 1.0,
            'kalman_q': 1e-4,
            'sg_window': 11,
            'sg_poly': 3,
            'is_active': True,
            'b': None, 'a': None, 'sos': None
        }

    def add_pipeline_filter(self):
        layer = {
            'impl': self.cb_filter_impl.currentText(),
            'type': self.cb_filter_type.currentText(),
            'order': self.spin_order.value(),
            'cutoff1': self.spin_cutoff1.value(),
            'cutoff2': self.spin_cutoff2.value(),
            'hampel_window': self.spin_hampel_window.value(),
            'hampel_sigma': self.spin_hampel_sigma.value(),
            'kalman_r': self.spin_kalman_r.value(),
            'kalman_q': self.spin_kalman_q.value(),
            'sg_window': self.spin_sg_window.value(),
            'sg_poly': self.spin_sg_poly.value(),
            'is_active': True,
            'b': None, 'a': None, 'sos': None
        }
        self.filter_pipeline.append(layer)
        self.update_pipeline_list()
        self.list_pipeline.setCurrentRow(len(self.filter_pipeline) - 1)
        if self.cb_auto_preview.isChecked():
            self.apply_filter(silent=True)
            
    def remove_pipeline_filter(self):
        idx = self.list_pipeline.currentRow()
        if 0 <= idx < len(self.filter_pipeline):
            self.filter_pipeline.pop(idx)
            self.update_pipeline_list()
            self.list_pipeline.setCurrentRow(max(0, idx - 1))
            if self.cb_auto_preview.isChecked():
                self.apply_filter(silent=True)

    def move_pipeline_filter_up(self):
        idx = self.list_pipeline.currentRow()
        if idx > 0:
            self.filter_pipeline[idx], self.filter_pipeline[idx-1] = self.filter_pipeline[idx-1], self.filter_pipeline[idx]
            self.update_pipeline_list()
            self.list_pipeline.setCurrentRow(idx - 1)
            if self.cb_auto_preview.isChecked():
                self.apply_filter(silent=True)

    def move_pipeline_filter_down(self):
        idx = self.list_pipeline.currentRow()
        if 0 <= idx < len(self.filter_pipeline) - 1:
            self.filter_pipeline[idx], self.filter_pipeline[idx+1] = self.filter_pipeline[idx+1], self.filter_pipeline[idx]
            self.update_pipeline_list()
            self.list_pipeline.setCurrentRow(idx + 1)
            if self.cb_auto_preview.isChecked():
                self.apply_filter(silent=True)

    def update_pipeline_list(self):
        self.list_pipeline.blockSignals(True)
        self.list_pipeline.clear()
        for i, layer in enumerate(self.filter_pipeline):
            status = "[v]" if layer['is_active'] else "[ ]"
            name = f"{status} {layer['impl']} - {layer['type']}"
            if layer['impl'] in ["Hampel", "Kalman (1D)", "Savitzky-Golay"]:
                name = f"{status} {layer['impl']}"
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if layer['is_active'] else Qt.CheckState.Unchecked)
            self.list_pipeline.addItem(item)
            
        if 0 <= self.current_filter_idx < len(self.filter_pipeline):
            self.list_pipeline.setCurrentRow(self.current_filter_idx)
            
        self.list_pipeline.blockSignals(False)

    def on_pipeline_selection_changed(self, idx):
        if 0 <= idx < len(self.filter_pipeline):
            self.current_filter_idx = idx
            self.populate_ui_from_layer(self.filter_pipeline[idx])
        else:
            self.current_filter_idx = -1

    def on_pipeline_item_changed(self, item):
        idx = self.list_pipeline.row(item)
        if 0 <= idx < len(self.filter_pipeline):
            is_checked = item.checkState() == Qt.CheckState.Checked
            if self.filter_pipeline[idx]['is_active'] != is_checked:
                self.filter_pipeline[idx]['is_active'] = is_checked
                self.update_pipeline_list()
                if self.cb_auto_preview.isChecked():
                    self.apply_filter(silent=True)

    def populate_ui_from_layer(self, layer):
        self._is_populating = True
        self.cb_filter_impl.setCurrentText(layer['impl'])
        self.cb_filter_type.setCurrentText(layer['type'])
        self.spin_order.setValue(layer['order'])
        self.spin_cutoff1.setValue(layer['cutoff1'])
        self.spin_cutoff2.setValue(layer['cutoff2'])
        self.spin_hampel_window.setValue(layer['hampel_window'])
        self.spin_hampel_sigma.setValue(layer['hampel_sigma'])
        self.spin_kalman_r.setValue(layer['kalman_r'])
        self.spin_kalman_q.setValue(layer['kalman_q'])
        self.spin_sg_window.setValue(layer['sg_window'])
        self.spin_sg_poly.setValue(layer['sg_poly'])
        self._is_populating = False
        self.on_filter_impl_changed()

    def update_layer_from_ui(self):
        if 0 <= self.current_filter_idx < len(self.filter_pipeline):
            layer = self.filter_pipeline[self.current_filter_idx]
            layer['impl'] = self.cb_filter_impl.currentText()
            layer['type'] = self.cb_filter_type.currentText()
            layer['order'] = self.spin_order.value()
            layer['cutoff1'] = self.spin_cutoff1.value()
            layer['cutoff2'] = self.spin_cutoff2.value()
            layer['hampel_window'] = self.spin_hampel_window.value()
            layer['hampel_sigma'] = self.spin_hampel_sigma.value()
            layer['kalman_r'] = self.spin_kalman_r.value()
            layer['kalman_q'] = self.spin_kalman_q.value()
            layer['sg_window'] = self.spin_sg_window.value()
            layer['sg_poly'] = self.spin_sg_poly.value()
            self.update_pipeline_list()

    def on_filter_impl_changed(self, *args):
        impl = self.cb_filter_impl.currentText()
        is_iir_fir = "IIR" in impl or "FIR" in impl
        is_hampel = "Hampel" in impl
        is_kalman = "Kalman" in impl
        is_sg = "Savitzky" in impl
        
        self.cb_filter_type.setVisible(is_iir_fir)
        self.label_f_type.setVisible(is_iir_fir)
        self.spin_order.setVisible(is_iir_fir)
        self.label_order.setVisible(is_iir_fir)
        self.spin_cutoff1.setVisible(is_iir_fir)
        self.label_cutoff1.setVisible(is_iir_fir)
        
        if "IIR" in impl and self.spin_order.value() > 10:
            self.spin_order.blockSignals(True)
            self.spin_order.setValue(4)
            self.spin_order.blockSignals(False)
        elif "FIR" in impl and self.spin_order.value() <= 10:
            self.spin_order.blockSignals(True)
            self.spin_order.setValue(101)
            self.spin_order.blockSignals(False)
        
        if is_iir_fir and self.cb_filter_type.currentText() in ["Bandpass", "Bandstop"]:
            self.spin_cutoff2.show()
            self.label_cutoff2.show()
        else:
            self.spin_cutoff2.hide()
            self.label_cutoff2.hide()
            
        self.spin_hampel_window.setVisible(is_hampel)
        self.label_hampel_window.setVisible(is_hampel)
        self.spin_hampel_sigma.setVisible(is_hampel)
        self.label_hampel_sigma.setVisible(is_hampel)
        
        self.spin_kalman_r.setVisible(is_kalman)
        self.label_kalman_r.setVisible(is_kalman)
        self.spin_kalman_q.setVisible(is_kalman)
        self.label_kalman_q.setVisible(is_kalman)
        
        self.spin_sg_window.setVisible(is_sg)
        self.label_sg_window.setVisible(is_sg)
        self.spin_sg_poly.setVisible(is_sg)
        self.label_sg_poly.setVisible(is_sg)
        
        if not is_iir_fir:
            self.freq_plot.setTitle(f"Đáp ứng Biên độ (Không áp dụng cho {impl})")
            self.phase_plot.setTitle(f"Đáp ứng Pha (Không áp dụng cho {impl})")
            self.freq_curve.setData([], [])
            self.phase_curve.setData([], [])
        else:
            self.freq_plot.setTitle("Đáp ứng Biên độ (Magnitude)")
            self.phase_plot.setTitle("Đáp ứng Pha (Phase)")

    def on_filter_type_changed(self, *args):
        f_type = self.cb_filter_type.currentText()
        if f_type in ["Bandpass", "Bandstop"] and ("IIR" in self.cb_filter_impl.currentText() or "FIR" in self.cb_filter_impl.currentText()):
            self.label_cutoff2.show()
            self.spin_cutoff2.show()
        else:
            self.label_cutoff2.hide()
            self.spin_cutoff2.hide()
            
        if self.cb_auto_preview.isChecked():
            self.design_filter(silent=True)
            self.apply_filter(silent=True)
            
    
    def on_show_response_changed(self, state):
        is_visible = self.cb_show_response.isChecked()
        self.freq_splitter.setVisible(is_visible)

    def on_param_changed(self, *args):
        if getattr(self, '_is_populating', False): return
        self.update_layer_from_ui()
        if self.cb_auto_preview.isChecked():
            self.apply_filter(silent=True)

    def design_filter(self, *args, silent=False):
        # Hàm này chỉ gọi apply_filter vì apply_filter giờ đã bao trọn việc vẽ Response
        self.apply_filter(silent=silent)

    def apply_filter(self, *args, silent=False):
        if isinstance(silent, bool) == False: silent = False
        
        if self.pure_offline_signal is None:
            if not silent: QMessageBox.warning(self, "Lỗi", "Chưa có dữ liệu. Vui lòng tải file CSV ở Tab Offline Analysis trước.")
            return
            
        signal = -self.pure_offline_signal if self.cb_filter_invert.isChecked() else self.pure_offline_signal
        
        # 1. Tính toán Frequency / Phase Response
        if self.cb_show_response.isChecked():
            fs = 100.0
            total_h = np.ones(8000, dtype=complex)
            w_axis = None
            has_freq_response = False
            
            for layer in self.filter_pipeline:
                if not layer.get('is_active', True): continue
                impl = layer['impl']
                f_type = layer['type']
                order = layer['order']
                cut1 = layer['cutoff1']
                cut2 = layer['cutoff2']
                
                if "IIR" in impl or "FIR" in impl:
                    nyq = 0.5 * fs
                    if f_type in ["Bandpass", "Bandstop"]:
                        if cut1 >= cut2:
                            continue
                        Wn = [cut1 / nyq, cut2 / nyq]
                    else:
                        Wn = cut1 / nyq
                        
                    if "IIR" in impl:
                        try:
                            layer['sos'] = butter(order, Wn, btype=f_type.lower(), output='sos')
                            w, h = sosfreqz(layer['sos'], worN=8000)
                            total_h *= h
                            w_axis = w
                            has_freq_response = True
                        except Exception:
                            pass
                    else: # FIR
                        try:
                            numtaps = order
                            if f_type.lower() in ['highpass', 'bandstop'] and numtaps % 2 == 0:
                                numtaps += 1
                            if f_type in ["Bandpass", "Bandstop"]:
                                cutoff = [cut1, cut2]
                            else:
                                cutoff = cut1
                            pass_zero = True
                            if f_type == "Highpass": pass_zero = False
                            elif f_type == "Bandpass": pass_zero = False
                            
                            b = firwin(numtaps, cutoff, pass_zero=pass_zero, fs=fs)
                            a = [1.0]
                            layer['b'] = b
                            layer['a'] = a
                            w, h = freqz(b, a, worN=8000)
                            total_h *= h
                            w_axis = w
                            has_freq_response = True
                        except Exception:
                            pass
                            
            if has_freq_response and w_axis is not None:
                f_hz = (w_axis * fs) / (2 * np.pi)
                mag_db = 20 * np.log10(np.maximum(np.abs(total_h), 1e-5))
                phase_rad = np.unwrap(np.angle(total_h))
                self.freq_curve.setData(f_hz, mag_db)
                self.phase_curve.setData(f_hz, phase_rad)
            else:
                self.freq_curve.setData([], [])
                self.phase_curve.setData([], [])

        # 2. Xử lý tín hiệu qua Pipeline
        filtered_signal = np.copy(signal)
        try:
            for layer in self.filter_pipeline:
                if not layer.get('is_active', True): continue
                impl = layer['impl']
                
                if "IIR" in impl:
                    if layer.get('sos') is not None:
                        padlen = min(3 * 2 * len(layer['sos']), len(filtered_signal) - 2)
                        if padlen < 0: padlen = 0
                        filtered_signal = sosfiltfilt(layer['sos'], filtered_signal, padlen=padlen)
                elif "FIR" in impl:
                    if layer.get('b') is not None:
                        try:
                            filtered_signal = filtfilt(layer['b'], layer['a'], filtered_signal)
                        except ValueError:
                            padlen = max(0, len(filtered_signal) - 2)
                            if padlen > 0:
                                filtered_signal = filtfilt(layer['b'], layer['a'], filtered_signal, padlen=padlen)
                elif "Hampel" in impl:
                    window_size = layer['hampel_window']
                    n_sigmas = layer['hampel_sigma']
                    rolling_median = median_filter(filtered_signal, size=window_size)
                    diff = np.abs(filtered_signal - rolling_median)
                    rolling_mad = median_filter(diff, size=window_size)
                    outliers = diff > (n_sigmas * rolling_mad)
                    filtered_signal = np.where(outliers, rolling_median, filtered_signal)
                elif "Kalman" in impl:
                    R = layer['kalman_r']
                    Q = layer['kalman_q']
                    n = len(filtered_signal)
                    temp_sig = np.zeros(n)
                    x_est = filtered_signal[0]
                    P_est = 1.0
                    for i in range(n):
                        x_pred = x_est
                        P_pred = P_est + Q
                        K = P_pred / (P_pred + R)
                        x_est = x_pred + K * (filtered_signal[i] - x_pred)
                        P_est = (1 - K) * P_pred
                        temp_sig[i] = x_est
                    filtered_signal = temp_sig
                elif "Savitzky" in impl:
                    window_length = layer['sg_window']
                    if window_length % 2 == 0: window_length += 1
                    polyorder = layer['sg_poly']
                    if polyorder >= window_length: polyorder = window_length - 1
                    filtered_signal = savgol_filter(filtered_signal, window_length, polyorder)
                    
            if np.any(np.isnan(filtered_signal)) or np.any(np.isinf(filtered_signal)):
                self.filt_curve.setData([])
                self.filt_time_plot.setTitle("Tín hiệu Đã lọc - LỖI: Bộ lọc không ổn định!")
                if not silent:
                    QMessageBox.warning(self, "Lỗi", "Tín hiệu sau lọc chứa NaN/Inf. Một trong số các bộ lọc không ổn định.")
                return
            
            self.filt_time_plot.setTitle("Tín hiệu Đã lọc (Pipeline)")
            self.filtered_signal_cache = filtered_signal
            self.offline_signal = filtered_signal # Lưu đè offline signal cho Tab Analysis
            self.orig_curve.setData(signal)
            self.filt_curve.setData(filtered_signal)
            
            if not silent: QMessageBox.information(self, "Hoàn tất", f"Đã áp dụng toàn bộ chuỗi lọc.")
            
        except Exception as e:
            self.filt_curve.setData([])
            self.filt_time_plot.setTitle(f"Tín hiệu Đã lọc - LỖI: {str(e)}")
            if not silent: QMessageBox.critical(self, "Lỗi lọc", f"Lỗi trong quá trình lọc tín hiệu: {str(e)}")

    def export_c_code(self):
        code = ""
        count = 0
        for i, layer in enumerate(self.filter_pipeline):
            if not layer.get('is_active', True): continue
            impl = layer['impl']
            if "IIR" in impl or "FIR" in impl:
                count += 1
                code += f"// --- Layer {i+1}: {impl} ({layer['type']}) ---\n"
                
                if "FIR" in impl:
                    b = layer.get('b')
                    if b is None: continue
                    code += f"#define FIR_LAYER{i+1}_TAPS {len(b)}\n"
                    code += f"const float fir_layer{i+1}_taps[FIR_LAYER{i+1}_TAPS] = {{\n    "
                    code += ", ".join([f"{x:.6e}" for x in b])
                    code += "\n};\n\n"
                else:
                    sos = layer.get('sos')
                    if sos is None: continue
                    code += f"#define IIR_LAYER{i+1}_NUM_SECTIONS {len(sos)}\n"
                    code += f"const float iir_layer{i+1}_sos[IIR_LAYER{i+1}_NUM_SECTIONS][6] = {{\n"
                    for sec in sos:
                        code += "    {" + ", ".join([f"{x:.6e}" for x in sec]) + "},\n"
                    code += "};\n\n"
                    
        if count == 0:
            QMessageBox.information(self, "Lưu ý", "Không có bộ lọc FIR/IIR nào đang Active trong Pipeline để xuất C/C++.")
            return
            
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton
        dlg = QDialog(self)
        dlg.setWindowTitle("C/C++ Code (Pipeline)")
        dlg.resize(600, 400)
        vbox = QVBoxLayout(dlg)
        text = QTextEdit()
        text.setPlainText(code)
        text.setReadOnly(True)
        vbox.addWidget(text)
        btn = QPushButton("Đóng")
        btn.clicked.connect(dlg.accept)
        vbox.addWidget(btn)
        dlg.exec()

    def export_filtered_csv(self):
        if not hasattr(self, 'filtered_signal_cache') or self.filtered_signal_cache is None:
            QMessageBox.warning(self, "Lỗi", "Chưa có dữ liệu đã lọc.")
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Lưu file CSV", "filtered_signal.csv", "CSV Files (*.csv)")
        if path:
            try:
                original = -self.pure_offline_signal if self.cb_filter_invert.isChecked() else self.pure_offline_signal
                df = pd.DataFrame({
                    'Original': original,
                    'Filtered': self.filtered_signal_cache
                })
                df.to_csv(path, index=False)
                QMessageBox.information(self, "Thành công", f"Đã lưu kết quả tại: {path}")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể lưu file: {str(e)}")
        
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
        if self.cb_invert_live.isChecked():
            p = -p
        self.data_p[:-1] = self.data_p[1:]
        self.data_p[-1] = p
        self.data_t[:-1] = self.data_t[1:]
        self.data_t[-1] = t
        self.data_s[:-1] = self.data_s[1:]
        self.data_s[-1] = s

    def mouseMoved_live(self, evt):
        pos = evt[0]
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mousePoint = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            x, y = mousePoint.x(), mousePoint.y()
            self.vLine_live.setPos(x)
            self.label_live.setHtml(f"Mẫu: {int(x)}<br>Biên độ: {y:.2f}")
            self.label_live.setPos(x, y)

    def mouseMoved_fft_ppg(self, evt):
        pos = evt[0]
        if self.fft_ppg_plot.sceneBoundingRect().contains(pos):
            mousePoint = self.fft_ppg_plot.plotItem.vb.mapSceneToView(pos)
            x, y = mousePoint.x(), mousePoint.y()
            self.vLine_fft_ppg.setPos(x)
            self.label_fft_ppg.setHtml(f"Mẫu: {int(x)}<br>Biên độ: {y:.2f}")
            self.label_fft_ppg.setPos(x, y)

    def mouseMoved_fft_widget(self, evt):
        pos = evt[0]
        if self.fft_widget.sceneBoundingRect().contains(pos):
            mousePoint = self.fft_widget.plotItem.vb.mapSceneToView(pos)
            x, y = mousePoint.x(), mousePoint.y()
            self.vLine_fft.setPos(x)
            self.label_fft.setHtml(f"Tần số: {x:.2f} Hz<br>Biên độ: {y:.4f}")
            self.label_fft.setPos(x, y)

    def mouseMoved_deriv(self, evt):
        pos = evt[0]
        x = None
        
        if self.deriv_ppg_plot.isVisible() and self.deriv_ppg_plot.vb.sceneBoundingRect().contains(pos):
            mousePoint = self.deriv_ppg_plot.vb.mapSceneToView(pos)
            x, y = mousePoint.x(), mousePoint.y()
            self.label_deriv_ppg.setHtml(f"Mẫu: {int(x)}<br>PPG: {y:.2f}")
            self.label_deriv_ppg.setPos(x, y)
            self.label_deriv_ppg.show()
            self.label_vpg.hide()
            self.label_apg.hide()
        elif self.vpg_plot.isVisible() and self.vpg_plot.vb.sceneBoundingRect().contains(pos):
            mousePoint = self.vpg_plot.vb.mapSceneToView(pos)
            x, y = mousePoint.x(), mousePoint.y()
            self.label_vpg.setHtml(f"Mẫu: {int(x)}<br>VPG: {y:.2f}")
            self.label_vpg.setPos(x, y)
            self.label_vpg.show()
            self.label_deriv_ppg.hide()
            self.label_apg.hide()
        elif self.apg_plot.isVisible() and self.apg_plot.vb.sceneBoundingRect().contains(pos):
            mousePoint = self.apg_plot.vb.mapSceneToView(pos)
            x, y = mousePoint.x(), mousePoint.y()
            self.label_apg.setHtml(f"Mẫu: {int(x)}<br>APG: {y:.2f}")
            self.label_apg.setPos(x, y)
            self.label_apg.show()
            self.label_deriv_ppg.hide()
            self.label_vpg.hide()
            
        if x is not None:
            self.vLine_deriv_ppg.setPos(x)
            self.vLine_vpg.setPos(x)
            self.vLine_apg.setPos(x)

    def toggle_dot_mode(self):
        self.cb_dot_mode.setChecked(not self.cb_dot_mode.isChecked())

    def mouseClicked_deriv(self, evt):
        if evt.button() == Qt.MouseButton.LeftButton:
            pos = evt.scenePos()
            x = None
            y = None
            clicked_plot = None
            for plot in [self.deriv_ppg_plot, self.vpg_plot, self.apg_plot]:
                if plot.isVisible() and plot.vb.sceneBoundingRect().contains(pos):
                    mousePoint = plot.vb.mapSceneToView(pos)
                    x = mousePoint.x()
                    y = mousePoint.y()
                    clicked_plot = plot
                    break
            
            if x is not None:
                if not hasattr(self, 'temp_rulers'):
                    self.temp_rulers = []
                if not hasattr(self, 'redo_rulers'):
                    self.redo_rulers = []
                
                if self.cb_dot_mode.isChecked():
                    scatter = pg.ScatterPlotItem(x=[x], y=[y], size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 255))
                    clicked_plot.addItem(scatter, ignoreBounds=True)
                    self.temp_rulers.append(("dot", scatter, clicked_plot))
                else:
                    # Thước tạm thời màu Cyan
                    pen = pg.mkPen('cyan', width=1.5, style=Qt.PenStyle.DashLine)
                    l1 = pg.InfiniteLine(pos=x, angle=90, pen=pen)
                    l2 = pg.InfiniteLine(pos=x, angle=90, pen=pen)
                    l3 = pg.InfiniteLine(pos=x, angle=90, pen=pen)
                    
                    self.deriv_ppg_plot.addItem(l1, ignoreBounds=True)
                    self.vpg_plot.addItem(l2, ignoreBounds=True)
                    self.apg_plot.addItem(l3, ignoreBounds=True)
                    
                    self.temp_rulers.append(("ruler", l1, l2, l3))
                    
                self.redo_rulers.clear()

    def undo_temp_ruler(self):
        if hasattr(self, 'temp_rulers') and len(self.temp_rulers) > 0:
            if not hasattr(self, 'redo_rulers'):
                self.redo_rulers = []
            group = self.temp_rulers.pop()
            
            if group[0] == "ruler":
                _, l1, l2, l3 = group
                try:
                    self.deriv_ppg_plot.removeItem(l1)
                    self.vpg_plot.removeItem(l2)
                    self.apg_plot.removeItem(l3)
                except:
                    pass
            elif group[0] == "dot":
                _, scatter, plot = group
                try:
                    plot.removeItem(scatter)
                except:
                    pass
                    
            self.redo_rulers.append(group)

    def redo_temp_ruler(self):
        if hasattr(self, 'redo_rulers') and len(self.redo_rulers) > 0:
            if not hasattr(self, 'temp_rulers'):
                self.temp_rulers = []
            group = self.redo_rulers.pop()
            
            if group[0] == "ruler":
                _, l1, l2, l3 = group
                self.deriv_ppg_plot.addItem(l1, ignoreBounds=True)
                self.vpg_plot.addItem(l2, ignoreBounds=True)
                self.apg_plot.addItem(l3, ignoreBounds=True)
            elif group[0] == "dot":
                _, scatter, plot = group
                plot.addItem(scatter, ignoreBounds=True)
                
            self.temp_rulers.append(group)

    def clear_temp_rulers(self):
        if hasattr(self, 'temp_rulers'):
            for group in self.temp_rulers:
                if group[0] == "ruler":
                    _, l1, l2, l3 = group
                    try:
                        self.deriv_ppg_plot.removeItem(l1)
                        self.vpg_plot.removeItem(l2)
                        self.apg_plot.removeItem(l3)
                    except:
                        pass
                elif group[0] == "dot":
                    _, scatter, plot = group
                    try:
                        plot.removeItem(scatter)
                    except:
                        pass
            self.temp_rulers.clear()
        if hasattr(self, 'redo_rulers'):
            self.redo_rulers.clear()

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
            elif 'Original' in df.columns and 'Filtered' in df.columns:
                # File dạng ma_filtered_signal.csv — cho người dùng chọn cột
                from PyQt6.QtWidgets import QInputDialog
                choices = ["Original (Tín hiệu gốc)", "Filtered (Tín hiệu đã lọc)"]
                choice, ok = QInputDialog.getItem(
                    self, "Chọn tín hiệu", 
                    "File chứa cả tín hiệu Gốc và Đã lọc.\nBạn muốn tải cột nào?",
                    choices, 0, False
                )
                if not ok:
                    return
                col = 'Original' if 'Original' in choice else 'Filtered'
                signal = df[col].values
            elif 'Original' in df.columns:
                signal = df['Original'].values
            elif 'Filtered' in df.columns:
                signal = df['Filtered'].values
            else:
                QMessageBox.warning(self, "Lỗi file", "Không tìm thấy cột dữ liệu PPG hợp lệ.\nCác cột được hỗ trợ: 'Gia_tri_PPG', 'Raw_IR', 'Original', 'Filtered'.")
                return
                
            self.offline_signal = signal
            self.pure_offline_signal = np.copy(signal)
            self.offline_filename = os.path.basename(file_path)
            
            self.redraw_offline_raw()
            
            self.btn_analyze.setEnabled(True)
            self.btn_analyze.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
            self.btn_find_peaks.setEnabled(True)
            
            # Tự động áp dụng Pipeline (nếu có) lên file mới
            if len(self.filter_pipeline) > 0 and self.cb_auto_preview.isChecked():
                self.apply_filter(silent=True)
            else:
                # Nếu không có pipeline, chỉ cần update đồ thị gốc ở tab filter
                self.orig_curve.setData(-self.pure_offline_signal if self.cb_filter_invert.isChecked() else self.pure_offline_signal)
                self.filt_curve.setData([])
            
            QMessageBox.information(self, "Đã tải file", "Đã tải dữ liệu thành công. Nhấn 'Bắt đầu Phân tích' để xem FFT và Đạo hàm.")
            
        except Exception as e:
            QMessageBox.critical(self, "Lỗi đọc file", str(e))

    def on_invert_changed(self):
        if self.offline_signal is not None:
            # Nếu đã tải dữ liệu, vẽ lại raw và tự động chạy phân tích lại
            self.redraw_offline_raw()
            self.analyze_data()

    def redraw_offline_raw(self):
        # Reset đồ thị
        self.fft_ppg_plot.clear()
        self.deriv_ppg_plot.clear()
        self.vpg_plot.clear()
        self.apg_plot.clear()
        self.fft_curve.setData([], []) # Xóa phổ FFT cũ
        
        if hasattr(self, 'temp_rulers'):
            self.temp_rulers.clear()
        if hasattr(self, 'redo_rulers'):
            self.redo_rulers.clear()
            
        signal = -self.offline_signal if self.cb_invert_offline.isChecked() else self.offline_signal
        
        # Chỉ vẽ tín hiệu gốc lên 2 Plot Time Domain
        self.fft_ppg_plot.plot(signal, pen=pg.mkPen('purple', width=1.5))
        self.fft_ppg_plot.setTitle(f"Tín hiệu PPG Thô - {self.offline_filename} ({len(signal)} mẫu)")
        self.fft_ppg_plot.addItem(self.vLine_fft_ppg, ignoreBounds=True)
        self.fft_ppg_plot.addItem(self.label_fft_ppg, ignoreBounds=True)
        
        self.deriv_ppg_plot.plot(signal, pen=pg.mkPen('purple', width=1.5))
        self.deriv_ppg_plot.setTitle(f"Tín hiệu PPG Thô - {self.offline_filename} ({len(signal)} mẫu)")
        self.deriv_ppg_plot.addItem(self.vLine_deriv_ppg, ignoreBounds=True)
        self.deriv_ppg_plot.addItem(self.label_deriv_ppg, ignoreBounds=True)
        
        self.vpg_plot.addItem(self.vLine_vpg, ignoreBounds=True)
        self.vpg_plot.addItem(self.label_vpg, ignoreBounds=True)
        
        self.apg_plot.addItem(self.vLine_apg, ignoreBounds=True)
        self.apg_plot.addItem(self.label_apg, ignoreBounds=True)

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
        if len(signal) < 100:
            QMessageBox.warning(self, "Lỗi phân tích", "Dữ liệu quá ngắn (ít hơn 100 mẫu). Không thể tính toán FFT hoặc đạo hàm.")
            return

        # Lật ngược tín hiệu nếu có tick
        if self.cb_invert_offline.isChecked():
            signal = -signal

        signal_no_dc = signal - np.mean(signal)
        
        # 1. Tính toán VPG và APG
        vpg = np.gradient(signal_no_dc)
        apg = np.gradient(vpg)
        
        self.vpg_plot.clear()
        self.apg_plot.clear()
        
        self.vpg_plot.plot(vpg, pen=pg.mkPen('g', width=1.5))
        self.apg_plot.plot(apg, pen=pg.mkPen('orange', width=1.5))
        
        self.vpg_plot.addItem(self.vLine_vpg, ignoreBounds=True)
        self.vpg_plot.addItem(self.label_vpg, ignoreBounds=True)
        
        self.apg_plot.addItem(self.vLine_apg, ignoreBounds=True)
        self.apg_plot.addItem(self.label_apg, ignoreBounds=True)
        
        # Reset lại 2 plot gốc trước khi vẽ đè
        self.fft_ppg_plot.clear()
        self.fft_ppg_plot.plot(signal, pen=pg.mkPen('purple', width=1.5))
        self.fft_ppg_plot.addItem(self.vLine_fft_ppg, ignoreBounds=True)
        self.fft_ppg_plot.addItem(self.label_fft_ppg, ignoreBounds=True)
        
        self.deriv_ppg_plot.clear()
        self.deriv_ppg_plot.plot(signal, pen=pg.mkPen('purple', width=1.5))
        self.deriv_ppg_plot.addItem(self.vLine_deriv_ppg, ignoreBounds=True)
        self.deriv_ppg_plot.addItem(self.label_deriv_ppg, ignoreBounds=True)
        
        # 2. Tính toán và vẽ FFT
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

    def find_peaks_bpm(self):
        """Tìm đỉnh (Peaks) và tính nhịp tim (BPM) trên tín hiệu PPG."""
        if self.offline_signal is None:
            QMessageBox.warning(self, "Lỗi", "Chưa có dữ liệu. Vui lòng tải file CSV trước.")
            return
            
        signal = self.offline_signal
        if len(signal) < 100:
            QMessageBox.warning(self, "Lỗi", "Dữ liệu quá ngắn (ít hơn 100 mẫu).")
            return

        # Lật ngược tín hiệu nếu có tick
        if self.cb_invert_offline.isChecked():
            signal = -signal

        signal_no_dc = signal - np.mean(signal)
        
        # Tìm điểm đỉnh (Peaks) trên PPG gốc
        peaks, properties = find_peaks(signal_no_dc, distance=50, prominence=np.max(signal_no_dc)*0.1)
        
        # Vẽ lại plot gốc rồi đè chấm đỏ lên
        self.fft_ppg_plot.clear()
        self.fft_ppg_plot.plot(signal, pen=pg.mkPen('purple', width=1.5))
        self.fft_ppg_plot.addItem(self.vLine_fft_ppg, ignoreBounds=True)
        self.fft_ppg_plot.addItem(self.label_fft_ppg, ignoreBounds=True)
        
        self.deriv_ppg_plot.clear()
        self.deriv_ppg_plot.plot(signal, pen=pg.mkPen('purple', width=1.5))
        self.deriv_ppg_plot.addItem(self.vLine_deriv_ppg, ignoreBounds=True)
        self.deriv_ppg_plot.addItem(self.label_deriv_ppg, ignoreBounds=True)
        
        # Thêm chấm đỏ đánh dấu đỉnh vào cả 2 tab
        scatter1 = pg.ScatterPlotItem(
            x=peaks, y=signal[peaks], 
            size=8, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 255)
        )
        scatter2 = pg.ScatterPlotItem(
            x=peaks, y=signal[peaks], 
            size=8, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 255)
        )
        self.fft_ppg_plot.addItem(scatter1)
        self.deriv_ppg_plot.addItem(scatter2)
        
        # Thêm đường dóng (Dashed Vertical Line) từ Đỉnh xuống đạo hàm
        for p in peaks:
            line_pen = pg.mkPen('r', width=1, style=Qt.PenStyle.DashLine)
            self.deriv_ppg_plot.addItem(pg.InfiniteLine(pos=p, angle=90, pen=line_pen))
            self.vpg_plot.addItem(pg.InfiniteLine(pos=p, angle=90, pen=line_pen))
            self.apg_plot.addItem(pg.InfiniteLine(pos=p, angle=90, pen=line_pen))
        
        # Tính BPM dựa trên khoảng cách giữa các đỉnh
        bpm = 0
        if len(peaks) > 1:
            intervals = np.diff(peaks) * 0.01 # Fs = 100Hz
            bpm = 60.0 / np.mean(intervals)
        
        title_str = f"Tín hiệu PPG - {self.offline_filename} | Đỉnh: {len(peaks)} | Nhịp tim: {bpm:.1f} BPM"
        self.fft_ppg_plot.setTitle(title_str)
        self.deriv_ppg_plot.setTitle(title_str)
        
        QMessageBox.information(self, "Kết quả", f"Tìm thấy {len(peaks)} đỉnh.\nNhịp tim trung bình: {bpm:.1f} BPM")

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
