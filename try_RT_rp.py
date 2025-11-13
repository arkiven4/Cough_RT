#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time, os, logging, json, glob, re, socket, requests
from datetime import datetime
from threading import Thread, Lock
from collections import deque
from types import SimpleNamespace

import tkinter as tk
from tkinter import ttk
from tkinter import font
import threading

import numpy as np
import alsaaudio as alsa
import soundfile as sf
import tempfile

# Import matplotlib libraries
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib import style

from utils import segment_cough

os.makedirs("Recorded_Data/automatic", exist_ok=True)
os.makedirs("Recorded_Data/soliced", exist_ok=True)
os.makedirs("logs/", exist_ok=True)

json_path = 'Recorded_Data/last_send.json'
if not os.path.exists(json_path):
    with open(json_path, 'w') as outfile:
        json.dump({
            "last_send_automatic": None,
            "last_send_soliced": None,
        }, outfile)

timestamp = datetime.now().strftime("%d-%m-%Y_%H%M")
log_filename = f"log_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler(f"logs/{log_filename}")]
)

GLOBAL_CONFIG = {}
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.json')
with open(config_path) as data_file:    
    GLOBAL_CONFIG = json.load(data_file, object_hook=lambda d: SimpleNamespace(**d))


class CoughTk():
    """CoughAnalyzer Program with GUI
    """

    DarkTheme = True

    CHANNELS = 2
    PERIOD_SIZE = 1024
    SAMPLE_RATE = 44100
    WINDOW_DURATION = 4
    OVERLAP_DURATION = 0.3
    STEP_DURATION = WINDOW_DURATION - OVERLAP_DURATION

    DEVICE_SOUND =  GLOBAL_CONFIG.DEVICE_SOUND # 'dmic_sv' "plughw:2,0"
    BTN1_FILE =  GLOBAL_CONFIG.BTN1_FILE # "gpio12" "/sys/class/gpio/gpio5/value"
    BTN2_FILE =  GLOBAL_CONFIG.BTN2_FILE # "gpio12" "/sys/class/gpio/gpio5/value"
    BTN3_FILE =  GLOBAL_CONFIG.BTN3_FILE # "gpio12" "/sys/class/gpio/gpio6/value"
    BTN4_FILE =  GLOBAL_CONFIG.BTN4_FILE # "gpio12" "/sys/class/gpio/gpio6/value"
    RECORD_LENGTH = int(GLOBAL_CONFIG.RECORD_LENGTH * SAMPLE_RATE)
    AUDIO_POINT_START = round(0.3 * SAMPLE_RATE)

    DEVICE_WLAN = GLOBAL_CONFIG.DEVICE_WLAN
    WEBPANEL_ROOT = GLOBAL_CONFIG.WEBPANEL_ROOT
    SERVER_DOMAIN = GLOBAL_CONFIG.SERVER_DOMAIN
    DEVICE_ID = GLOBAL_CONFIG.DEVICE_ID

    def __init__(self):
        super(CoughTk, self).__init__()

        # Main Window
        self.window = tk.Tk()
        self.window.geometry("480x320")
        self.window.title("TBCare - CoughAnalyzer")
        self.current_page = 1

        self.main_frame = tk.Frame(self.window)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.init_variables()

        self.create_home_page()
        self.create_analyzer_page()
        self.create_prediction_page()
        self.create_info_page()

        self.show_page(self.current_page)

        if self.DarkTheme:
            self.configure_dark_theme()

        self.start_background_processes()
        self.window.mainloop()

    def init_variables(self):
        """Initialize shared variables for both pages"""
        # Font
        self.wndfont = font.Font(self.window, family="Liberation Mono", size=12)
        self.title_font = font.Font(self.window, family="Liberation Mono", size=16, weight="bold")
        self.subtitle_font = font.Font(self.window, family="Liberation Mono", size=14, weight="bold")
        self.small_font = font.Font(self.window, family="Liberation Mono", size=9)
        
        # Status variablessxxxsx
        self.internet_status = tk.StringVar()
        self.internet_status.set("Offline")
        self.EdgeIP = tk.StringVar()
        self.EdgeIP.set(self.getwlanip())
        self.solicoughcount = tk.StringVar()
        self.solicoughcount.set("Longi: 0 || Solic: 0")
        self.txtrecord = tk.StringVar()
        self.txtrecord.set("Recording: Automatic")

        self.current_patient = tk.StringVar()
        self.current_patient.set("Not Set Yet")

        # Battery status variable
        self.battery_status = tk.StringVar()
        self.battery_status.set("🔋100%")
        
        # Audio processing variables
        self.audio_buffer = deque()
        self.buffer_lock = Lock()
        self.next_time = time.time()
        self.RECORD_FLAG = False
        self.last_cough_np = np.zeros((1000,))
        self.recording_start_time = None
        self.recording_time_thread = None
        self.recording_time_stop_event = threading.Event()
        self.buffer_size = int(self.STEP_DURATION * self.SAMPLE_RATE)
        self.window_size = int(self.WINDOW_DURATION * self.SAMPLE_RATE)

        # Initialize a
        # udio system
        self.pcm = alsa.PCM(alsa.PCM_CAPTURE, alsa.PCM_NORMAL,
                       channels=self.CHANNELS, rate=self.SAMPLE_RATE, format=alsa.PCM_FORMAT_S16_LE,
                       periodsize=self.PERIOD_SIZE, device=self.DEVICE_SOUND)

    def create_status_bar(self, parent_frame):
        """Create a universal status bar for all pages"""
        status_bar = tk.Frame(parent_frame)
        status_bar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        # IP Address (Left)
        ip_label = tk.Label(status_bar, textvariable=self.EdgeIP, font=self.small_font)
        ip_label.pack(side=tk.LEFT)

        # Info
        info_label = tk.Label(status_bar, text="Tim PKM-KC TBCare ITS", font=self.small_font)
        info_label.pack(side=tk.LEFT, padx=(75, 0))
                
        # Battery Status (Right)
        battery_label = tk.Label(status_bar, textvariable=self.battery_status, font=self.small_font)
        battery_label.pack(side=tk.RIGHT)
        
        # Connection Status (Center)
        connection_label = tk.Label(status_bar, textvariable=self.internet_status, font=self.small_font)
        connection_label.pack(side=tk.RIGHT, padx=(0, 5))
        
        # Store references for dark theme configuration
        return [ip_label, info_label, battery_label, connection_label]

    def create_home_page(self):
        """Create the home page with device status and instructions"""
        self.home_frame = tk.Frame(self.main_frame)

        # Status Bar
        self.home_status_widgets = self.create_status_bar(self.home_frame)
        
        # Separator
        separator = ttk.Separator(self.home_frame, orient='horizontal')
        separator.pack(fill=tk.X, pady=2)
        
        # Title
        title_label = tk.Label(self.home_frame, text="🫁 TBCare Device", font=self.title_font)
        title_label.pack(pady=10)
        
        # Instructions Frame
        instructions_frame = tk.Frame(self.home_frame)
        instructions_frame.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)
        
        # Button Instructions
        btn_frame = tk.Frame(instructions_frame)
        btn_frame.pack(fill=tk.X)
        
        # Button 1 instruction
        btn0_frame = tk.Frame(btn_frame)
        btn0_frame.pack(fill=tk.X, pady=3)
        btn0_label = tk.Label(btn0_frame, text="🔴 Button 1:", font=self.wndfont)
        btn0_label.pack(side=tk.LEFT)
        btn0_desc = tk.Label(btn0_frame, text="Active & Passive Recording Page", font=self.wndfont)
        btn0_desc.pack(side=tk.LEFT, padx=(10, 0))

        # Button 1 instruction
        btn1_frame = tk.Frame(btn_frame)
        btn1_frame.pack(fill=tk.X, pady=3)
        btn1_label = tk.Label(btn1_frame, text="🟡 Button 2:", font=self.wndfont)
        btn1_label.pack(side=tk.LEFT)
        btn1_desc = tk.Label(btn1_frame, text="Instant Cough Prediction Page", font=self.wndfont)
        btn1_desc.pack(side=tk.LEFT, padx=(10, 0))
        
        btn2_frame = tk.Frame(btn_frame)
        btn2_frame.pack(fill=tk.X, pady=3)
        btn2_label = tk.Label(btn2_frame, text="🔵 Button 3:", font=self.wndfont)
        btn2_label.pack(side=tk.LEFT)
        btn2_desc = tk.Label(btn2_frame, text="Information Page", font=self.wndfont)
        btn2_desc.pack(side=tk.LEFT, padx=(10, 0))
        
        btn3_frame = tk.Frame(btn_frame)
        btn3_frame.pack(fill=tk.X, pady=3)
        btn3_label = tk.Label(btn3_frame, text="🔴 Button 4:", font=self.wndfont)
        btn3_label.pack(side=tk.LEFT)
        btn3_desc = tk.Label(btn3_frame, text="Show This Page", font=self.wndfont)
        btn3_desc.pack(side=tk.LEFT, padx=(10, 0))
        
        separator = ttk.Separator(instructions_frame, orient='horizontal')
        separator.pack(fill=tk.X, pady=5)

        # Additional info
        info_frame = tk.Frame(instructions_frame)
        info_frame.pack(fill=tk.X, pady=10)
        info_text = tk.Label(info_frame, 
                           text="💡 The device automatically detects and records cough sounds.\n"
                                "💡 Manual recording allows you to capture specific cough samples.\n"
                                "💡 All recordings are saved and can be uploaded when online.",
                           font=font.Font(self.window, family="Liberation Mono", size=9, weight="bold"), 
                           justify=tk.LEFT, wraplength=470)
        info_text.pack()

    def create_info_page(self):
        """Create the home page with device status and instructions"""
        self.info_frame = tk.Frame(self.main_frame)

        # Status Bar
        self.info_status_widgets = self.create_status_bar(self.info_frame)

        # Separator
        separator = ttk.Separator(self.info_frame, orient='horizontal')
        separator.pack(fill=tk.X, pady=2)

        # Title
        title_label = tk.Label(self.info_frame, text="🫁 TB Care Device", font=self.title_font)
        title_label.pack(pady=10)
        
        # Status Frame
        status_frame = tk.Frame(self.info_frame)
        status_frame.pack(pady=10)
        
        # Device Status
        device_status_frame = tk.Frame(status_frame)
        device_status_frame.pack(fill=tk.X, pady=5)
        
        # Info 1
        info1_frame = tk.Frame(device_status_frame)
        info1_frame.pack(fill=tk.X, pady=2)
        info1_label = tk.Label(info1_frame, text="🖥️ Device ID:", font=self.wndfont)
        info1_label.pack(side=tk.LEFT)
        self.home_ip_label = tk.Label(info1_frame, text=self.DEVICE_ID, font=self.wndfont)
        self.home_ip_label.pack(side=tk.LEFT, padx=(10, 0))

        # Info 2
        info2_frame = tk.Frame(device_status_frame)
        info2_frame.pack(fill=tk.X, pady=2)
        info2_label = tk.Label(info2_frame, text="👤 Current Patient:", font=self.wndfont)
        info2_label.pack(side=tk.LEFT)
        self.home_ip_label = tk.Label(info2_frame, textvariable=self.current_patient, font=self.wndfont)
        self.home_ip_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Cough Count
        cough_frame = tk.Frame(device_status_frame)
        cough_frame.pack(fill=tk.X, pady=2)
        cough_label = tk.Label(cough_frame, text="🔢 Cough Count", font=self.wndfont)
        cough_label.pack(side=tk.LEFT)
        self.home_cough_label = tk.Label(cough_frame, textvariable=self.solicoughcount, font=self.wndfont)
        self.home_cough_label.pack(side=tk.LEFT, padx=(10, 0))

    def create_analyzer_page(self):
        """Create the analyzer page (original functionality)"""
        self.analyzer_frame = tk.Frame(self.main_frame)

        # Status Bar
        self.analyzer_status_widgets = self.create_status_bar(self.analyzer_frame)

        # Separator
        separator = ttk.Separator(self.analyzer_frame, orient='horizontal')
        separator.pack(fill=tk.X, pady=2)

        # Title
        title_label = tk.Label(self.analyzer_frame, text="🎤︎︎ Active & Passive Recording", font=self.title_font)
        title_label.pack(pady=5)
        
        # Info Frame
        self.infofrm = tk.Frame(self.analyzer_frame)

        # Solic Coughs Status
        self.lb_patient = tk.Label(self.infofrm, textvariable=self.current_patient)
        self.lb_patient.config(font=self.wndfont)
        self.lb_patient.pack(side=tk.TOP)

        # Solic Coughs Status
        self.lb_coughso = tk.Label(self.infofrm, textvariable=self.solicoughcount)
        self.lb_coughso.config(font=self.wndfont)
        self.lb_coughso.pack(side=tk.TOP)

        # Recording Status
        self.sttrecord = tk.Label(self.infofrm, textvariable=self.txtrecord)
        self.sttrecord.config(font=self.wndfont)
        self.sttrecord.pack(side=tk.TOP)

        # Pack Info Frame
        self.infofrm.pack(side=tk.TOP)

        # Graph Frame
        self.graphfrm = tk.Frame(self.analyzer_frame)
        # Example Figure Plot
        self.fig = Figure(figsize=(5, 1), dpi=96, facecolor='black')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('black')
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(-1, 1)
        self.patch_plot = self.ax.add_patch(patches.Rectangle((-1.0, 0.0), 2.0, 1.0, linewidth=1, edgecolor='black', facecolor='blue'))
        style.use('ggplot')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graphfrm)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.LEFT)
        self.last_updateFigure = time.time()
        self.graphfrm.pack(side=tk.BOTTOM)

        # Graph Data
        self.graphfrm2 = tk.Frame(self.analyzer_frame)
        self.waveform_duration = 2.0  # 2 seconds
        self.waveform_length = 200   # Display points (downsampled)
        self.waveform_samples = int(self.waveform_duration * self.SAMPLE_RATE)  # Total samples for 2 seconds
        self.downsample_ratio = self.waveform_samples // self.waveform_length  # ~88 samples per point
        self.X = np.arange(0, self.waveform_length, 1)
        self.Y_buffer = deque(maxlen=self.waveform_length)
        self.audio_accumulator = deque(maxlen=self.waveform_samples)  # Raw audio buffer for 2 seconds
        # Initialize with zeros
        for _ in range(self.waveform_length):
            self.Y_buffer.append(0.0)
        for _ in range(self.waveform_samples):
            self.audio_accumulator.append(0.0)
        # Example Figure Plot
        self.fig2 = Figure(figsize=(5, 2.5), dpi=96,facecolor='black')
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_facecolor('black')
        self.ax2.get_xaxis().set_visible(False)
        self.ax2.get_yaxis().set_visible(False)
        self.ax2.grid(True, which='both', ls='-', color='#333333')
        self.ax2.set_ylim(-0.2, 0.2)
        self.ax2.set_xlim(0, len(self.X) - 1)
        self.line2, = self.ax2.plot(self.X, list(self.Y_buffer), color='cyan', linewidth=1)
        style.use('ggplot')
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.graphfrm2)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.graphfrm2.pack(side=tk.BOTTOM)
        self.downsample_counter = 0
        self.downsample_accumulator = 0.0

    def create_prediction_page(self):
        """Create the prediction page with cough TB classification"""
        self.prediction_frame = tk.Frame(self.main_frame)

        # Status Bar
        self.prediction_status_widgets = self.create_status_bar(self.prediction_frame)

        # Separator
        separator = ttk.Separator(self.prediction_frame, orient='horizontal')
        separator.pack(fill=tk.X, pady=2)

        # Title
        title_label = tk.Label(self.prediction_frame, text="🔬 Cough TB Prediction", font=self.title_font)
        title_label.pack(pady=5)
        
        # Info Frame
        self.pred_infofrm = tk.Frame(self.prediction_frame)

        # Recording Status
        self.pred_sttrecord = tk.Label(self.pred_infofrm, textvariable=self.txtrecord)
        self.pred_sttrecord.config(font=self.wndfont)
        self.pred_sttrecord.pack(side=tk.TOP)

        # Pack Info Frame
        self.pred_infofrm.pack(side=tk.TOP)

        # Prediction Results Frame
        prediction_frame = tk.Frame(self.prediction_frame)
        prediction_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Results Title
        # results_title = tk.Label(prediction_frame, text="📊 Prediction Results", font=self.subtitle_font)
        # results_title.pack(pady=(0, 15))
        
        # Two column frame for predictions
        columns_frame = tk.Frame(prediction_frame)
        columns_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left Column - Non TB
        left_frame = tk.Frame(columns_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Non TB Label
        non_tb_label = tk.Label(left_frame, text="✅ Non TB", font=self.subtitle_font, fg="green")
        non_tb_label.pack(pady=5)
        
        # Non TB Percentage
        self.non_tb_percentage = tk.StringVar()
        self.non_tb_percentage.set("0%")
        non_tb_percent_label = tk.Label(left_frame, textvariable=self.non_tb_percentage, 
                                    font=font.Font(self.window, family="Liberation Mono", size=18, weight="bold"))
        non_tb_percent_label.pack(pady=10)
        
        # Non TB Progress Bar
        self.non_tb_progress = ttk.Progressbar(left_frame, length=150, mode='determinate')
        self.non_tb_progress.pack(pady=5)
        
        # Right Column - TB
        right_frame = tk.Frame(columns_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # TB Label
        tb_label = tk.Label(right_frame, text="⚠️ TB", font=self.title_font, fg="red")
        tb_label.pack(pady=5)
        
        # TB Percentage
        self.tb_percentage = tk.StringVar()
        self.tb_percentage.set("0%")
        tb_percent_label = tk.Label(right_frame, textvariable=self.tb_percentage,
                                font=font.Font(self.window, family="Liberation Mono", size=18, weight="bold"))
        tb_percent_label.pack(pady=10)
        
        # TB Progress Bar
        self.tb_progress = ttk.Progressbar(right_frame, length=150, mode='determinate')
        self.tb_progress.pack(pady=5)
        
        # Bottom Status Label
        self.prediction_status = tk.StringVar()
        self.prediction_status.set("🎤 Waiting for cough sample...")
        status_label = tk.Label(prediction_frame, textvariable=self.prediction_status, font=self.wndfont)
        status_label.pack(pady=15)
        
        # Manual Prediction Button
        predict_btn = tk.Button(prediction_frame, text="🔍 Analyze Last Cough", 
                            command=self.manual_prediction, font=self.wndfont)
        predict_btn.pack(pady=10)

    def manual_prediction(self):
        """Manually trigger prediction on the last recorded cough"""
        self.prediction_status.set("🔄 Analyzing cough sample...")
        time.sleep(2)
        # Simulate prediction (replace with actual ML model)
        import random
        tb_prob = random.uniform(0, 100)
        non_tb_prob = 100 - tb_prob
        
        # Update percentages
        self.tb_percentage.set(f"{tb_prob:.1f}%")
        self.non_tb_percentage.set(f"{non_tb_prob:.1f}%")
        
        # Update progress bars
        self.tb_progress['value'] = tb_prob
        self.non_tb_progress['value'] = non_tb_prob
        
        # Update status
        if tb_prob > 50:
            self.prediction_status.set("⚠️ High TB probability detected!")
        else:
            self.prediction_status.set("✅ Low TB probability - likely healthy cough")

    def show_page(self, page_num):
        """Switch between pages"""
        if page_num == 1:
            self.analyzer_frame.pack_forget()
            self.prediction_frame.pack_forget()
            self.info_frame.pack_forget()
            self.home_frame.pack(fill=tk.BOTH, expand=True)
            self.current_page = 1
        elif page_num == 2:
            self.home_frame.pack_forget()
            self.prediction_frame.pack_forget()
            self.info_frame.pack_forget()
            self.analyzer_frame.pack(fill=tk.BOTH, expand=True)
            self.current_page = 2
            # Start animation only when on analyzer page
            if not hasattr(self, 'ani'):
                self.ani = animation.FuncAnimation(self.fig2, self.graphupdate, interval=200, blit=False)
        elif page_num == 3:
            self.home_frame.pack_forget()
            self.analyzer_frame.pack_forget()
            self.info_frame.pack_forget()
            self.prediction_frame.pack(fill=tk.BOTH, expand=True)
            self.current_page = 3
            self.txtrecord.set("Ready to Record")
        elif page_num == 4:
            self.home_frame.pack_forget()
            self.analyzer_frame.pack_forget()
            self.prediction_frame.pack_forget()
            self.info_frame.pack(fill=tk.BOTH, expand=True)
            self.current_page = 4

    def configure_dark_theme(self):
        """Configure dark theme for all elements"""
        # Main window
        self.window.config(bg="black")
        self.main_frame.config(bg="black")

        # Configure status bars
        if hasattr(self, 'home_status_widgets'):
            for widget in self.home_status_widgets:
                widget.config(bg='black', fg='white')

        if hasattr(self, 'info_status_widgets'):
            for widget in self.info_status_widgets:
                widget.config(bg='black', fg='white')
                
        if hasattr(self, 'analyzer_status_widgets'):
            for widget in self.analyzer_status_widgets:
                widget.config(bg='black', fg='white')
                
        if hasattr(self, 'prediction_status_widgets'):
            for widget in self.prediction_status_widgets:
                widget.config(bg='black', fg='white')

        # Home page elements
        self.home_frame.config(bg="black")
        for widget in self.home_frame.winfo_children():
            self.configure_widget_dark_theme(widget)
        
        # Home page elements
        self.analyzer_frame.config(bg="black")
        for widget in self.analyzer_frame.winfo_children():
            self.configure_widget_dark_theme(widget)

        self.info_frame.config(bg="black")
        for widget in self.info_frame.winfo_children():
            self.configure_widget_dark_theme(widget)
            
        # Home page elements
        self.prediction_frame.config(bg="black")
        for widget in self.prediction_frame.winfo_children():
            self.configure_widget_dark_theme(widget)

    def configure_widget_dark_theme(self, widget):
        """Recursively configure dark theme for widgets"""
        try:
            widget_class = widget.winfo_class()
            if widget_class in ['Frame', 'Label', 'Button']:
                widget.config(bg='black')
                if widget_class in ['Label', 'Button']:
                    widget.config(fg='white')
            
            # Recursively apply to children
            for child in widget.winfo_children():
                self.configure_widget_dark_theme(child)
        except Exception as e:
            # Some widgets might not support these config options
            pass

    def start_background_processes(self):
        """Start all background processes"""
        Thread(target=self.getinternetstatsprocess, daemon=True).start()
        Thread(target=self.getipprocess, daemon=True).start()
        Thread(target=self.getCoughCount, daemon=True).start()
        Thread(target=self.getCurrentPatient, daemon=True).start()
        Thread(target=self.button_navigation_loop, daemon=True).start()
        Thread(target=self.record_audio_loop, daemon=True).start()
            

    def getwlanip(self):
        # wlp3s0 wlan0
        ipv4 = os.popen(
            'ip addr show '+ self.DEVICE_WLAN +' | grep "\<inet\>" | awk \'{ print $2 }\' | awk -F "/" \'{ print $1 }\'').read().strip()
        return ipv4

    def getipprocess(self):
        """Get IP Loop"""
        while True:
            ipstring = self.getwlanip()
            if ipstring == "":
                "0.0.0.0"
            self.EdgeIP.set("📡" + ipstring)
            time.sleep(10)

    def getCurrentPatient(self):
        """Get IP Loop"""
        while True:
            try:
                with open(self.WEBPANEL_ROOT + '/data/current_patient.json') as pf:
                    patient = json.load(pf)
                    patient_nik = patient.get('nik') or patient.get('NIK') or patient.get('id') or "unknown"
                    if not patient_nik:
                        patient_nik = "unknown"
                    self.current_patient.set(f"👤 {patient.get('nik')} ({patient.get('name')})")
            except Exception as e:
                logging.error(f"[ERROR]: {e}")
                self.current_patient.set(f"❌ {str(e)}")
            time.sleep(5)


    def button_navigation_loop(self):
        """Handle button navigation between pages"""
        btn1_last_state = '1'
        btn2_last_state = '1'
        btn3_last_state = '1'
        btn4_last_state = '1'
        while True:
            try:
                with open(self.BTN1_FILE, "r") as f:
                    btn1_current = f.read().strip()
                with open(self.BTN2_FILE, "r") as f:
                    btn2_current = f.read().strip()
                with open(self.BTN3_FILE, "r") as f:
                    btn3_current = f.read().strip()
                with open(self.BTN4_FILE, "r") as f:
                    btn4_current = f.read().strip()
                
                # Button 1 Navigation Logic
                if btn1_current == '0' and btn1_last_state == '1':
                    if self.current_page == 1:
                        self.show_page(2)
                    with open(self.BTN1_FILE, "w") as f:
                        f.write('1')
                        
                # Button 2 Navigation Logic
                if btn2_current == '0' and btn2_last_state == '1':
                    if self.current_page == 1:
                        self.show_page(3)
                    with open(self.BTN2_FILE, "w") as f:
                        f.write('1')
                
                # Button 3 Navigation Logic
                if btn3_current == '0' and btn3_last_state == '1':
                    if self.current_page == 1:
                        self.show_page(4)
                    with open(self.BTN3_FILE, "w") as f:
                        f.write('1')
                
                # Button 4 Navigation Logic
                if btn4_current == '0' and btn4_last_state == '1':
                    if self.current_page in [2, 3, 4]:
                        self.show_page(1)
                    with open(self.BTN4_FILE, "w") as f:
                        f.write('1')

                btn1_last_state = btn1_current
                btn2_last_state = btn2_current
                btn3_last_state = btn3_current
                btn4_last_state = btn4_current
            except Exception as e:
                logging.error(f"[ERROR] Button navigation error: {e}")
            time.sleep(0.1)

    def getinternetstatsprocess(self, host="8.8.8.8", port=53, timeout=3):
        while True:
            try:
                socket.setdefaulttimeout(timeout)
                socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
                self.internet_status.set("🌍On|")
            except socket.error:
                self.internet_status.set("📵Off|")

            time.sleep(5)

    def sendcoughdataprocess(self):
        while True:
            if self.internet_status.get() == "Online" and GLOBAL_CONFIG.SEND_COUGH == True:
                #self.send_one_data_server("last_send_automatic", "cough", "automatic")
                self.send_one_data_server("last_send_soliced", "solic", "soliced")

            time.sleep(5)

    # def sendstatusdeviceAPIprocess(self):
    #     while True:
    #         autocoughcount = len(next(os.walk("Recorded_Data/automatic"))[2])
    #         solicoughcount = len(next(os.walk("Recorded_Data/soliced"))[2])

    #         response = requests.post(f"{self.SERVER_DOMAIN}/api/device_status", 
    #                         data={'device_id': self.DEVICE_ID, 'autocoughcount': autocoughcount, 'solicoughcount': solicoughcount})
    #         time.sleep(3)

    def start_recording_time_update(self):
        self.recording_time_stop_event.clear()
        self.recording_time_thread = threading.Thread(target=self.update_recording_time_loop, daemon=True)
        self.recording_time_thread.start()

    def stop_recording_time_update(self):
        self.recording_time_stop_event.set()

    def update_recording_time_loop(self):
        while not self.recording_time_stop_event.is_set() and self.recording_start_time:
            elapsed_time = time.time() - self.recording_start_time
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = int(elapsed_time % 60)
            self.txtrecord.set(f"Recording: {hours:02d}:{minutes:02d}:{seconds:02d}")
            time.sleep(1)

    def getCoughCount(self):
        while True:
            autocoughcount = len(next(os.walk("Recorded_Data/automatic"))[2])
            solicoughcount = 0
            try:
                with open(self.WEBPANEL_ROOT + '/data/current_patient.json') as pf:
                    patient = json.load(pf)
                    patient_nik = patient.get('nik') or patient.get('NIK') or patient.get('id') or "unknown"
                    if not patient_nik:
                        patient_nik = "unknown"
                
                patient_dir = os.path.join("Recorded_Data", "soliced", str(patient_nik))
                if os.path.exists(patient_dir):
                    solicoughcount = len(next(os.walk(patient_dir))[2])
            except Exception as e:
                logging.warning(f"[WARNING] Could not read current_patient.json or count files: {e}")
                try:
                    solicoughcount = len(next(os.walk("Recorded_Data/soliced"))[2])
                except:
                    solicoughcount = 0

            self.solicoughcount.set(f"Longi: {autocoughcount} |-| Solic: {solicoughcount}")
            time.sleep(3)

    def graphupdate(self, _):
        self.line2.set_ydata(list(self.Y_buffer))
        return (self.line2,)
    
    def do_updatefigure(self, ignore_cooldown=False):
        if ignore_cooldown:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.last_updateFigure = time.time()
        else:
            if time.time() - self.last_updateFigure > 3.0:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                self.last_updateFigure = time.time()

    def record_audio_loop(self):
        while True:
            if self.current_page == 2 or self.current_page == 3:
                length, data = self.pcm.read()
                if length > 0:
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    audio_data = audio_data.reshape(-1, self.CHANNELS)
                    mono = np.mean(audio_data, axis=1).astype(np.float32) / 32768.0
                    
                    if np.all(mono == 0):
                        self.patch_plot.set_facecolor('red')
                        self.do_updatefigure(ignore_cooldown=True)
                        time.sleep(0.01)
                        continue
                        
                    if self.RECORD_FLAG:
                        with self.buffer_lock:
                            self.audio_buffer.extend(mono)
                            if self.RECORD_LENGTH > 1000:
                                should_stop = len(self.audio_buffer) >= self.RECORD_LENGTH
                            else:
                                with open(self.BTN2_FILE, "r") as stt:
                                    RecSttop = stt.read().strip()
                                should_stop = RecSttop == '0'
                                if should_stop:
                                    with open(self.BTN2_FILE, "w") as out:
                                        out.write('1')
                                        
                            if self.recording_start_time and not self.recording_time_thread:
                                self.start_recording_time_update()
                                
                            if should_stop:
                                data_np = np.array(self.audio_buffer, dtype=np.float32)
                                self.RECORD_FLAG = False
                                self.recording_start_time = None

                                logging.info(f"[DEBUG] Recording stopped. Buffer length: {len(self.audio_buffer)}, RECORD_LENGTH: {self.RECORD_LENGTH}")

                                if self.current_page == 2:
                                    self.txtrecord.set("Recording: Automatic")
                                elif self.current_page == 3:
                                    self.txtrecord.set("Ready to Record")
                                self.patch_plot.set_facecolor('blue')
                                self.do_updatefigure(ignore_cooldown=True)

                                t = Thread(target=self.handle_record_soli, args=(data_np.copy(),))
                                t.start()
                                self.next_time = time.time()
                                self.audio_buffer.clear()
                    else:
                        for sample in mono:
                            self.audio_accumulator.append(sample)
                            self.downsample_accumulator += sample
                            self.downsample_counter += 1
                            if self.downsample_counter >= self.downsample_ratio:
                                avg_sample = self.downsample_accumulator / self.downsample_counter
                                self.Y_buffer.append(avg_sample)
                                self.downsample_counter = 0
                                self.downsample_accumulator = 0.0

                        with open(self.BTN1_FILE, "r") as stt:
                            RecStt = stt.read().strip()

                        if RecStt == '0':
                            logging.info("Button press detected, starting manual recording")
                            with open(self.BTN1_FILE, "w") as out:
                                out.write('1')
                            self.audio_buffer.clear()
                            self.RECORD_FLAG = True
                            self.recording_start_time = time.time()
                            self.start_recording_time_update()

                            logging.info(f"[DEBUG] Recording started. Buffer cleared. RECORD_FLAG: {self.RECORD_FLAG}")

                            self.txtrecord.set("Recording: 00:00:00")
                            self.patch_plot.set_facecolor('yellow')
                            self.do_updatefigure(ignore_cooldown=True)
                        else:
                            if self.current_page == 2:
                                with self.buffer_lock:
                                    self.audio_buffer.extend(mono)
                                    while len(self.audio_buffer) > self.window_size:
                                        self.audio_buffer.popleft()

                                current_gaptime = time.time() - self.next_time
                                if len(self.audio_buffer) >= self.window_size and current_gaptime >= self.STEP_DURATION:
                                    with self.buffer_lock:
                                        data_np = np.array(self.audio_buffer, dtype=np.float32)
            
                                    self.next_time += self.STEP_DURATION #= time.time()

                                    self.patch_plot.set_facecolor('blue')
                                    self.do_updatefigure()

                                    t = Thread(target=self.handle_record_auto,
                                            args=(data_np.copy(),))
                                    t.start()
                                

                    time.sleep(0.01)
                else:
                    time.sleep(0.01)

    def handle_record_auto(self, audio_np):
        audio_np = audio_np[self.AUDIO_POINT_START:]

        with open('autocough_config.json') as config_file:
            cough_config = json.load(config_file)
        
        # Extract parameters with defaults
        cough_padding = cough_config.get('cough_padding', 0.2)
        min_cough_len = cough_config.get('min_cough_len', 0.2)
        th_l_multiplier = cough_config.get('th_l_multiplier', 0.02)
        th_h_multiplier = cough_config.get('th_h_multiplier', 1)
        adaptive_method = cough_config.get('adaptive_method', 'default')
    
        coughSegments, cough_mask = segment_cough(audio_np, self.SAMPLE_RATE, 
                                                cough_padding=cough_padding, 
                                                min_cough_len=min_cough_len,
                                                th_l_multiplier=th_l_multiplier, 
                                                th_h_multiplier=th_h_multiplier, 
                                                adaptive_method=adaptive_method)

        if len(coughSegments) > 0:
            logging.info(f"[INFO] Detected Cough: {len(coughSegments)}")

        for idx, now_cough in enumerate(coughSegments):
            logging.info(f"[INFO] Similarity Last Cough: {self.method_similarity_ratio(now_cough, self.last_cough_np)}")
            #if self.method_similarity_ratio(now_cough, self.last_cough_np) < 0.8:
            self.patch_plot.set_facecolor('green')
            self.do_updatefigure(ignore_cooldown=True)

            onlyfiles = next(os.walk("Recorded_Data/automatic"))[2]
            cough_count = len(onlyfiles) + 1
            now_cough = now_cough.astype(np.float32)
            timestamp = datetime.now().strftime("%d-%m-%Y_%H%M")
            sf.write(f'Recorded_Data/automatic/{timestamp}_{cough_count}.wav', now_cough, self.SAMPLE_RATE, 'PCM_24')
                #self.last_cough_np = now_cough

    def handle_record_soli(self, audio_np):
        if self.current_page == 2:
            try:
                logging.info(f"[INFO] Processing solicited recording with shape: {audio_np.shape}")
                if len(audio_np) == 0:
                    logging.warning("[WARNING] Empty audio data for solicited recording")
                    return

                patient_nik = "unknown"
                try:
                    with open('/home/alarm/web_panel/data/current_patient.json') as pf:
                        patient = json.load(pf)
                        patient_nik = patient.get('nik') or patient.get('NIK') or patient.get('id') or "unknown"
                        if not patient_nik:
                            patient_nik = "unknown"
                except Exception as e:
                    logging.warning(f"[WARNING] Could not read current_patient.json: {e}")

                # ensure patient-specific folder exists
                patient_dir = os.path.join("Recorded_Data", "soliced", str(patient_nik))
                os.makedirs(patient_dir, exist_ok=True)

                onlyfiles = []
                try:
                    onlyfiles = next(os.walk(patient_dir))[2]
                except StopIteration:
                    onlyfiles = []
                cough_count = len(onlyfiles) + 1
                
                audio_np = audio_np[self.AUDIO_POINT_START:]
                timestamp = datetime.now().strftime("%d-%m-%Y_%H%M")
                filename = f'{timestamp}_{cough_count}.wav'
                filepath = os.path.join(patient_dir, filename)
                sf.write(filepath, audio_np, self.SAMPLE_RATE, 'PCM_24')
                
                logging.info(f"[INFO] Saved solicited recording: {filename}")
                rel_path = os.path.join(str(patient_nik), filename)
                self.append_to_lastsend_json("last_send_soliced", rel_path)
            except Exception as e:
                logging.error(f"[ERROR] Failed to save solicited recording: {e}")
        elif self.current_page == 3:
            self.txtrecord.set("Waiting For Prediction to complete....")
            self.manual_prediction()
            self.txtrecord.set("Ready to Record")


    def method_similarity_ratio(self, a, b):
        if a.shape != b.shape:
            return 0.0
        return np.mean(a == b)

    def read_lastsend_json(self):
        with open('Recorded_Data/last_send.json') as data_file:    
            last_send_json = json.load(data_file)

        return last_send_json

    def modify_lastsend_json(self, last_send_json):
        with open('Recorded_Data/last_send.json', 'w') as outfile:
            json.dump(last_send_json, outfile)

    def append_to_lastsend_json(self, key, filename):
        last_send_json = self.read_lastsend_json()
        if key not in last_send_json or not isinstance(last_send_json[key], list):
            last_send_json[key] = []
        if filename not in last_send_json[key]:
            last_send_json[key].append(filename)
        self.modify_lastsend_json(last_send_json)

    def remove_from_lastsend_json(self, key, filename):
        last_send_json = self.read_lastsend_json()
        if key in last_send_json and filename in last_send_json[key]:
            last_send_json[key].remove(filename)
        self.modify_lastsend_json(last_send_json)

    def send_one_data_server(self, current_key_send, file_prefix, folder_send):
        last_send_json = self.read_lastsend_json()
        files_to_send = last_send_json.get(current_key_send, [])
        files_sorted = sorted(files_to_send, key=lambda x: int(re.search(r'_(\d+)\.wav$', x).group(1)) if re.search(r'_(\d+)\.wav$', x) else 0)

        for nowfile_base in files_sorted:
            nowfile = f'Recorded_Data/{folder_send}/{nowfile_base}'
            if os.path.exists(nowfile):
                logging.warning(f"[SENDING]: {nowfile}")

                patient_nik = "unknown"
                if '/' in nowfile_base:
                    patient_nik = nowfile_base.split('/')[0]
                elif folder_send == "soliced":
                    path_parts = nowfile.split('/')
                    if len(path_parts) >= 3:
                        patient_nik = path_parts[-2]

                try:
                    with open(nowfile, 'rb') as f:
                        response = requests.post(
                            f"{self.SERVER_DOMAIN}/api/device/sendData_TBPrimer/{self.DEVICE_ID}", 
                            files={'file_batuk': f}, 
                            data={'nama': 'pasien', 'gender': 'unknown', 'umur': 0, 'cough_type': file_prefix, 'nik': patient_nik},
                            timeout=30
                        )
                    logging.warning(f"[SENDING_STATUS]: {response.status_code}")
                    
                    if response.status_code == 200:
                        try:
                            response_json = response.json()
                            if response_json.get('status') == 'success':
                                self.remove_from_lastsend_json(current_key_send, nowfile_base)
                                logging.warning(f"[SUCCESS] File sent successfully: {nowfile_base}")
                            else:
                                logging.warning(f"[WARNING] Server returned error: {response_json}")
                        except json.JSONDecodeError:
                            logging.warning(f"[WARNING] Invalid JSON response: {response.text}")
                    else:
                        logging.warning(f"[WARNING] Error Send File To Server: {response.status_code} - {response.text}")
                        
                except requests.exceptions.Timeout:
                    logging.warning(f"[ERROR] Request timeout for file: {nowfile}")
                except requests.exceptions.ConnectionError:
                    logging.warning(f"[ERROR] Connection error for file: {nowfile}")
                except requests.exceptions.RequestException as e:
                    logging.warning(f"[ERROR] Request failed for file {nowfile}: {e}")
                except Exception as e:
                    logging.warning(f"[ERROR] Unexpected error sending file {nowfile}: {e}")

                time.sleep(2)

if __name__ == "__main__":
    cough = CoughTk()
