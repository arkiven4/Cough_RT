#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time, os, logging, json, glob, re, socket, requests
from datetime import datetime
from threading import Thread, Lock
from collections import deque
from types import SimpleNamespace

import tkinter as tk
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
    REC_IND_FILE =  GLOBAL_CONFIG.REC_IND_FILE # "gpio12" "/sys/class/gpio/gpio5/value"
    REC_STOP_FILE =  GLOBAL_CONFIG.REC_STOP_FILE # "gpio12" "/sys/class/gpio/gpio6/value"
    RECORD_LENGTH = int(GLOBAL_CONFIG.RECORD_LENGTH * SAMPLE_RATE)
    AUDIO_POINT_START = round(0.3 * SAMPLE_RATE)

    SERVER_DOMAIN = GLOBAL_CONFIG.SERVER_DOMAIN
    DEVICE_ID = GLOBAL_CONFIG.DEVICE_ID

    def __init__(self):
        super(CoughTk, self).__init__()

        # Main Window
        self.window = tk.Tk()
        self.window.geometry("480x320")
        self.window.title("TBCare - CoughAnalyzer")
        self.current_page = 1

        # Window Font
        wndfont = font.Font(self.window, family="Liberation Mono", size=15)
        #self.lbltitle.config(font=wndfont)

        # Info Frame
        self.infofrm = tk.Frame(self.window)

        # Create a sub-frame for IP and Internet status to be side by side
        self.ip_internet_frame = tk.Frame(self.infofrm)

        # Internet Connection
        self.internet_status = tk.StringVar()
        self.internet_status.set("Offline")
        self.lb_internet = tk.Label(self.ip_internet_frame, textvariable=self.internet_status)
        self.lb_internet.config(font=wndfont)
        self.lb_internet.pack(side=tk.RIGHT, padx=(10, 0))  # Add some padding between them
        thd_ip = Thread(target=self.getinternetstatsprocess).start()

        # Status Connection
        self.EdgeIP = tk.StringVar()
        self.EdgeIP.set(self.getwlanip())
        self.sttconn = tk.Label(self.ip_internet_frame, textvariable=self.EdgeIP)
        self.sttconn.config(font=wndfont)
        self.sttconn.pack(side=tk.LEFT)
        thd_ip = Thread(target=self.getipprocess).start()

        # Pack the IP/Internet frame
        self.ip_internet_frame.pack(side=tk.BOTTOM)

        # Solic Coughs Status
        self.solicoughcount = tk.StringVar()
        self.solicoughcount.set("Longi: 0 || Solic: 0")
        self.lb_coughso = tk.Label(self.infofrm, textvariable=self.solicoughcount)
        self.lb_coughso.config(font=wndfont)
        self.lb_coughso.pack(side=tk.BOTTOM)

        # Auto Coughs Status
        # self.autocoughcount = tk.StringVar()
        # self.autocoughcount.set("Auto Coughs: 0")
        # self.lb_cougha = tk.Label(self.infofrm, textvariable=self.autocoughcount)
        # self.lb_cougha.config(font=wndfont)
        # self.lb_cougha.pack(side=tk.BOTTOM)

        thd_coughCount = Thread(target=self.getCoughCount).start()

        # Recording Status
        self.txtrecord = tk.StringVar()
        self.txtrecord.set("Recording: Automatic")
        self.sttrecord = tk.Label(self.infofrm, textvariable=self.txtrecord)
        self.sttrecord.config(font=wndfont)
        self.sttrecord.pack(side=tk.BOTTOM)

        # Pack Info Frame
        self.infofrm.pack(side=tk.TOP)

        # Graph Frame
        self.graphfrm = tk.Frame()
        # Example Figure Plot
        self.fig = Figure(figsize=(5, 1), dpi=96, facecolor='black')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('black')
        #self.ax.grid(True, which='both', ls='-', color='#333333')
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
        self.graphfrm2 = tk.Frame()
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

        # Dark Theme Config
        if self.DarkTheme:
            self.window.config(bg="black")
            #self.lbltitle.config(bg='black', fg='white')
            self.sttrecord.config(bg='black', fg='white')
            #self.lb_cougha.config(bg='black', fg='white')
            self.lb_coughso.config(bg='black', fg='white')
            self.sttconn.config(bg='black', fg='white')
            self.lb_internet.config(bg='black', fg='white')
            self.infofrm.config(bg='black')
            self.ip_internet_frame.config(bg='black')
            self.graphfrm.config(bg='black')
            self.graphfrm2.config(bg='black')

        # === Shared buffer ===
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

        # # start graph animation
        self.ani = animation.FuncAnimation(self.fig2, self.graphupdate, interval=200, blit=False)

        self.pcm = alsa.PCM(alsa.PCM_CAPTURE, alsa.PCM_NORMAL,
                       channels=self.CHANNELS, rate=self.SAMPLE_RATE, format=alsa.PCM_FORMAT_S16_LE,
                       periodsize=self.PERIOD_SIZE, device=self.DEVICE_SOUND)
        
        Thread(target=self.record_audio_loop).start()
        #Thread(target=self.sendcoughdataprocess).start()
        # Thread(target=self.sendstatusdeviceAPIprocess).start()

        # Main Loop
        self.window.mainloop()

    def getwlanip(self):
        # wlp3s0 wlan0
        ipv4 = os.popen(
            'ip addr show wlp3s0 | grep "\<inet\>" | awk \'{ print $2 }\' | awk -F "/" \'{ print $1 }\'').read().strip()
        return ipv4

    def getipprocess(self):
        """Get IP Loop"""
        while True:
            ipstring = self.getwlanip()
            if ipstring == "":
                "0.0.0.0"
            self.EdgeIP.set(ipstring)
            time.sleep(10)

    def getinternetstatsprocess(self, host="8.8.8.8", port=53, timeout=3):
        while True:
            try:
                socket.setdefaulttimeout(timeout)
                socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
                self.internet_status.set("Online")
            except socket.error:
                self.internet_status.set("Offline")

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
            solicoughcount = len(next(os.walk("Recorded_Data/soliced"))[2])

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
                            with open(self.REC_STOP_FILE, "r") as stt:
                                RecSttop = stt.read().strip()
                            should_stop = RecSttop == '0'
                            if should_stop:
                                with open(self.REC_IND_FILE, "w") as out:
                                    out.write('1')
                                    
                        if self.recording_start_time and not self.recording_time_thread:
                            self.start_recording_time_update()
                            
                        if should_stop:
                            data_np = np.array(self.audio_buffer, dtype=np.float32)
                            self.RECORD_FLAG = False
                            self.recording_start_time = None

                            logging.info(f"[DEBUG] Recording stopped. Buffer length: {len(self.audio_buffer)}, RECORD_LENGTH: {self.RECORD_LENGTH}")

                            self.txtrecord.set("Recording: Automatic")
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

                    with open(self.REC_IND_FILE, "r") as stt:
                        RecStt = stt.read().strip()

                    if RecStt == '0':
                        logging.info("Button press detected, starting manual recording")
                        with open(self.REC_IND_FILE, "w") as out:
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
        th_l_multiplier = cough_config.get('th_l_multiplier', 0.08)
        th_h_multiplier = cough_config.get('th_h_multiplier', 2)
        adaptive_method = cough_config.get('adaptive_method', 'combination')
    
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
        try:
            logging.info(f"[INFO] Processing solicited recording with shape: {audio_np.shape}")
            if len(audio_np) == 0:
                logging.warning("[WARNING] Empty audio data for solicited recording")
                return
                
            onlyfiles = next(os.walk("Recorded_Data/soliced"))[2]
            cough_count = len(onlyfiles) + 1
            
            audio_np = audio_np[self.AUDIO_POINT_START:]
            timestamp = datetime.now().strftime("%d-%m-%Y_%H%M")
            filename = f'Recorded_Data/soliced/{timestamp}_{cough_count}.wav'
            sf.write(filename, audio_np, self.SAMPLE_RATE, 'PCM_24')
            
            logging.info(f"[INFO] Saved solicited recording: {filename}")
            self.append_to_lastsend_json("last_send_soliced", os.path.basename(filename))
        except Exception as e:
            logging.error(f"[ERROR] Failed to save solicited recording: {e}")

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
                try:
                    with open(nowfile, 'rb') as f:
                        response = requests.post(
                            f"{self.SERVER_DOMAIN}/api/device/sendData_TBPrimer/{self.DEVICE_ID}", 
                            files={'file_batuk': f}, 
                            data={'nama': 'pasien', 'gender': 'unknown', 'umur': 0, 'cough_type': file_prefix},
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
