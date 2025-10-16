#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time, os, logging, json, glob, re, socket, requests
from datetime import datetime
from threading import Thread, Lock
from collections import deque
from types import SimpleNamespace

import tkinter as tk
from tkinter import font

import numpy as np
import alsaaudio as alsa
import soundfile as sf
import onnxruntime as ort

# Import matplotlib libraries
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib import style

from utils import process_audio_with_original

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
with open('config.json') as data_file:    
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
    REC_IND_FILE =  "/sys/class/gpio/gpio5/value" # "gpio12" "/sys/class/gpio/gpio12/value"
    RECORD_LENGTH = int(GLOBAL_CONFIG.RECORD_LENGTH * SAMPLE_RATE)

    SERVER_DOMAIN = GLOBAL_CONFIG.SERVER_DOMAIN
    DEVICE_ID = GLOBAL_CONFIG.DEVICE_ID
    ONNX_SESSION = ort.InferenceSession(GLOBAL_CONFIG.ONNX_PATH, providers=["CPUExecutionProvider"])

    def __init__(self):
        super(CoughTk, self).__init__()

        # Main Window
        self.window = tk.Tk()
        self.window.geometry("320x480")
        self.window.title("Tk CoughAnalyzer")

        # Title Label
        self.lbltitle = tk.Label(self.window, text="Cough Analyzer Prototype")
        self.lbltitle.pack(side=tk.TOP)

        # Window Font
        wndfont = font.Font(self.window, family="Liberation Mono", size=15)
        self.lbltitle.config(font=wndfont)

        # Info Frame
        self.infofrm = tk.Frame(self.window)

        # Intertet Connection
        self.internet_status = tk.StringVar()
        self.internet_status.set("Offline")
        self.lb_internet = tk.Label(self.infofrm, textvariable=self.internet_status)
        self.lb_internet.config(font=wndfont)
        self.lb_internet.pack(side=tk.BOTTOM)
        thd_ip = Thread(target=self.getinternetstatsprocess).start()

        # Status Connection
        self.EdgeIP = tk.StringVar()
        self.EdgeIP.set(self.getwlanip())
        self.sttconn = tk.Label(self.infofrm, textvariable=self.EdgeIP)
        self.sttconn.config(font=wndfont)
        self.sttconn.pack(side=tk.BOTTOM)
        thd_ip = Thread(target=self.getipprocess).start()

        # Solic Coughs Status
        self.solicoughcount = tk.StringVar()
        self.solicoughcount.set("Solic Coughs: 0")
        self.lb_coughso = tk.Label(self.infofrm, textvariable=self.solicoughcount)
        self.lb_coughso.config(font=wndfont)
        self.lb_coughso.pack(side=tk.BOTTOM)

        # Auto Coughs Status
        self.autocoughcount = tk.StringVar()
        self.autocoughcount.set("Auto Coughs: 0")
        self.lb_cougha = tk.Label(self.infofrm, textvariable=self.autocoughcount)
        self.lb_cougha.config(font=wndfont)
        self.lb_cougha.pack(side=tk.BOTTOM)

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
        self.fig = Figure(figsize=(5, 1), dpi=100, facecolor='black')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('black')
        #self.ax.grid(True, which='both', ls='-', color='#333333')
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(-1, 1)
        self.patch_plot = self.ax.add_patch(patches.Rectangle(
            (-1.0, 0.0), 2.0, 1.0, linewidth=1, edgecolor='black', facecolor='blue'))

        style.use('ggplot')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graphfrm)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.LEFT)
        self.last_updateFigure = time.time()
        self.graphfrm.pack(side=tk.BOTTOM, expand=True)

        # Graph Data
        self.graphfrm2 = tk.Frame()
        self.X = np.arange(0, self.PERIOD_SIZE, 1)
        self.Y = np.zeros(self.PERIOD_SIZE, dtype=np.float32)
        # Example Figure Plot
        self.fig2 = Figure(figsize=(5, 2), dpi=100,facecolor='black')
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_facecolor('black')
        self.ax2.grid(True, which='both', ls='-', color='#333333')
        self.ax2.set_ylim(-0.5, 0.5)
        self.ax2.set_xlim(0, len(self.X) - 1)
        self.line2, = self.ax2.plot(self.X, self.Y, color='cyan', linewidth=1)
        style.use('ggplot')
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.graphfrm2)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.graphfrm2.pack(side=tk.BOTTOM, expand=True)

        # Dark Theme Config
        if self.DarkTheme:
            self.window.config(bg="black")
            self.lbltitle.config(bg='black', fg='white')
            self.sttrecord.config(bg='black', fg='white')
            self.lb_cougha.config(bg='black', fg='white')
            self.lb_coughso.config(bg='black', fg='white')
            self.sttconn.config(bg='black', fg='white')
            self.lb_internet.config(bg='black', fg='white')
            self.infofrm.config(bg='black')

        # === Shared buffer ===
        self.audio_buffer = deque()
        self.buffer_lock = Lock()
        self.next_time = time.time()
        self.RECORD_FLAG = False
        self.last_cough_np = np.zeros((1000,))

        self.buffer_size = int(self.STEP_DURATION * self.SAMPLE_RATE)
        self.window_size = int(self.WINDOW_DURATION * self.SAMPLE_RATE)

        # # start graph animation
        self.ani = animation.FuncAnimation(self.fig2, self.graphupdate, interval=67, blit=False)

        self.pcm = alsa.PCM(alsa.PCM_CAPTURE, alsa.PCM_NORMAL,
                       channels=self.CHANNELS, rate=self.SAMPLE_RATE, format=alsa.PCM_FORMAT_S16_LE,
                       periodsize=self.PERIOD_SIZE, device=self.DEVICE_SOUND)
        
        Thread(target=self.record_audio_loop).start()
        Thread(target=self.sendcoughdataprocess).start()
        Thread(target=self.sendstatusdeviceAPIprocess).start()

        # Main Loop
        self.window.mainloop()

    def getwlanip(self):
        # wlp3s0 wlan0
        ipv4 = os.popen(
            'ip addr show wlan0 | grep "\<inet\>" | awk \'{ print $2 }\' | awk -F "/" \'{ print $1 }\'').read().strip()
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
            if self.internet_status.get() == "Online" and GLOBAL_CONFIG.SERVER_DOMAIN == True:
                self.send_one_data_server("last_send_automatic", "cough", "automatic")
                self.send_one_data_server("last_send_soliced", "solic", "soliced")

            time.sleep(5)

    def sendstatusdeviceAPIprocess(self):
        while True:
            autocoughcount = len(next(os.walk("Recorded_Data/automatic"))[2])
            solicoughcount = len(next(os.walk("Recorded_Data/soliced"))[2])

            response = requests.post(f"{self.SERVER_DOMAIN}/api/device_status", 
                            data={'device_id': self.DEVICE_ID, 'autocoughcount': autocoughcount, 'solicoughcount': solicoughcount})
            
            time.sleep(3)


    def getCoughCount(self):
        while True:
            autocoughcount = len(next(os.walk("Recorded_Data/automatic"))[2])
            solicoughcount = len(next(os.walk("Recorded_Data/soliced"))[2])

            self.autocoughcount.set(f"Auto Coughs: {autocoughcount}")
            self.solicoughcount.set(f"Solic Coughs: {solicoughcount}")
            time.sleep(3)

    def graphupdate(self, _):
        #if self.RunGraph:
        self.line2.set_ydata(self.Y)
        self.canvas2.draw_idle()
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
                self.Y = mono

                if self.RECORD_FLAG:
                    with self.buffer_lock:
                        self.audio_buffer.extend(mono)
                        if len(self.audio_buffer) >= self.RECORD_LENGTH:
                            data_np = np.array(self.audio_buffer, dtype=np.float32)
                            self.RECORD_FLAG = False

                            self.txtrecord.set("Recording: Automatic")
                            self.patch_plot.set_facecolor('blue')
                            self.do_updatefigure(ignore_cooldown=True)

                            t = Thread(target=self.handle_record_soli,
                                       args=(data_np.copy(),))
                            t.start()
                            self.next_time = time.time()
                            self.audio_buffer.clear()
                else:
                    with open(self.REC_IND_FILE, "r") as stt:
                        RecStt = stt.read().strip()

                    if RecStt == '1':
                        with open(self.REC_IND_FILE, "w") as out:
                            out.write('')
                        self.audio_buffer.clear()
                        self.RECORD_FLAG = True

                        self.txtrecord.set("Recording: Active")
                        self.patch_plot.set_facecolor('yellow')
                        self.do_updatefigure(ignore_cooldown=True)
                    else:
                        with self.buffer_lock:
                            self.audio_buffer.extend(mono)
                            while len(self.audio_buffer) > self.window_size:
                                self.audio_buffer.popleft()
                            data_np = np.array(self.audio_buffer, dtype=np.float32)

                        current_gaptime = time.time() - self.next_time
                        if len(self.audio_buffer) >= self.window_size and current_gaptime >= self.STEP_DURATION:
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
        coughSegments, _ = process_audio_with_original(audio_np, self.SAMPLE_RATE, session=self.ONNX_SESSION, min_cough_samples=0.0, padding=0.0)
        # print(f"[INFO] Processing chunk with shape: {audio_np.shape}")

        if len(coughSegments) > 0:
            logging.info(f"[INFO] Detected Cough: {len(coughSegments)}")

        for idx, now_cough in enumerate(coughSegments):
            logging.info(f"[INFO] Similarity Last Cough: {self.method_similarity_ratio(now_cough, self.last_cough_np)}")
            #if self.method_similarity_ratio(now_cough, self.last_cough_np) < 0.8:
            self.patch_plot.set_facecolor('green')
            self.do_updatefigure(ignore_cooldown=True)

            #self.last_cough_np = now_cough
            onlyfiles = next(os.walk("Recorded_Data/automatic"))[2]
            cough_count = len(onlyfiles) + 1
            timestamp = datetime.now().strftime("%d-%m-%Y_%H%M")
            sf.write(f'Recorded_Data/automatic/{timestamp}_{cough_count}.wav', now_cough, self.SAMPLE_RATE, 'PCM_24')

    def handle_record_soli(self, audio_np):
        onlyfiles = next(os.walk("Recorded_Data/soliced"))[2]
        cough_count = len(onlyfiles) + 1
        timestamp = datetime.now().strftime("%d-%m-%Y_%H%M")
        sf.write(
            f'Recorded_Data/soliced/{timestamp}_{cough_count}.wav', audio_np, self.SAMPLE_RATE, 'PCM_24')

    def method_similarity_ratio(self, a, b):
        if a.shape != b.shape:
            return 0.0
        return np.mean(a == b)

    def read_lastsend_json(self):
        with open('Recorded_Data/last_send.json') as data_file:    
            last_send_json = json.load(data_file)

        return last_send_json

    def modify_lastsend_json(self, last_send_json, modify_key, current_file):
        last_send_json[modify_key] = current_file
        with open('Recorded_Data/last_send.json', 'w') as outfile:
            json.dump(last_send_json, outfile)

    def send_one_data_server(self, current_key_send, file_prefix, folder_send):
        files = glob.glob(f'Recorded_Data/{folder_send}/{file_prefix}_*.wav')
        files_sorted = sorted(files, key=lambda x: int(re.search(file_prefix + r'_(\d+)\.wav', x).group(1)))

        sendfile_flag = False
        last_send_json = self.read_lastsend_json()
        if last_send_json[current_key_send] == None:
            sendfile_flag = True
        for nowfile in files_sorted:
            if sendfile_flag:
                full_path = f'Recorded_Data/{folder_send}/{nowfile}'
                with open(nowfile, 'rb') as f:
                    response = requests.post(f"{self.SERVER_DOMAIN}/api/device/sendData_TBPrimer/{self.DEVICE_ID}", 
                                                files={'file_batuk': f}, 
                                                data={'nama': 'pasien', 'gender': 'unknown', 'umur': 0, 'cough_type': file_prefix})
                if response.status_code == 200:
                    if response.json()['status'] == 'success':               
                        last_send_json = self.read_lastsend_json()
                        self.modify_lastsend_json(last_send_json, current_key_send, nowfile)
                else:
                    logging.warning(f"[WARNING] Error Send File To Server: {response.status_code}")

                time.sleep(2)
            else:
                if nowfile == last_send_json[current_key_send]:
                    sendfile_flag = True

if __name__ == "__main__":
    cough = CoughTk()
