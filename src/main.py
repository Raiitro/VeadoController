import os
import sys
from collections import deque
import time
import json
import math
import threading

# --- CONFIGURATION DYNAMIQUE TCL/TK ---
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
    tcl_dir = os.path.join(base_path, 'tcl', 'tcl8.6')
    tk_dir = os.path.join(base_path, 'tcl', 'tk8.6')
else:
    base_path = sys.base_prefix
    tcl_dir = os.path.join(base_path, "tcl", "tcl8.6")
    tk_dir = os.path.join(base_path, "tcl", "tk8.6")

if os.path.exists(tcl_dir):
    os.environ['TCL_LIBRARY'] = tcl_dir
    os.environ['TK_LIBRARY'] = tk_dir

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import PIL.Image, PIL.ImageTk
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui

def resource_path(relative_path):
    """ Path management for files included in the EXE """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- CONSTANTES ---
CONFIG_FILE = "config.json"
FACE_MODEL_PATH = resource_path('models/face_landmarker.task')
HAND_MODEL_PATH = resource_path('models/hand_landmarker.task')

# --- STABILIZATION LOGIC ---
class StableScore:
    def __init__(self, window_size=5):
        self.history = deque(maxlen=window_size)
        self.state = False 

    def update(self, new_value, high_thresh, low_thresh):
        self.history.append(new_value)
        smoothed_value = sum(self.history) / len(self.history)

        if not self.state and smoothed_value > high_thresh:
            self.state = True
        elif self.state and smoothed_value < low_thresh:
            self.state = False
            
        return self.state, smoothed_value

# --- LOGIC AI ---
class EmotionDetector:
    def __init__(self):
        self.running = False
        self.cap = None
        
        base_options_face = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
        options_face = vision.FaceLandmarkerOptions(
            base_options=base_options_face,
            output_face_blendshapes=True,
            num_faces=1
        )
        self.face_detector = vision.FaceLandmarker.create_from_options(options_face)

        try:
            base_options_hand = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
            options_hand = vision.HandLandmarkerOptions(
                base_options=base_options_hand,
                num_hands=2,
                min_hand_detection_confidence=0.5
            )
            self.hand_detector = vision.HandLandmarker.create_from_options(options_hand)
            self.has_hand_model = True
        except:
            self.has_hand_model = False

        self.current_action = "NEUTRAL"
        
        self.stables = {
            "FROWN": StableScore(window_size=6),
            "SMILE": StableScore(window_size=6),
            "RAISE": StableScore(window_size=4),
            "MALICIOUS": StableScore(window_size=5),
            "TILT": StableScore(window_size=6),
            "WINK": StableScore(window_size=4),
            "THINKING": StableScore(window_size=5)
        }
        
        self.unlock_time = 0 

    def detect(self, frame, thresholds, enabled_dict, min_durations):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        face_result = self.face_detector.detect(mp_image)
        
        hand_result = None
        if self.has_hand_model:
            hand_result = self.hand_detector.detect(mp_image)
        
        physical_action = "NEUTRAL"

        if face_result.face_landmarks and face_result.face_blendshapes:
            landmarks = face_result.face_landmarks[0]
            s = {b.category_name: b.score for b in face_result.face_blendshapes[0]}
            
            raw_brow_down = max(s.get('browDownLeft', 0), s.get('browDownRight', 0))
            raw_brow_up = s.get('browInnerUp', 0)
            raw_smile = (s.get('mouthSmileLeft', 0) + s.get('mouthSmileRight', 0)) / 2
            
            left_eye_y = landmarks[33].y
            right_eye_y = landmarks[263].y
            raw_tilt = abs(left_eye_y - right_eye_y) * 5.0

            raw_thinking = 0.0
            if hand_result and hand_result.hand_landmarks:
                chin = landmarks[152]
                for hand in hand_result.hand_landmarks:
                    index_tip = hand[8]
                    thumb_tip = hand[4]
                    dist_index = math.sqrt((chin.x - index_tip.x)**2 + (chin.y - index_tip.y)**2)
                    dist_thumb = math.sqrt((chin.x - thumb_tip.x)**2 + (chin.y - thumb_tip.y)**2)
                    min_dist = min(dist_index, dist_thumb)
                    if min_dist < 0.20:
                        raw_thinking = max(raw_thinking, (0.20 - min_dist) * 5.0)

            # CALCULATION OF STATES
            is_thinking = False
            if enabled_dict.get("THINKING", True) and self.has_hand_model:
                is_thinking, _ = self.stables["THINKING"].update(raw_thinking, thresholds['THINKING'], thresholds['THINKING'] - 0.15)

            is_malicious = False
            if enabled_dict.get("MALICIOUS", True):
                raw_malicious = (raw_brow_down + raw_smile) / 2
                if raw_brow_down < 0.25 or raw_smile < 0.25: raw_malicious = 0
                is_malicious, _ = self.stables["MALICIOUS"].update(raw_malicious, thresholds['MALICIOUS'], thresholds['MALICIOUS'] - 0.15)

            is_winking = False
            if enabled_dict.get("WINK", True) and not is_malicious:
                raw_wink = abs(s.get('eyeBlinkLeft', 0) - s.get('eyeBlinkRight', 0))
                is_winking, _ = self.stables["WINK"].update(raw_wink, thresholds['WINK'], thresholds['WINK'] - 0.15)

            is_tilting = False
            if enabled_dict.get("TILT", True) and not is_malicious and not is_winking:
                is_tilting, _ = self.stables["TILT"].update(raw_tilt, thresholds['TILT'], thresholds['TILT'] - 0.10)

            is_frowning = False
            if enabled_dict.get("FROWN", True) and not is_malicious and not is_winking:
                frown_score = raw_brow_down
                if raw_brow_up > 0.4: frown_score = 0 
                is_frowning, _ = self.stables["FROWN"].update(frown_score, thresholds['FROWN'], thresholds['FROWN'] - 0.15)

            is_raising = False
            if enabled_dict.get("RAISE", True) and not is_tilting:
                is_raising, _ = self.stables["RAISE"].update(raw_brow_up, thresholds['RAISE'], thresholds['RAISE'] - 0.15)

            is_smiling = False
            if enabled_dict.get("SMILE", True) and not is_malicious and not is_winking and not is_tilting:
                is_smiling, _ = self.stables["SMILE"].update(raw_smile, thresholds['SMILE'], thresholds['SMILE'] - 0.15)

            if is_thinking: physical_action = "THINKING"
            elif is_malicious: physical_action = "MALICIOUS"
            elif is_winking: physical_action = "WINK"
            elif is_tilting: physical_action = "TILT"
            elif is_frowning: physical_action = "FROWN"
            elif is_raising: physical_action = "RAISE"
            elif is_smiling: physical_action = "SMILE"

        now = time.time()
        if self.current_action != "NEUTRAL" and now < self.unlock_time:
            return self.current_action
        
        if physical_action != self.current_action:
            if physical_action != "NEUTRAL":
                duration = float(min_durations.get(physical_action, 0.0))
                self.unlock_time = now + duration
            return physical_action
        
        return self.current_action

# --- CONFIG WIZARD WINDOW ---
class SetupWizard(tk.Toplevel):
    def __init__(self, parent, config):
        super().__init__(parent)
        self.title("Key Calibration Assistant")
        self.geometry("500x500")
        self.config = config
        self.parent = parent
        
        tk.Label(self, text="Click on 'Send' to simulate the key.\nYou have 3 seconds to click on your Stream software.", 
                 font=("Arial", 10), bg="#f0f0f0", pady=10).pack(fill="x")

        # dropdown menu
        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = tk.Frame(self.canvas)
        
        self.scroll_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Items
        # 1. NEUTRAL
        self.create_row("NEUTRAL", self.config['keys'].get("NEUTRAL", ""))
        
        # 2. other
        for act in ["SMILE", "FROWN", "RAISE", "MALICIOUS", "TILT", "WINK", "THINKING"]:
            self.create_row(act, self.config['keys'].get(act, ""))

    def create_row(self, label, key):
        frame = tk.Frame(self.scroll_frame, pady=5, padx=5, highlightbackground="#ccc", highlightthickness=1)
        frame.pack(fill="x", pady=2, padx=5)
        
        tk.Label(frame, text=f"{label}", width=15, anchor="w", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        tk.Label(frame, text=f"[{key}]", width=8, fg="blue").pack(side=tk.LEFT)
        
        btn = tk.Button(frame, text="⏱️ Send (3s)", bg="#ddd", 
                        command=lambda k=key: self.send_key_delayed(k, btn))
        btn.pack(side=tk.RIGHT)

    def send_key_delayed(self, key, btn):
        if not key: return
        
        def run():
            orig_text = btn.cget("text")
            btn.config(bg="orange", text="3...")
            time.sleep(1)
            btn.config(text="2...")
            time.sleep(1)
            btn.config(text="1...")
            time.sleep(1)
            
            pyautogui.press(key)
            print(f"Simulation key : {key}")
            
            btn.config(bg="#4CAF50", text="SENT !")
            time.sleep(1)
            btn.config(bg="#ddd", text=orig_text)

        threading.Thread(target=run, daemon=True).start()

# --- MAIN INTERFACE ---
class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1100x750")

        self.detector = EmotionDetector()
        self.video_source = 0
        self.load_config()

        self.left_container = tk.Frame(window, width=400, bg="#f0f0f0")
        self.left_container.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        self.left_container.pack_propagate(False)

        # 1. HEADER
        self.header_frame = tk.Frame(self.left_container, bg="#f0f0f0")
        self.header_frame.pack(side=tk.TOP, fill="x", padx=10, pady=10)
        tk.Label(self.header_frame, text="PNGTuber Controller v1.0", font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=5)
        
        frame_neutral = tk.LabelFrame(self.header_frame, text="Back to Calm (NEUTRAL)", bg="#e6e6e6", pady=5)
        frame_neutral.pack(fill="x", pady=5)
        tk.Label(frame_neutral, text="key:", bg="#e6e6e6").pack(side=tk.LEFT, padx=5)
        self.entry_neutral = tk.Entry(frame_neutral, width=8)
        self.entry_neutral.insert(0, self.config['keys'].get("NEUTRAL", "f13"))
        self.entry_neutral.pack(side=tk.LEFT, padx=5)

        # 2. FOOTER
        self.footer_frame = tk.Frame(self.left_container, bg="#f0f0f0")
        self.footer_frame.pack(side=tk.BOTTOM, fill="x", padx=10, pady=10)
        
        # Wizard button
        self.btn_wizard = tk.Button(self.footer_frame, text="🛠️ Assistant Setup keys", command=self.open_wizard, bg="#FFD700", height=1)
        self.btn_wizard.pack(pady=5, fill="x")

        self.var_preview = tk.BooleanVar(value=True)
        tk.Checkbutton(self.footer_frame, text="Show Camera Preview", var=self.var_preview, bg="#f0f0f0").pack(pady=5)
        self.btn_save = tk.Button(self.footer_frame, text="💾 Save Config", command=self.save_config, bg="#ddd", height=2)
        self.btn_save.pack(pady=5, fill="x")
        self.btn_start = tk.Button(self.footer_frame, text="▶ START", command=self.toggle_camera, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), height=2)
        self.btn_start.pack(pady=5, fill="x")
        self.lbl_status = tk.Label(self.footer_frame, text="Status: Stopped", fg="red", bg="#f0f0f0")
        self.lbl_status.pack(pady=5)

        # 3. SCROLL
        self.canvas_scroll = tk.Canvas(self.left_container, bg="#f0f0f0", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.left_container, orient="vertical", command=self.canvas_scroll.yview)
        self.scrollable_frame = tk.Frame(self.canvas_scroll, bg="#f0f0f0")
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas_scroll.configure(scrollregion=self.canvas_scroll.bbox("all")))
        self.scrollbar.pack(side="right", fill="y")
        self.canvas_scroll.pack(side="left", fill="both", expand=True, padx=5)
        self.canvas_scroll.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=380)
        self.canvas_scroll.configure(yscrollcommand=self.scrollbar.set)
        self.canvas_scroll.bind_all("<MouseWheel>", self._on_mousewheel)

        # 4. LISTE
        self.sliders = {}
        self.entries = {}
        self.entries_duration = {}
        self.vars_enabled = {}
        
        actions = ["SMILE", "FROWN", "RAISE", "MALICIOUS", "TILT", "WINK", "THINKING"]
        
        for act in actions:
            frame = tk.LabelFrame(self.scrollable_frame, bg="#f0f0f0", pady=5)
            frame.pack(fill="x", pady=5, padx=5)
            
            var_enable = tk.BooleanVar(value=self.config['enabled'].get(act, True))
            self.vars_enabled[act] = var_enable
            
            title = act
            if act == "THINKING": title += " (Requires Hands)"
            
            chk = tk.Checkbutton(frame, text=title, variable=var_enable, font=("Arial", 10, "bold"), bg="#f0f0f0", anchor="w")
            chk.pack(fill="x", padx=5)

            grid_frame = tk.Frame(frame, bg="#f0f0f0")
            grid_frame.pack(fill="x", padx=5, pady=2)

            tk.Label(grid_frame, text="key:", bg="#f0f0f0").grid(row=0, column=0, sticky="w")
            entry = tk.Entry(grid_frame, width=6)
            entry.insert(0, self.config['keys'].get(act, ""))
            entry.grid(row=0, column=1, padx=5)
            self.entries[act] = entry

            tk.Label(grid_frame, text="Hold Duration:", bg="#f0f0f0").grid(row=0, column=2, sticky="w")
            entry_dur = tk.Entry(grid_frame, width=5)
            entry_dur.insert(0, str(self.config['min_durations'].get(act, 0.5)))
            entry_dur.grid(row=0, column=3, padx=5)
            self.entries_duration[act] = entry_dur

            tk.Label(frame, text="sensitivity:", bg="#f0f0f0", font=("Arial", 8)).pack(anchor="w", padx=5)
            slider = tk.Scale(frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, bg="#f0f0f0", length=250)
            slider.set(self.config['thresholds'].get(act, 0.5))
            slider.pack(fill="x", padx=5)
            self.sliders[act] = slider

        self.right_frame = tk.Frame(window, bg="black")
        self.right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)
        self.lbl_current_action = tk.Label(self.right_frame, text="waiting...", font=("Arial", 20, "bold"), bg="black", fg="white")
        self.lbl_current_action.pack(side=tk.TOP, fill="x", pady=5)
        self.canvas = tk.Canvas(self.right_frame, bg="#222")
        self.canvas.pack(expand=True, fill="both")
        self.delay = 15
        self.update()

    def _on_mousewheel(self, event):
        self.canvas_scroll.yview_scroll(int(-1*(event.delta/120)), "units")

    def open_wizard(self):
        self.save_config_silent()
        SetupWizard(self.window, self.config)

    def load_config(self):
        default = {
            "keys": {
                "NEUTRAL": "f13", "SMILE": "f14", "FROWN": "f15", 
                "RAISE": "f16", "MALICIOUS": "f17", "TILT": "f18", 
                "WINK": "f19", "THINKING": "f20"
            },
            "thresholds": {
                "SMILE": 0.5, "FROWN": 0.4, "RAISE": 0.5, 
                "MALICIOUS": 0.7, "TILT": 0.3, "WINK": 0.2, "THINKING": 0.6
            },
            "enabled": {
                "SMILE": True, "FROWN": True, "RAISE": True, 
                "MALICIOUS": True, "TILT": True, "WINK": True, "THINKING": True
            },
            "min_durations": {
                "SMILE": 0.5, "FROWN": 0.5, "RAISE": 0.5, 
                "MALICIOUS": 1.0, "TILT": 0.5, "WINK": 0.2, "THINKING": 1.0
            }
        }
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    loaded = json.load(f)
                    self.config = default.copy()
                    if "keys" in loaded and "HAPPY" in loaded["keys"]:
                        self.config = default
                    else:
                        self.config.update(loaded)
                        for key in ["keys", "thresholds", "enabled", "min_durations"]:
                             if key not in self.config: self.config[key] = default[key]
                             for act in default[key]:
                                 if act not in self.config[key]: self.config[key][act] = default[key][act]
            except:
                self.config = default
        else:
            self.config = default

    def save_config_silent(self):
        """ Saving without displaying popup (useful for the wizard) """
        self.config['keys']['NEUTRAL'] = self.entry_neutral.get()
        if 'min_durations' not in self.config: self.config['min_durations'] = {}
        for act in self.sliders:
            self.config['thresholds'][act] = self.sliders[act].get()
            self.config['keys'][act] = self.entries[act].get()
            self.config['enabled'][act] = self.vars_enabled[act].get()
            try: val = float(self.entries_duration[act].get())
            except: val = 0.0
            self.config['min_durations'][act] = val
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=4)

    def save_config(self):
        self.save_config_silent()
        messagebox.showinfo("Info", "Configuration saved !")

    def toggle_camera(self):
        if self.detector.running:
            self.detector.running = False
            if self.detector.cap:
                self.detector.cap.release()
            self.btn_start.config(text="▶ START", bg="#4CAF50")
            self.lbl_status.config(text="Status: Stopped", fg="red")
            self.canvas.delete("all")
        else:
            self.detector.cap = cv2.VideoCapture(self.video_source)
            self.detector.running = True
            self.btn_start.config(text="⏹ STOP", bg="#f44336")
            self.lbl_status.config(text="Status: Running", fg="green")

    def update(self):
        if self.detector.running and self.detector.cap.isOpened():
            ret, frame = self.detector.cap.read()
            if ret:
                thresholds = {act: s.get() for act, s in self.sliders.items()}
                enabled_dict = {act: v.get() for act, v in self.vars_enabled.items()}
                min_durations = {}
                for act, entry in self.entries_duration.items():
                    try: min_durations[act] = float(entry.get())
                    except: min_durations[act] = 0.0

                action = self.detector.detect(frame, thresholds, enabled_dict, min_durations)
                
                if action != self.detector.current_action:
                    key = self.entry_neutral.get() if action == "NEUTRAL" else self.entries.get(action, tk.Entry()).get()
                    if key:
                        pyautogui.press(key)
                        print(f"Action: {action} -> {key}")
                    
                    self.detector.current_action = action
                    
                    color_map = {
                        "NEUTRAL": "#00FF00", "MALICIOUS": "#A020F0", 
                        "FROWN": "#FF0000", "RAISE": "#FFFF00", 
                        "SMILE": "#00FFFF", "TILT": "#FF1493",
                        "WINK": "#FFA500", "THINKING": "#1E90FF"
                    }
                    color = color_map.get(action, "white")
                    self.lbl_current_action.config(text=f"ACTION : {action}", fg=color)

                if self.var_preview.get():
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cw = self.canvas.winfo_width()
                    ch = self.canvas.winfo_height()
                    h, w, _ = frame.shape
                    
                    if cw > 10 and ch > 10:
                        scale = min(cw/w, ch/h)
                        new_w, new_h = int(w*scale), int(h*scale)
                        frame = cv2.resize(frame, (new_w, new_h))
                        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                        self.canvas.create_image((cw - new_w)//2, (ch - new_h)//2, image=self.photo, anchor=tk.NW)
                    
                    if self.detector.has_hand_model and action == "THINKING":
                        self.canvas.create_text(20, 20, text="HAND DETECTED", fill="cyan", anchor="nw")

                else:
                    if len(self.canvas.find_all()) > 1:
                         self.canvas.delete("all")
                         self.canvas.create_text(self.canvas.winfo_width()//2, self.canvas.winfo_height()//2, 
                                               text="MONITORING ACTIVE\n(Preview disabled)", fill="gray", font=("Arial", 14))

        self.window.after(self.delay, self.update)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root, "PNGTuber Controller v1.0")
    root.mainloop()