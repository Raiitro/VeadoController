import os
import sys
from collections import deque
import time
import json
import math
import threading

# --- 1. CONFIGURATION DYNAMIQUE TCL/TK ---
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
    os.environ['TCL_LIBRARY'] = os.path.join(base_path, 'tcl', 'tcl8.6')
    os.environ['TK_LIBRARY'] = os.path.join(base_path, 'tcl', 'tk8.6')
else:
    python_dir = sys.exec_prefix
    tcl_path = os.path.join(python_dir, "tcl", "tcl8.6")
    tk_path = os.path.join(python_dir, "tcl", "tk8.6")
    if os.path.exists(tcl_path):
        os.environ['TCL_LIBRARY'] = tcl_path
        os.environ['TK_LIBRARY'] = tk_path

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import PIL.Image, PIL.ImageTk
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui

def resource_path(relative_path):
    """ Gestion des chemins pour les fichiers inclus dans l'EXE """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- CONSTANTES ---
CONFIG_FILE = "config.json"
FACE_MODEL_PATH = resource_path('models/face_landmarker.task')
HAND_MODEL_PATH = resource_path('models/hand_landmarker.task')

# --- LOGIQUE DE STABILISATION ---
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

# --- LOGIQUE IA ---
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

        self.current_action = "NEUTRE"
        
        self.stables = {
            "FRONCEMENT": StableScore(window_size=6),
            "SOURIRE": StableScore(window_size=6),
            "HAUSSEMENT": StableScore(window_size=4),
            "MALICIEUX": StableScore(window_size=5),
            "PENCHE": StableScore(window_size=6),
            "CLIN_DOEIL": StableScore(window_size=4),
            "REFLECHIR": StableScore(window_size=5)
        }
        
        self.unlock_time = 0 

    def detect(self, frame, thresholds, enabled_dict, min_durations):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        face_result = self.face_detector.detect(mp_image)
        
        hand_result = None
        if self.has_hand_model:
            hand_result = self.hand_detector.detect(mp_image)
        
        physical_action = "NEUTRE"

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

            # CALCUL DES ETATS
            is_thinking = False
            if enabled_dict.get("REFLECHIR", True) and self.has_hand_model:
                is_thinking, _ = self.stables["REFLECHIR"].update(raw_thinking, thresholds['REFLECHIR'], thresholds['REFLECHIR'] - 0.15)

            is_malicious = False
            if enabled_dict.get("MALICIEUX", True):
                raw_malicious = (raw_brow_down + raw_smile) / 2
                if raw_brow_down < 0.25 or raw_smile < 0.25: raw_malicious = 0
                is_malicious, _ = self.stables["MALICIEUX"].update(raw_malicious, thresholds['MALICIEUX'], thresholds['MALICIEUX'] - 0.15)

            is_winking = False
            if enabled_dict.get("CLIN_DOEIL", True) and not is_malicious:
                raw_wink = abs(s.get('eyeBlinkLeft', 0) - s.get('eyeBlinkRight', 0))
                is_winking, _ = self.stables["CLIN_DOEIL"].update(raw_wink, thresholds['CLIN_DOEIL'], thresholds['CLIN_DOEIL'] - 0.15)

            is_tilting = False
            if enabled_dict.get("PENCHE", True) and not is_malicious and not is_winking:
                is_tilting, _ = self.stables["PENCHE"].update(raw_tilt, thresholds['PENCHE'], thresholds['PENCHE'] - 0.10)

            is_frowning = False
            if enabled_dict.get("FRONCEMENT", True) and not is_malicious and not is_winking:
                frown_score = raw_brow_down
                if raw_brow_up > 0.4: frown_score = 0 
                is_frowning, _ = self.stables["FRONCEMENT"].update(frown_score, thresholds['FRONCEMENT'], thresholds['FRONCEMENT'] - 0.15)

            is_raising = False
            if enabled_dict.get("HAUSSEMENT", True) and not is_tilting:
                is_raising, _ = self.stables["HAUSSEMENT"].update(raw_brow_up, thresholds['HAUSSEMENT'], thresholds['HAUSSEMENT'] - 0.15)

            is_smiling = False
            if enabled_dict.get("SOURIRE", True) and not is_malicious and not is_winking and not is_tilting:
                is_smiling, _ = self.stables["SOURIRE"].update(raw_smile, thresholds['SOURIRE'], thresholds['SOURIRE'] - 0.15)

            if is_thinking: physical_action = "REFLECHIR"
            elif is_malicious: physical_action = "MALICIEUX"
            elif is_winking: physical_action = "CLIN_DOEIL"
            elif is_tilting: physical_action = "PENCHE"
            elif is_frowning: physical_action = "FRONCEMENT"
            elif is_raising: physical_action = "HAUSSEMENT"
            elif is_smiling: physical_action = "SOURIRE"

        now = time.time()
        if self.current_action != "NEUTRE" and now < self.unlock_time:
            return self.current_action
        
        if physical_action != self.current_action:
            if physical_action != "NEUTRE":
                duration = float(min_durations.get(physical_action, 0.0))
                self.unlock_time = now + duration
            return physical_action
        
        return self.current_action

# --- NOUVEAU : FENÊTRE ASSISTANT DE CONFIG ---
class SetupWizard(tk.Toplevel):
    def __init__(self, parent, config):
        super().__init__(parent)
        self.title("Assistant de Calibrage des Touches")
        self.geometry("500x500")
        self.config = config
        self.parent = parent
        
        tk.Label(self, text="Cliquez sur 'Envoyer' pour simuler la touche.\nVous avez 3 secondes pour cliquer sur votre logiciel de Stream.", 
                 font=("Arial", 10), bg="#f0f0f0", pady=10).pack(fill="x")

        # Liste déroulante
        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = tk.Frame(self.canvas)
        
        self.scroll_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Items
        # 1. Neutre
        self.create_row("NEUTRE", self.config['keys'].get("NEUTRE", ""))
        
        # 2. Autres
        for act in ["SOURIRE", "FRONCEMENT", "HAUSSEMENT", "MALICIEUX", "PENCHE", "CLIN_DOEIL", "REFLECHIR"]:
            self.create_row(act, self.config['keys'].get(act, ""))

    def create_row(self, label, key):
        frame = tk.Frame(self.scroll_frame, pady=5, padx=5, highlightbackground="#ccc", highlightthickness=1)
        frame.pack(fill="x", pady=2, padx=5)
        
        tk.Label(frame, text=f"{label}", width=15, anchor="w", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        tk.Label(frame, text=f"[{key}]", width=8, fg="blue").pack(side=tk.LEFT)
        
        btn = tk.Button(frame, text="⏱️ Envoyer (3s)", bg="#ddd", 
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
            print(f"Simulation touche : {key}")
            
            btn.config(bg="#4CAF50", text="ENVOYÉ !")
            time.sleep(1)
            btn.config(bg="#ddd", text=orig_text)

        threading.Thread(target=run, daemon=True).start()

# --- INTERFACE PRINCIPALE ---
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
        
        frame_neutral = tk.LabelFrame(self.header_frame, text="Retour au calme (NEUTRE)", bg="#e6e6e6", pady=5)
        frame_neutral.pack(fill="x", pady=5)
        tk.Label(frame_neutral, text="Touche:", bg="#e6e6e6").pack(side=tk.LEFT, padx=5)
        self.entry_neutral = tk.Entry(frame_neutral, width=8)
        self.entry_neutral.insert(0, self.config['keys'].get("NEUTRE", "f13"))
        self.entry_neutral.pack(side=tk.LEFT, padx=5)

        # 2. FOOTER
        self.footer_frame = tk.Frame(self.left_container, bg="#f0f0f0")
        self.footer_frame.pack(side=tk.BOTTOM, fill="x", padx=10, pady=10)
        
        # Bouton Assistant (NOUVEAU)
        self.btn_wizard = tk.Button(self.footer_frame, text="🛠️ Assistant Setup Touches", command=self.open_wizard, bg="#FFD700", height=1)
        self.btn_wizard.pack(pady=5, fill="x")

        self.var_preview = tk.BooleanVar(value=True)
        tk.Checkbutton(self.footer_frame, text="Afficher Retour Caméra", var=self.var_preview, bg="#f0f0f0").pack(pady=5)
        self.btn_save = tk.Button(self.footer_frame, text="💾 Sauvegarder Config", command=self.save_config, bg="#ddd", height=2)
        self.btn_save.pack(pady=5, fill="x")
        self.btn_start = tk.Button(self.footer_frame, text="▶ DÉMARRER", command=self.toggle_camera, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), height=2)
        self.btn_start.pack(pady=5, fill="x")
        self.lbl_status = tk.Label(self.footer_frame, text="Status: Arrêté", fg="red", bg="#f0f0f0")
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
        
        actions = ["SOURIRE", "FRONCEMENT", "HAUSSEMENT", "MALICIEUX", "PENCHE", "CLIN_DOEIL", "REFLECHIR"]
        
        for act in actions:
            frame = tk.LabelFrame(self.scrollable_frame, bg="#f0f0f0", pady=5)
            frame.pack(fill="x", pady=5, padx=5)
            
            var_enable = tk.BooleanVar(value=self.config['enabled'].get(act, True))
            self.vars_enabled[act] = var_enable
            
            title = act
            if act == "REFLECHIR": title += " (Nécessite Mains)"
            
            chk = tk.Checkbutton(frame, text=title, variable=var_enable, font=("Arial", 10, "bold"), bg="#f0f0f0", anchor="w")
            chk.pack(fill="x", padx=5)

            grid_frame = tk.Frame(frame, bg="#f0f0f0")
            grid_frame.pack(fill="x", padx=5, pady=2)

            tk.Label(grid_frame, text="Touche:", bg="#f0f0f0").grid(row=0, column=0, sticky="w")
            entry = tk.Entry(grid_frame, width=6)
            entry.insert(0, self.config['keys'].get(act, ""))
            entry.grid(row=0, column=1, padx=5)
            self.entries[act] = entry

            tk.Label(grid_frame, text="Durée(s):", bg="#f0f0f0").grid(row=0, column=2, sticky="w")
            entry_dur = tk.Entry(grid_frame, width=5)
            entry_dur.insert(0, str(self.config['min_durations'].get(act, 0.5)))
            entry_dur.grid(row=0, column=3, padx=5)
            self.entries_duration[act] = entry_dur

            tk.Label(frame, text="Sensibilité:", bg="#f0f0f0", font=("Arial", 8)).pack(anchor="w", padx=5)
            slider = tk.Scale(frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, bg="#f0f0f0", length=250)
            slider.set(self.config['thresholds'].get(act, 0.5))
            slider.pack(fill="x", padx=5)
            self.sliders[act] = slider

        self.right_frame = tk.Frame(window, bg="black")
        self.right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)
        self.lbl_current_action = tk.Label(self.right_frame, text="En attente...", font=("Arial", 20, "bold"), bg="black", fg="white")
        self.lbl_current_action.pack(side=tk.TOP, fill="x", pady=5)
        self.canvas = tk.Canvas(self.right_frame, bg="#222")
        self.canvas.pack(expand=True, fill="both")
        self.delay = 15
        self.update()

    def _on_mousewheel(self, event):
        self.canvas_scroll.yview_scroll(int(-1*(event.delta/120)), "units")

    def open_wizard(self):
        # Sauvegarde d'abord pour être sûr d'avoir les touches à jour
        self.save_config_silent()
        SetupWizard(self.window, self.config)

    def load_config(self):
        default = {
            "keys": {
                "NEUTRE": "f13", "SOURIRE": "f14", "FRONCEMENT": "f15", 
                "HAUSSEMENT": "f16", "MALICIEUX": "f17", "PENCHE": "f18", 
                "CLIN_DOEIL": "f19", "REFLECHIR": "f20"
            },
            "thresholds": {
                "SOURIRE": 0.5, "FRONCEMENT": 0.4, "HAUSSEMENT": 0.5, 
                "MALICIEUX": 0.7, "PENCHE": 0.3, "CLIN_DOEIL": 0.2, "REFLECHIR": 0.6
            },
            "enabled": {
                "SOURIRE": True, "FRONCEMENT": True, "HAUSSEMENT": True, 
                "MALICIEUX": True, "PENCHE": True, "CLIN_DOEIL": True, "REFLECHIR": True
            },
            "min_durations": {
                "SOURIRE": 0.5, "FRONCEMENT": 0.5, "HAUSSEMENT": 0.5, 
                "MALICIEUX": 1.0, "PENCHE": 0.5, "CLIN_DOEIL": 0.2, "REFLECHIR": 1.0
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
        """ Sauvegarde sans afficher de popup (utile pour le wizard) """
        self.config['keys']['NEUTRE'] = self.entry_neutral.get()
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
        messagebox.showinfo("Info", "Configuration sauvegardée !")

    def toggle_camera(self):
        if self.detector.running:
            self.detector.running = False
            if self.detector.cap:
                self.detector.cap.release()
            self.btn_start.config(text="▶ DÉMARRER", bg="#4CAF50")
            self.lbl_status.config(text="Status: Arrêté", fg="red")
            self.canvas.delete("all")
        else:
            self.detector.cap = cv2.VideoCapture(self.video_source)
            self.detector.running = True
            self.btn_start.config(text="⏹ STOP", bg="#f44336")
            self.lbl_status.config(text="Status: En cours", fg="green")

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
                    key = self.entry_neutral.get() if action == "NEUTRE" else self.entries.get(action, tk.Entry()).get()
                    if key:
                        pyautogui.press(key)
                        print(f"Action: {action} -> {key}")
                    
                    self.detector.current_action = action
                    
                    color_map = {
                        "NEUTRE": "#00FF00", "MALICIEUX": "#A020F0", 
                        "FRONCEMENT": "#FF0000", "HAUSSEMENT": "#FFFF00", 
                        "SOURIRE": "#00FFFF", "PENCHE": "#FF1493",
                        "CLIN_DOEIL": "#FFA500", "REFLECHIR": "#1E90FF"
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
                    
                    if self.detector.has_hand_model and action == "REFLECHIR":
                        self.canvas.create_text(20, 20, text="MAIN DETECTEE", fill="cyan", anchor="nw")

                else:
                    if len(self.canvas.find_all()) > 1:
                         self.canvas.delete("all")
                         self.canvas.create_text(self.canvas.winfo_width()//2, self.canvas.winfo_height()//2, 
                                               text="MONITORING ACTIF\n(Aperçu désactivé)", fill="gray", font=("Arial", 14))

        self.window.after(self.delay, self.update)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root, "PNGTuber Controller v1.0")
    root.mainloop()