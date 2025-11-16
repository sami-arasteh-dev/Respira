import threading, time, json, hashlib, queue, base64
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.signal import butter, filtfilt, find_peaks

# --- CONSTANTS and Utilities ---
FILE_SIGNATURE = b"RESPIRA_V1.5"
MIN_PROMINENCE_THRESHOLD = 0.005

def butter_lowpass(fc, fs, order=4):
    b, a = butter(order, fc/(fs/2), btype='low')
    return b, a

def compute_envelope(x, fs, lowpass_hz=0.7):
    if len(x) == 0:
        return np.array([], dtype=np.float32)
    max_val = np.max(np.abs(x))
    if max_val < 1e-6:
        return np.zeros_like(x)
    x_norm = x / max_val
    rect = np.abs(x_norm)
    b, a = butter_lowpass(lowpass_hz, fs)
    return filtfilt(b, a, rect)

# (VAD and Cycle Detection functions remain the same for robustness)
def adaptive_vad_mask(x, fs, frame_sec=0.03, noise_sample_sec=3.0, noise_factor=2.5):
    frame_len = int(frame_sec * fs)
    if frame_len <= 0:
        frame_len = int(0.03 * fs)
        
    noise_samples = int(noise_sample_sec * fs)
    noise_segment = x[:noise_samples]
    
    if len(noise_segment) < frame_len:
        adaptive_thresh = 0.005 
    else:
        noise_energies = [np.mean(np.abs(noise_segment[i:i + frame_len])) 
                          for i in range(0, len(noise_segment) - frame_len, frame_len)]
        noise_floor = np.median(noise_energies) if noise_energies else 1e-4
        
        adaptive_thresh = max(0.005, noise_floor * noise_factor)

    flags = []
    for i in range(0, len(x), frame_len):
        frame = x[i:i + frame_len]
        if len(frame) == 0:
            continue
        energy = np.mean(np.abs(frame))
        flags.extend([energy > adaptive_thresh] * len(frame))
        
    return np.array(flags[:len(x)]) if len(flags) > 0 else np.zeros_like(x, dtype=bool)

def detect_breath_cycles(env, fs, min_breath_sec=2.0):
    if len(env) < int(2.0 * fs):
        return [], None

    env_std = float(np.std(env))
    prominence = max(0.2 * env_std, MIN_PROMINENCE_THRESHOLD)
    
    distance = int(fs * min_breath_sec)
    peaks, props = find_peaks(env, distance=distance, prominence=prominence)

    if len(peaks) < 2:
        return [], None

    intervals = np.diff(peaks) / fs

    deriv_raw = np.gradient(env)
    dwin = 9
    kernel = np.ones(dwin) / dwin
    deriv = np.convolve(deriv_raw, kernel, mode='same')

    cycles = []
    for i in range(len(peaks) - 1):
        s, e = peaks[i], peaks[i + 1]
        if e <= s:
            continue
        seg = env[s:e]
        if len(seg) < int(0.3 * fs):
            continue

        peak_abs = s + int(np.argmax(seg))
        peak_val = env[peak_abs]
        energy_thresh = 0.6 * peak_val

        trans_idx = None
        for k in range(peak_abs, e):
            if deriv[k] < 0 and env[k] <= energy_thresh:
                trans_idx = k
                break
        if trans_idx is None:
            for k in range(peak_abs, e):
                if env[k] <= energy_thresh:
                    trans_idx = k
                    break
        if trans_idx is None:
            trans_idx = s + int(0.65 * (e - s))

        t_inhale = (trans_idx - s) / fs
        t_exhale = (e - trans_idx) / fs
        cycle_dur = (e - s) / fs

        if t_inhale < 0.3 or t_exhale < 0.3:
            continue
        if cycle_dur < 0.6 or cycle_dur > 8.0:
            continue

        amp = float(np.max(seg))
        inh_ex_ratio = float((t_inhale + 1e-8) / (t_exhale + 1e-8))

        if not (0.4 <= inh_ex_ratio <= 2.5):
            continue

        cycles.append({
            "start_sample": int(s), "end_sample": int(e),
            "duration_sec": float(cycle_dur), "amplitude": amp,
            "inhale_sec": float(t_inhale), "exhale_sec": float(t_exhale),
            "inhale_exhale_ratio": float(inh_ex_ratio)
        })

    return cycles, intervals

def compute_features(cycles, intervals, x, fs, speech_mask):
    if intervals is None or len(intervals) == 0:
        return None

    rpm = 60.0 / float(np.median(intervals))

    if len(cycles) == 0:
        noise_level = float(np.mean(np.abs(x[speech_mask])) if np.any(speech_mask) else 0.0)
        return {
            "respirations_per_min": float(rpm),
            "interval_mean_sec": float(np.mean(intervals)),
            "amplitude_mean": 0.0,
            "amplitude_cv": 0.0,
            "noise_level": noise_level, 
            "confidence": 0.0
        }

    amp_series = np.array([c["amplitude"] for c in cycles])
    dur_series = np.array([c["duration_sec"] for c in cycles])
    inh_ex_series = np.array([c["inhale_exhale_ratio"] for c in cycles])

    amp_mean = float(np.mean(amp_series)) if len(amp_series) else 0.0
    amp_std = float(np.std(amp_series)) if len(amp_series) else 0.0
    amp_cv = float(amp_std / amp_mean) if amp_mean > 0 else 0.0

    noise_level = float(np.mean(np.abs(x[speech_mask])) if np.any(speech_mask) else 0.0)

    return {
        "respirations_per_min": float(rpm),
        "interval_mean_sec": float(np.mean(intervals)),
        "amplitude_mean": amp_mean,
        "amplitude_cv": amp_cv,
        "duration_mean_sec": float(np.mean(dur_series)) if len(dur_series) else 0.0,
        "inhale_exhale_ratio_mean": float(np.mean(inh_ex_series)) if len(inh_ex_series) else 0.0,
        "noise_level": noise_level,
        "confidence": float(len(cycles) / max(1, len(intervals)))
    }

# --- Splash Screen ---
class SplashScreen(tk.Toplevel):
    def __init__(self, parent):
        tk.Toplevel.__init__(self, parent)
        self.overrideredirect(True) 
        self.configure(bg="#000033")
        
        # Center the splash screen
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        window_width = 400
        window_height = 200
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        self.geometry(f'{window_width}x{window_height}+{x}+{y}')
        
        ttk.Label(self, text="Respira", font=("Arial", 36, "bold"), foreground="white", background="#000033").pack(pady=30)
        ttk.Label(self, text="Real-time Signal Analysis", font=("Arial", 12), foreground="#AAAAAA", background="#000033").pack(pady=5)
        ttk.Label(self, text="Loading...", font=("Arial", 10), foreground="#CCCCCC", background="#000033").pack(pady=10)

        self.update()
        self.after(2000, self.destroy) # Display for 2 seconds

# --- Settings Dialog ---
class SettingsDialog(simpledialog.Dialog):
    def __init__(self, parent, app_instance):
        self.app = app_instance
        self.temp_lowpass = tk.DoubleVar(value=app_instance.lowpass_hz)
        self.temp_min_breath = tk.DoubleVar(value=app_instance.min_breath_sec)
        self.temp_vad_factor = tk.DoubleVar(value=app_instance.vad_noise_factor)
        super().__init__(parent, title="Analysis Settings")

    def body(self, master):
        ttk.Label(master, text="Env. Lowpass (Hz):").grid(row=0, sticky=tk.W)
        ttk.Entry(master, textvariable=self.temp_lowpass).grid(row=0, column=1)

        ttk.Label(master, text="Min Breath Duration (sec):").grid(row=1, sticky=tk.W)
        ttk.Entry(master, textvariable=self.temp_min_breath).grid(row=1, column=1)

        ttk.Label(master, text="VAD Noise Factor:").grid(row=2, sticky=tk.W)
        ttk.Entry(master, textvariable=self.temp_vad_factor).grid(row=2, column=1)

        return master

    def apply(self):
        self.app.lowpass_hz = self.temp_lowpass.get()
        self.app.min_breath_sec = self.temp_min_breath.get()
        self.app.vad_noise_factor = self.temp_vad_factor.get()
        messagebox.showinfo("Settings", "Settings applied successfully. They will be used in the next recording.")

# --- GUI ---
class RespiratoryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Respira — Real-time Respiratory Analysis")
        self.root.withdraw() # Hide main window until splash is done
        
        # Configurable Parameters
        self.fs = 16000
        self.lowpass_hz = 0.7
        self.min_breath_sec = 2.0
        self.vad_frame = 0.03          
        self.vad_noise_factor = 2.5    
        self.record_seconds = tk.IntVar(value=20)
        
        # Internal State
        self.audio_queue = queue.Queue()
        self.running = False
        self.audio_buffer = np.array([], dtype=np.float32)
        self.speech_mask = None
        self.env_display = np.array([], dtype=np.float32)
        self.cycles = []
        self.features = None
        self.animation_state = 0
        self.animation_id = None

        # Show splash screen first
        splash = SplashScreen(root)
        root.wait_window(splash)
        self.root.deiconify() # Show main window
        
        # *** FIX 1: Prevent window resizing ***
        self.root.resizable(False, False) 

        self._build_ui()
        self._build_plot()
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    # *** FIX 2: open_settings method moved inside the class ***
    def open_settings(self):
        """Opens the Settings dialog."""
        if self.running:
             messagebox.showwarning("Respira", "Cannot change settings while recording is active.")
             return
        SettingsDialog(self.root, self)

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding="10 10 10 10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left side: Controls and Features
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=5)
        
        # Control Frame
        top = ttk.Frame(left_frame)
        top.pack(side=tk.TOP, fill=tk.X, pady=5)

        ttk.Label(top, text="Duration (sec):").pack(side=tk.LEFT)
        ttk.Entry(top, width=6, textvariable=self.record_seconds).pack(side=tk.LEFT, padx=6)

        self.btn_start = ttk.Button(top, text="Start Recording", command=self.start_recording)
        self.btn_start.pack(side=tk.LEFT, padx=6)
        self.btn_stop = ttk.Button(top, text="Stop", command=self.stop_recording, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=6)
        
        ttk.Button(top, text="Settings ⚙️", command=self.open_settings).pack(side=tk.LEFT, padx=12)

        export_import_frame = ttk.Frame(top)
        export_import_frame.pack(side=tk.LEFT, padx=10)
        self.btn_export = ttk.Button(export_import_frame, text="Export .rspr", command=self.export_rsper, state=tk.DISABLED)
        self.btn_export.pack(side=tk.LEFT, padx=3)
        self.btn_import = ttk.Button(export_import_frame, text="Import .rspr", command=self.import_rsper)
        self.btn_import.pack(side=tk.LEFT, padx=3)

        # Feature Table
        table_frame = ttk.LabelFrame(left_frame, text="Analytical Features")
        table_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=10)
        self.table = ttk.Treeview(table_frame, columns=("metric","value"), show="headings", height=10)
        self.table.heading("metric", text="Metric")
        self.table.heading("value", text="Value")
        self.table.column("metric", width=220)
        self.table.column("value", width=180)
        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.table.configure(yscrollcommand=scroll.set)

        # Cycle Plot
        cycle_frame = ttk.LabelFrame(left_frame, text="Cycle Plot (Amplitude vs Index)")
        cycle_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=10)
        self.cycle_fig = Figure(figsize=(6,2), dpi=100)
        self.cycle_ax = self.cycle_fig.add_subplot(111)
        self.cycle_ax.set_title("Breath Cycles Amplitude")
        self.cycle_ax.set_xlabel("Cycle index")
        self.cycle_ax.set_ylabel("Amplitude")
        self.cycle_canvas = FigureCanvasTkAgg(self.cycle_fig, master=cycle_frame)
        self.cycle_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Right side: Animation and Live Plot
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Animation Frame
        self.anim_frame = ttk.LabelFrame(right_frame, text="🎙️ Microphone Guide")
        self.anim_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        # *** FIX 3: Container with fixed width to prevent window resizing from text length ***
        anim_container = ttk.Frame(self.anim_frame, width=350, height=35) 
        anim_container.pack_propagate(False) 
        anim_container.pack(fill=tk.X, pady=5, padx=10)
        
        self.anim_label = ttk.Label(anim_container, text="آماده ضبط", font=("Arial", 12))
        self.anim_label.pack(expand=True, fill=tk.BOTH)
        
        self.start_animation()


    def _build_plot(self):
        # Live Plot
        plot_frame = ttk.LabelFrame(self.root, text="Real-time Respiratory Envelope")
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.fig = Figure(figsize=(6,2.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Respiratory Envelope (live)")
        self.ax.set_xlabel("Samples")
        self.ax.set_ylabel("Amplitude")
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # --- Animation ---
    def start_animation(self):
        if self.animation_id:
            self.root.after_cancel(self.animation_id)
        self._update_animation()

    def _update_animation(self):
        # States for the animation
        states = [
            "🎙️ 👈 Mouth (Closer)",
            "🎙️ 👈 Mouth (10 - 15 cm)",
            "🎙️ 👈 Mouth (reduce ambient noise)"
        ]
        
        self.anim_label.config(text=states[self.animation_state])
        self.animation_state = (self.animation_state + 1) % len(states)
        self.animation_id = self.root.after(1500, self._update_animation)

    def stop_animation(self):
        if self.animation_id:
            self.root.after_cancel(self.animation_id)
            self.animation_id = None
        self.anim_label.config(text="Analyzing ..")
        
    def _on_closing(self):
        """Handle window closing event."""
        if self.running:
            self.stop_recording(auto_stop=False)
        if self.animation_id:
            self.root.after_cancel(self.animation_id)
        self.root.destroy()
        
    def audio_callback(self, indata, frames, time_info, status):
        """Callback from sounddevice stream."""
        if self.running:
            if status:
                print(f"[Audio status warning] {status}")
            self.audio_queue.put(indata.copy())

    def start_recording(self):
        if self.running: return
        
        self.running = True
        self.audio_buffer = np.array([], dtype=np.float32)
        self.env_display = np.array([], dtype=np.float32)
        self.cycles = []
        self.features = None
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_export.config(state=tk.DISABLED)
        self.stop_animation() # Stop guidance animation
        self.anim_label.config(text="Recording ..")


        try:
            self.stream = sd.InputStream(samplerate=self.fs, channels=1, dtype='float32', callback=self.audio_callback)
            self.stream.start()
        except Exception as e:
            messagebox.showerror("Respira", f"Failed to start audio stream:\n{e}")
            self.running = False
            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            self.start_animation()
            return

        threading.Thread(target=self._consume_audio, daemon=True).start()
        threading.Thread(target=self._auto_stop_after, daemon=True).start()

    def _auto_stop_after(self):
        """Stops recording after the specified duration."""
        t_end = time.time() + int(self.record_seconds.get())
        while self.running and time.time() < t_end:
            time.sleep(0.1)
        
        if self.running:
            self.root.after(0, self.stop_recording) 

    def _consume_audio(self):
        """Thread to consume audio data from the queue and update the live plot."""
        chunk_list = []
        try:
            while self.running:
                try:
                    data = self.audio_queue.get(timeout=0.1)
                    chunk = data.squeeze().astype(np.float32)
                    chunk_list.append(chunk)
                    
                    self.audio_buffer = np.concatenate(chunk_list) if len(chunk_list) > 0 else np.array([], dtype=np.float32)
                    
                    N = min(len(self.audio_buffer), int(self.fs * 8))
                    if N > int(self.fs * 0.5):
                        window = self.audio_buffer[-N:]
                        env = compute_envelope(window, self.fs, self.lowpass_hz)
                        self.env_display = env
                        self.root.after(0, lambda e=env: self._update_live_plot(e))
                        
                except queue.Empty:
                    continue
                
        except Exception as e:
            print(f"[Audio Consumer Error] {e}")
            self.root.after(0, lambda: messagebox.showerror("Respira Error", f"Critical audio processing error:\n{e}"))
            self.root.after(0, self.stop_recording)

    def _update_live_plot(self, env):
        """Updates the Matplotlib live plot (must be called from the main thread)."""
        try:
            self.ax.cla()
            self.ax.set_title("Respiratory Envelope (live)")
            self.ax.set_xlabel(f"Samples (fs={self.fs}Hz)")
            self.ax.set_ylabel("Amplitude (Normalized)")
            if env is not None and len(env) > 0:
                self.ax.plot(env, color='blue', lw=1.2)
            self.canvas.draw_idle()
        except Exception as e:
            print(f"[Live plot warning] {e}")

    def stop_recording(self, auto_stop=True):
        """Stops the audio stream and performs final analysis."""
        if not self.running: return

        self.running = False
        try:
            if hasattr(self, 'stream') and self.stream:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
            
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.anim_label.config(text="Analyzing ..")

        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        """Performs VAD, envelope, cycle detection, and feature calculation."""
        if len(self.audio_buffer) < self.fs * 4:
            self.root.after(0, lambda: messagebox.showwarning("Respira", "Recording too short. Please record longer."))
            self.root.after(0, lambda: self._populate_table({}))
            self.root.after(0, self.start_animation)
            return

        self.speech_mask = adaptive_vad_mask(
            self.audio_buffer, 
            self.fs, 
            frame_sec=self.vad_frame,
            noise_factor=self.vad_noise_factor
        )
        x_ns = self.audio_buffer[~self.speech_mask] if np.any(~self.speech_mask) else self.audio_buffer.copy()

        env = compute_envelope(x_ns, self.fs, self.lowpass_hz)

        cycles, intervals = detect_breath_cycles(env, self.fs, min_breath_sec=self.min_breath_sec)
        self.cycles = cycles
        self.features = compute_features(cycles, intervals, self.audio_buffer, self.fs, self.speech_mask)
        
        if self.features is None:
            self.root.after(0, lambda: messagebox.showwarning("Respira", "No valid cycles detected. Please adjust microphone position or settings."))
            self.root.after(0, lambda: self._populate_table({}))
            self.root.after(0, lambda: self._update_cycle_plot([]))
            self.root.after(0, lambda: self.btn_export.config(state=tk.NORMAL))
            self.root.after(0, self.start_animation)
            return

        self.root.after(0, lambda: self._populate_table(self.features))
        self.root.after(0, lambda: self._update_cycle_plot(self.cycles))
        self.root.after(0, lambda: self.btn_export.config(state=tk.NORMAL))
        self.root.after(0, self.start_animation)


    def _populate_table(self, features):
        self.table.delete(*self.table.get_children())
        if not features:
            self.table.insert("", tk.END, values=("status", "No valid cycles"))
            return

        ordered_keys = [
            "respirations_per_min", "interval_mean_sec", 
            "amplitude_mean", "amplitude_cv", 
            "duration_mean_sec", 
            "inhale_exhale_ratio_mean", 
            "noise_level", "confidence"
        ]
        for k in ordered_keys:
            val = features.get(k, "")
            if isinstance(val, float):
                self.table.insert("", tk.END, values=(k, f"{val:.4f}"))
            else:
                self.table.insert("", tk.END, values=(k, str(val)))

    def _update_cycle_plot(self, cycles):
        try:
            self.cycle_ax.cla()
            self.cycle_ax.set_title("Breath Cycles Amplitude")
            self.cycle_ax.set_xlabel("Cycle index")
            self.cycle_ax.set_ylabel("Amplitude (Normalized)")
            if cycles and len(cycles) > 0:
                amps = [c["amplitude"] for c in cycles]
                self.cycle_ax.plot(range(len(amps)), amps, "o-", color="green")
            self.cycle_canvas.draw_idle()
        except Exception as e:
            print(f"[Cycle plot warning] {e}")

    # (Export/Import methods remain the same)
    def export_rsper(self):
        header = {
            "version": "1.5", "fs": self.fs,
            "duration_sec": int(self.record_seconds.get()),
            "timestamp": int(time.time()), "device": "RespiratoryApp",
            "method": {
                "envelope_lowpass_hz": self.lowpass_hz,
                "vad": "adaptive_noise_threshold",
                "vad_noise_factor": self.vad_noise_factor,
                "cycle_split": "smoothed_derivative + adaptive_energy + fallbacks",
                "min_breath_sec": self.min_breath_sec
            }
        }
        payload = {
            "header": header,
            "signature": self.features if self.features else {},
            "cycles": self.cycles if self.cycles else []
        }
        data_str = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        data_bytes = data_str.encode('utf-8')
        hash_hex = hashlib.sha256(data_bytes).hexdigest()
        encoded_data = base64.b64encode(data_bytes).decode('utf-8')
        final_file_content = {
            "signature_id": FILE_SIGNATURE.decode('utf-8'),
            "hash": hash_hex,
            "encoded_data": encoded_data
        }
        file_json_str = json.dumps(final_file_content, separators=(',', ':'))
        fname = filedialog.asksaveasfilename(
            title="Save .rspr",
            defaultextension=".rspr",
            filetypes=[("Respira Signature", "*.rspr")]
        )
        if not fname: return
        try:
            with open(fname, "w", encoding="utf-8") as f:
                f.write(file_json_str)
            messagebox.showinfo("Respira", f"Saved signature to:\n{fname}")
        except Exception as e:
            messagebox.showerror("Respira", f"Failed to save file:\n{e}")

    def import_rsper(self):
        fname = filedialog.askopenfilename(
            title="Open .rspr",
            filetypes=[("Respira Signature", "*.rspr")]
        )
        if not fname: return
        try:
            with open(fname, "r", encoding="utf-8") as f:
                file_content = json.load(f)
        except Exception as e:
            messagebox.showerror("Respira", f"Failed to open/parse file:\n{e}")
            return
        if file_content.get("signature_id") != FILE_SIGNATURE.decode('utf-8'):
             messagebox.showerror("Respira", "Invalid or unsupported .rspr file version.")
             return
        encoded_data = file_content.get("encoded_data")
        expected_hash = file_content.get("hash")
        if not encoded_data or not expected_hash:
            messagebox.showerror("Respira", "Invalid .rspr file format (missing data/hash).")
            return
        try:
            data_bytes = base64.b64decode(encoded_data)
            data_str = data_bytes.decode('utf-8')
            computed_hash = hashlib.sha256(data_bytes).hexdigest()
            if computed_hash != expected_hash:
                 messagebox.showwarning("Respira", "File integrity check failed: Hash mismatch. Data might be corrupted or tampered with.")
            data = json.loads(data_str)
        except Exception as e:
            messagebox.showerror("Respira", f"Failed to decode or process file content:\n{e}")
            return
        if "signature" not in data or "cycles" not in data:
            messagebox.showerror("Respira", "Invalid decoded .rspr content.")
            return

        self.features = data.get("signature", {})
        self.cycles = data.get("cycles", [])

        self._populate_table(self.features)
        self._update_cycle_plot(self.cycles)
        messagebox.showinfo("Respira", "Imported .rspr successfully.")


# --- Run ---
def main():
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    app = RespiratoryApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
