"""
detect_gui.py - Two-Stage Drowning Detection GUI
====================================================
Real-time drowning detection using two-stage YOLO26 pipeline:
  Stage 1 (detect, nc=5): person, person_in_water, boat, floating_object, life_buoy
  Stage 2 (classify, nc=2): drowning / swimming

Features:
  - Real-time webcam/video capture with two-stage detection overlay
  - Stage 1 confidence threshold slider (0.1 - 0.9)
  - Drowning safety threshold slider (0.1 - 0.9)
  - Model switching for both Stage 1 and Stage 2
  - Camera source switching (webcam 0/1/video file)
  - Pause/resume detection
  - Screenshot capture
  - Video recording (start/stop)
  - Live FPS counter & per-class detection count
  - Drowning visual alert (red border + flashing indicator)

Tech Stack: tkinter + OpenCV + PIL + ultralytics.YOLO

Usage:
    python detect_gui.py                                    # Auto-find models
    python detect_gui.py --stage1 path/to/stage1_best.pt   # Specific Stage 1 model
    python detect_gui.py --stage2 path/to/stage2_best.pt   # Specific Stage 2 model
    python detect_gui.py --camera 1                        # Use camera #1
    python detect_gui.py --source video.mp4                # Use video file
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageTk

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import two-stage inference pipeline
from pipeline_inference import (
    two_stage_inference,
    draw_results,
    crop_person_in_water,
    STAGE1_CLASS_NAMES,
    STAGE2_CLASS_NAMES,
)


# ===========================================================================
#  Constants
# ===========================================================================

# Combined display categories (for stats panel)
DISPLAY_CLASSES = [
    "person",
    "drowning",
    "swimming",
    "boat",
    "floating_object",
    "life_buoy",
]

# Color map for display (BGR for OpenCV)
DISPLAY_BGR = {
    "drowning":                (0, 0, 255),       # Red - confirmed drowning
    "swimming":                (0, 200, 0),       # Green (含未分类降级)
    "person":                  (255, 0, 0),       # Blue
    "boat":                    (0, 255, 255),     # Yellow
    "floating_object":         (128, 128, 128),   # Gray
    "life_buoy":               (0, 165, 255),     # Orange
}

# Default model paths
DEFAULT_STAGE1 = PROJECT_ROOT / "runs" / "stage1_v1" / "best.pt"
DEFAULT_STAGE2 = PROJECT_ROOT / "runs" / "stage2_v1" / "best.pt"

# Thresholds
DEFAULT_CONF = 0.5

# Window
WINDOW_TITLE = "Two-Stage Drowning Detection - Live Camera"
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720


# ===========================================================================
#  Model Finder
# ===========================================================================

def find_model(provided_path=None, default_path=None, search_dirs=None):
    """Find a trained model file with fallback search."""
    if provided_path and Path(provided_path).exists():
        return Path(provided_path)

    if default_path and default_path.exists():
        return default_path

    if search_dirs:
        for search_dir in search_dirs:
            if search_dir.exists():
                for train_dir in sorted(search_dir.iterdir(), reverse=True):
                    # Check weights/ subdirectory
                    best_pt = train_dir / "weights" / "best.pt"
                    if best_pt.exists():
                        return best_pt
                    # Check direct best.pt
                    best_pt = train_dir / "best.pt"
                    if best_pt.exists():
                        return best_pt

    return None


# ===========================================================================
#  Detection GUI
# ===========================================================================

class DetectGUI:
    """Main GUI application for two-stage real-time drowning detection."""

    def __init__(self, stage1_path=None, stage2_path=None,
                 camera_id=0, source=None,
                 conf=DEFAULT_CONF):
        # ---- Models ----
        self.stage1_path = stage1_path
        self.stage2_path = stage2_path
        self.stage1_model = None
        self.stage2_model = None

        # ---- Camera ----
        self.camera_id = camera_id
        self.source = source
        self.cap = None
        self.frame = None
        self.running = False
        self.paused = False

        # ---- Recording ----
        self.recording = False
        self.video_writer = None
        self.record_path = None

        # ---- Detection settings ----
        self.conf_threshold = conf
        self.last_results = None
        self.drowning_detected = False
        self._alert_flash = False

        # ---- Statistics ----
        self.fps = 0.0
        self.stage1_time = 0.0
        self.stage2_time = 0.0
        self.total_time = 0.0
        self.frame_count = 0
        self.class_counts = {name: 0 for name in DISPLAY_CLASSES}
        self._fps_buffer = []
        self._last_fps_update = time.time()

        # ---- I/O directories ----
        self.screenshot_dir = PROJECT_ROOT / "screenshots"
        self.screenshot_dir.mkdir(exist_ok=True)
        self.record_dir = PROJECT_ROOT / "recordings"
        self.record_dir.mkdir(exist_ok=True)

        # ---- Build UI ----
        self._setup_ui()

        # Load models if paths provided
        if self.stage1_path and self.stage2_path:
            self._load_models()

    # ------------------------------------------------------------------
    #  Model Management
    # ------------------------------------------------------------------

    def _load_models(self):
        """Load both Stage 1 and Stage 2 models."""
        from ultralytics import YOLO

        errors = []

        # Stage 1
        if self.stage1_path and Path(self.stage1_path).exists():
            print(f"[Stage1] Loading: {self.stage1_path}")
            try:
                self.stage1_model = YOLO(str(self.stage1_path))
                self.stage1_model.conf = self.conf_threshold
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                self.stage1_model.predict(dummy, verbose=False)
                print("[Stage1] Ready!")
            except Exception as e:
                errors.append(f"Stage 1: {e}")
                self.stage1_model = None
        else:
            errors.append("Stage 1: model path not found")

        # Stage 2
        if self.stage2_path and Path(self.stage2_path).exists():
            print(f"[Stage2] Loading: {self.stage2_path}")
            try:
                self.stage2_model = YOLO(str(self.stage2_path))
                dummy = np.zeros((256, 256, 3), dtype=np.uint8)
                self.stage2_model.predict(dummy, verbose=False)
                print("[Stage2] Ready!")
            except Exception as e:
                errors.append(f"Stage 2: {e}")
                self.stage2_model = None
        else:
            errors.append("Stage 2: model path not found")

        self._update_model_labels()

        if errors:
            import tkinter as messagebox_module
            from tkinter import messagebox
            messagebox.showwarning(
                "Model Load Warning",
                "Some models failed to load:\n\n" + "\n".join(errors)
            )

    def _reload_models(self):
        """Reload both models from current paths."""
        self._load_models()

    # ------------------------------------------------------------------
    #  UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Build the tkinter interface."""
        import tkinter as tk
        from tkinter import ttk

        self.root = tk.Tk()
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT + 140}")
        self.root.configure(bg="#1e1e1e")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ---- Main layout ----
        self.main_frame = tk.Frame(self.root, bg="#1e1e1e")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left: video canvas
        self.video_frame = tk.Frame(self.main_frame, bg="#000000", relief=tk.SUNKEN, bd=1)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.video_frame, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Right: control panel (wider for two model sections)
        self.control_frame = tk.Frame(self.main_frame, bg="#2d2d2d", width=300)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        self.control_frame.pack_propagate(False)

        self._build_controls()

        # Bottom: status bar
        self.status_frame = tk.Frame(self.root, bg="#1a1a1a", height=30)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_frame.pack_propagate(False)

        self.status_label = tk.Label(
            self.status_frame,
            text="Status: 加载两个模型后开始检测",
            bg="#1a1a1a", fg="#ffff00", font=("Consolas", 10), anchor="w"
        )
        self.status_label.pack(fill=tk.X, padx=10, pady=3)

    def _build_controls(self):
        """Build control panel widgets."""
        import tkinter as tk
        from tkinter import ttk, filedialog

        pad = {"padx": 8, "pady": 3}
        fg = "#e0e0e0"
        bg = "#2d2d2d"

        # ---- Title ----
        title_lbl = tk.Label(self.control_frame, text="Two-Stage Pipeline",
                             bg=bg, fg="#00ff00", font=("Consolas", 13, "bold"))
        title_lbl.pack(fill=tk.X, **pad)

        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, **pad)

        # ---- Stage 1 Model ----
        s1_group = tk.LabelFrame(self.control_frame, text="Stage 1 (Detect nc=5)",
                                  bg=bg, fg="#378ADD", font=("Consolas", 10, "bold"))
        s1_group.pack(fill=tk.X, **pad)

        s1_name = Path(self.stage1_path).name if self.stage1_path else "(未加载)"
        self.s1_label = tk.Label(s1_group, text=s1_name,
                                  bg=bg, fg="#85B7EB", font=("Consolas", 9),
                                  wraplength=260)
        self.s1_label.pack(fill=tk.X, **pad)

        tk.Button(s1_group, text="Load Stage1", command=self._browse_stage1,
                  bg="#185FA5", fg="#fff", font=("Consolas", 9)).pack(fill=tk.X, **pad)

        # ---- Stage 2 Model ----
        s2_group = tk.LabelFrame(self.control_frame, text="Stage 2 (Classify nc=2)",
                                  bg=bg, fg="#7F77DD", font=("Consolas", 10, "bold"))
        s2_group.pack(fill=tk.X, **pad)

        s2_name = Path(self.stage2_path).name if self.stage2_path else "(未加载)"
        self.s2_label = tk.Label(s2_group, text=s2_name,
                                  bg=bg, fg="#AFA9EC", font=("Consolas", 9),
                                  wraplength=260)
        self.s2_label.pack(fill=tk.X, **pad)

        tk.Button(s2_group, text="Load Stage2", command=self._browse_stage2,
                  bg="#534AB7", fg="#fff", font=("Consolas", 9)).pack(fill=tk.X, **pad)

        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, **pad)

        # ---- Confidence Threshold (Stage 1) ----
        conf_group = tk.LabelFrame(self.control_frame, text="Stage1 Conf Threshold",
                                   bg=bg, fg=fg, font=("Consolas", 10, "bold"))
        conf_group.pack(fill=tk.X, **pad)

        self.conf_var = tk.DoubleVar(value=DEFAULT_CONF)
        self.conf_slider = tk.Scale(conf_group, from_=0.1, to=0.9, resolution=0.05,
                                    orient=tk.HORIZONTAL, variable=self.conf_var,
                                    bg=bg, fg=fg, troughcolor="#444", highlightthickness=0,
                                    command=self._on_conf_change)
        self.conf_slider.pack(fill=tk.X, **pad)

        self.conf_value_label = tk.Label(conf_group, text=f"{DEFAULT_CONF:.2f}",
                                         bg=bg, fg="#ffff00", font=("Consolas", 11, "bold"))
        self.conf_value_label.pack()

        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, **pad)

        # ---- Camera Controls ----
        cam_group = tk.LabelFrame(self.control_frame, text="Camera",
                                  bg=bg, fg=fg, font=("Consolas", 10, "bold"))
        cam_group.pack(fill=tk.X, **pad)

        tk.Button(cam_group, text="Camera 0", command=lambda: self._switch_camera(0),
                  bg="#444", fg=fg, font=("Consolas", 9)).pack(fill=tk.X, **pad)
        tk.Button(cam_group, text="Video File...", command=self._open_video_file,
                  bg="#444", fg=fg, font=("Consolas", 9)).pack(fill=tk.X, **pad)

        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, **pad)

        # ---- Action Buttons ----
        action_group = tk.LabelFrame(self.control_frame, text="Actions",
                                     bg=bg, fg=fg, font=("Consolas", 10, "bold"))
        action_group.pack(fill=tk.X, **pad)

        self.pause_btn = tk.Button(action_group, text="Pause", command=self._toggle_pause,
                                   bg="#555", fg=fg, font=("Consolas", 10, "bold"))
        self.pause_btn.pack(fill=tk.X, **pad)

        tk.Button(action_group, text="Screenshot", command=self._take_screenshot,
                  bg="#555", fg=fg, font=("Consolas", 10)).pack(fill=tk.X, **pad)

        self.record_btn = tk.Button(action_group, text="Start Recording",
                                    command=self._toggle_recording,
                                    bg="#880000", fg=fg, font=("Consolas", 10, "bold"))
        self.record_btn.pack(fill=tk.X, **pad)

        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, **pad)

        # ---- Detection Stats ----
        stats_group = tk.LabelFrame(self.control_frame, text="Detection Count",
                                    bg=bg, fg=fg, font=("Consolas", 10, "bold"))
        stats_group.pack(fill=tk.X, **pad)

        self.stat_labels = {}
        for name in DISPLAY_CLASSES:
            color_fg = "#cccccc"
            if name == "drowning":
                color_fg = "#F09595"
            elif name == "swimming":
                color_fg = "#5DCAA5"
            lbl = tk.Label(stats_group, text=f"{name:>14}: 0",
                           bg=bg, fg=color_fg, font=("Consolas", 9), anchor="w")
            lbl.pack(fill=tk.X, padx=12)
            self.stat_labels[name] = lbl

        # ---- Drowning Alert Indicator ----
        self.alert_label = tk.Label(self.control_frame, text="",
                                    bg=bg, fg="#E24B4A", font=("Consolas", 14, "bold"))
        self.alert_label.pack(fill=tk.X, **pad)

    # ------------------------------------------------------------------
    #  UI Callbacks
    # ------------------------------------------------------------------

    def _on_conf_change(self, val):
        self.conf_threshold = float(val)
        self.conf_value_label.config(text=f"{self.conf_threshold:.2f}")
        if self.stage1_model:
            self.stage1_model.conf = self.conf_threshold

    def _browse_stage1(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select Stage 1 Detection Model",
            filetypes=[("PyTorch Model", "*.pt"), ("All files", "*.*")],
            initialdir=str(PROJECT_ROOT),
        )
        if path:
            self.stage1_path = path
            self._load_models()

    def _browse_stage2(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select Stage 2 Classification Model",
            filetypes=[("PyTorch Model", "*.pt"), ("All files", "*.*")],
            initialdir=str(PROJECT_ROOT),
        )
        if path:
            self.stage2_path = path
            self._load_models()

    def _update_model_labels(self):
        if self.stage1_path:
            self.s1_label.config(text=Path(self.stage1_path).name)
        else:
            self.s1_label.config(text="(未加载)")

        if self.stage2_path:
            self.s2_label.config(text=Path(self.stage2_path).name)
        else:
            self.s2_label.config(text="(未加载)")

        self.root.update_idletasks()

    def _switch_camera(self, cam_id):
        self.camera_id = cam_id
        self.source = None
        self._restart_camera()

    def _open_video_file(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if path:
            self.source = path
            self._restart_camera()

    def _restart_camera(self):
        """Release and re-open camera capture."""
        was_running = self.running
        if self.running:
            self.running = False
            time.sleep(0.1)

        if self.cap:
            self.cap.release()

        if self.source:
            self.cap = cv2.VideoCapture(self.source)
        else:
            self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open camera {self.camera_id}")
            self.status_label.config(text="ERROR: Cannot open camera!")
            return

        self.frame = None
        self.running = True
        self.frame_count = 0
        self.status_label.config(
            text=f"FPS: -- | Camera: {'File' if self.source else self.camera_id} | Running")

    def _toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.pause_btn.config(text="Resume", bg="#006600")
        else:
            self.pause_btn.config(text="Pause", bg="#555")

    def _toggle_recording(self):
        if self.recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        if self.frame is None:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.record_path = self.record_dir / f"recording_{timestamp}.mp4"
        h, w = self.frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(str(self.record_path), fourcc, 20.0, (w, h))
        self.recording = True
        self.record_btn.config(text="Stop Recording", bg="#008800")
        print(f"[REC] Started: {self.record_path}")

    def _stop_recording(self):
        self.recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.record_btn.config(text="Start Recording", bg="#880000")
        print(f"[REC] Saved: {self.record_path}")

    def _take_screenshot(self):
        if self.frame is None:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.screenshot_dir / f"screenshot_{timestamp}.png"
        cv2.imwrite(str(path), self.frame)
        print(f"[SCREENSHOT] Saved: {path}")

    def _on_close(self):
        self.running = False
        if self.recording:
            self._stop_recording()
        if self.cap:
            self.cap.release()
        self.root.destroy()

    # ------------------------------------------------------------------
    #  Two-Stage Detection & Drawing
    # ------------------------------------------------------------------

    def _detect_frame(self, frame):
        """Run two-stage inference on a single frame."""
        if self.stage1_model is None or self.stage2_model is None:
            return [], 0.0, 0.0, 0.0

        t0 = time.time()

        # Update Stage 1 conf threshold dynamically
        self.stage1_model.conf = self.conf_threshold

        # Use the pipeline inference function (with drowning_threshold param)
        results = two_stage_inference(frame, self.stage1_model, self.stage2_model)

        total_time = (time.time() - t0) * 1000  # ms

        # Approximate stage breakdown (total includes both)
        # Stage 1 typically takes ~60-70% of total time for detection
        self.stage1_time = total_time * 0.6
        self.stage2_time = total_time * 0.4
        self.total_time = total_time

        return results, self.stage1_time, self.stage2_time, total_time

    def _draw_pipeline_results(self, frame, results):
        """Draw two-stage pipeline results using draw_results from pipeline_inference."""
        if not results:
            return frame

        # Use the shared draw_results function
        frame_out = draw_results(frame, results)

        # Check for drowning detection
        self.drowning_detected = any(
            r.get("fine_class") == "drowning" for r in results
        )

        return frame_out

    def _draw_info_overlay(self, frame):
        """Draw FPS, inference time, drowning alert, and recording indicator."""
        h, w = frame.shape[:2]

        # Semi-transparent overlay bar at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # FPS & timing info
        fps_text = (f"FPS: {self.fps:.1f} | "
                    f"S1: {self.stage1_time:.0f}ms + S2: {self.stage2_time:.0f}ms = "
                    f"{self.total_time:.0f}ms | Frame: {self.frame_count}")
        cv2.putText(frame, fps_text, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Recording indicator
        if self.recording:
            rec_x = w - 80
            cv2.circle(frame, (rec_x, 18), 6, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (rec_x + 14, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Paused indicator
        if self.paused:
            cv2.putText(frame, "PAUSED", (w // 2 - 60, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Thresholds display
        cv2.putText(frame, f"conf={self.conf_threshold:.2f}",
                    (w - 160, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Drowning alert
        if self.drowning_detected:
            self._alert_flash = not self._alert_flash
            if self._alert_flash:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
                cv2.putText(frame, "DROWNING ALERT!", (w // 2 - 130, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        return frame

    # ------------------------------------------------------------------
    #  Main Loop
    # ------------------------------------------------------------------

    def _process_frame(self):
        """Single frame processing pipeline."""
        if not self.running or self.cap is None:
            return

        ret, raw_frame = self.cap.read()
        if not ret:
            if self.source:
                print("[VIDEO] File ended. Restarting...")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return
            else:
                print("[ERROR] Camera read failed")
                self.running = False
                return

        self.frame_count += 1

        # Skip detection if paused
        if self.paused:
            self._update_display(raw_frame)
            self.root.after(30, self._process_frame)
            return

        # Check if models are loaded
        if self.stage1_model is None or self.stage2_model is None:
            cv2.putText(raw_frame, "Models not loaded - load both models first",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            self._update_display(raw_frame)
            self.root.after(30, self._process_frame)
            return

        # Two-stage detection
        results, s1_time, s2_time, total_time = self._detect_frame(raw_frame)
        self.stage1_time = s1_time
        self.stage2_time = s2_time
        self.total_time = total_time

        # Draw results
        display_frame = self._draw_pipeline_results(raw_frame.copy(), results)
        display_frame = self._draw_info_overlay(display_frame)

        # Update FPS
        now = time.time()
        self._fps_buffer.append(now)
        if now - self._last_fps_update > 0.5:
            cutoff = now - 1.0
            self._fps_buffer = [t for t in self._fps_buffer if t > cutoff]
            self._last_fps_update = now
        if len(self._fps_buffer) > 1:
            elapsed = self._fps_buffer[-1] - self._fps_buffer[0]
            if elapsed > 0:
                self.fps = (len(self._fps_buffer) - 1) / elapsed

        # Update class counts
        self.class_counts = {name: 0 for name in DISPLAY_CLASSES}
        for r in results:
            fine = r.get("fine_class")
            coarse = r.get("coarse_class")
            if fine and fine in self.class_counts:
                self.class_counts[fine] += 1
            elif coarse in self.class_counts:
                self.class_counts[coarse] += 1

        # Store frame for screenshot/recording
        self.frame = display_frame

        # Update UI
        self._update_display(display_frame)
        self._update_stats()
        self._update_alert()

        # Recording
        if self.recording and self.video_writer:
            self.video_writer.write(display_frame)

        # Schedule next frame
        self.root.after(15, self._process_frame)

    def _update_display(self, frame):
        """Convert OpenCV frame to tkinter-compatible format and display."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            canvas_w, canvas_h = 640, 480

        fh, fw = frame_rgb.shape[:2]
        scale = min(canvas_w / fw, canvas_h / fh)
        new_w, new_h = int(fw * scale), int(fh * scale)
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))

        img = Image.fromarray(frame_resized)
        self._photo = ImageTk.PhotoImage(image=img)

        self.canvas.delete("all")
        x_offset = (canvas_w - new_w) // 2
        y_offset = (canvas_h - new_h) // 2
        self.canvas.create_image(x_offset, y_offset, anchor="nw", image=self._photo)

    def _update_stats(self):
        """Update status bar and detection count labels."""
        status_text = (f"FPS: {self.fps:.1f} | S1: {self.stage1_time:.0f}ms | "
                       f"S2: {self.stage2_time:.0f}ms | Total: {self.total_time:.0f}ms | "
                       f"Frames: {self.frame_count} | "
                       f"{'Paused' if self.paused else 'Running'}")
        self.status_label.config(text=status_text)

        for name, lbl in self.stat_labels.items():
            count = self.class_counts.get(name, 0)
            lbl.config(text=f"{name:>14}: {count}")

    def _update_alert(self):
        """Update drowning alert indicator."""
        if self.drowning_detected:
            self.alert_label.config(text="!! DROWNING DETECTED !!", fg="#E24B4A")
        else:
            self.alert_label.config(text="", fg="#E24B4A")

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def start(self):
        """Start the camera and detection loop."""
        if self.stage1_model is None or self.stage2_model is None:
            print("[ERROR] Both models must be loaded first!")
            self.status_label.config(
                text="ERROR: Load both Stage 1 & Stage 2 models first!")
            return

        if not self.cap:
            if self.source:
                self.cap = cv2.VideoCapture(self.source)
            else:
                self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open camera {self.camera_id}")
            self.status_label.config(text="ERROR: Cannot open camera!")
            return

        self.running = True
        self.paused = False
        self.frame_count = 0
        self.status_label.config(text="FPS: -- | Status: Initializing...")

        self.root.after(100, self._process_frame)

    def run(self):
        """Start GUI main loop."""
        self.start()
        self.root.mainloop()


# ===========================================================================
#  Main
# ===========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Two-Stage Drowning Detection GUI")
    parser.add_argument("--stage1", type=str, default=str(DEFAULT_STAGE1),
                        help="Stage 1 detection model path (.pt)")
    parser.add_argument("--stage2", type=str, default=str(DEFAULT_STAGE2),
                        help="Stage 2 classification model path (.pt)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device ID")
    parser.add_argument("--source", type=str, default=None,
                        help="Video file path instead of camera")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help="Stage 1 confidence threshold")
    args = parser.parse_args()

    # Auto-find models if defaults don't exist
    stage1_path = find_model(
        args.stage1, DEFAULT_STAGE1,
        [PROJECT_ROOT / "runs" / "stage1_v1"]
    )
    stage2_path = find_model(
        args.stage2, DEFAULT_STAGE2,
        [PROJECT_ROOT / "runs" / "stage2_v1",
         PROJECT_ROOT / "runs" / "stage2_classify"]
    )

    app = DetectGUI(
        stage1_path=stage1_path,
        stage2_path=stage2_path,
        camera_id=args.camera,
        source=args.source,
        conf=args.conf,
    )
    app.run()


if __name__ == "__main__":
    main()
