"""
detect_media_gui.py - Two-Stage Media Detection GUI
=====================================================
Photo & video detection using two-stage YOLO26 pipeline:
  Stage 1 (detect, nc=2): person_in_water / person(岸上)
  Stage 2 (classify, nc=2): drowning / swimming

Features:
  - Photo Mode: single image + batch folder detection, zoom/scroll, thumbnails
  - Video Mode: playback controls, progress slider, speed control, save output
  - Two-stage pipeline integration with drowning safety threshold
  - Drowning visual alert (red border + flashing indicator)
  - Model switching for both Stage 1 and Stage 2

Tech Stack: tkinter + OpenCV + PIL + ultralytics.YOLO

Usage:
    python detect_media_gui.py                                    # Auto-find models
    python detect_media_gui.py --stage1 path/to/stage1_best.pt   # Specific Stage 1 model
    python detect_media_gui.py --stage2 path/to/stage2_best.pt   # Specific Stage 2 model
"""

import os
import sys
import time
import importlib.util
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageTk

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---- 双版本推理 pipeline 导入 (使用 importlib 避免同名模块冲突) ----
# 标准版 (无时序过滤)
_spec_std = importlib.util.spec_from_file_location(
    "pipeline_std", str(PROJECT_ROOT / "pipeline_inference.py"))
_pipeline_std = importlib.util.module_from_spec(_spec_std)
_spec_std.loader.exec_module(_pipeline_std)

# 时序增强版 (含 DrowningTracker)
_spec_tmp = importlib.util.spec_from_file_location(
    "pipeline_temporal", str(PROJECT_ROOT / "pipeline_infrence" / "pipeline_inference.py"))
_pipeline_temporal = importlib.util.module_from_spec(_spec_tmp)
_spec_tmp.loader.exec_module(_pipeline_temporal)

# 共用常量 (两版一致)
from pipeline_inference import (
    STAGE1_CLASS_NAMES,
    STAGE2_CLASS_NAMES,
    ROUTE_CONF,
    DROWNING_CONFIRM,
    MIN_CLASS_CONF,
    crop_person_in_water,
)

# 推理模块映射 (运行时按 pipeline_mode 切换)
_PIPELINE_MODULES = {
    "standard": _pipeline_std,
    "temporal": _pipeline_temporal,
}


# ===========================================================================
#  Constants
# ===========================================================================

# Combined display categories (for stats panel)
# 用户关心的三大类: 岸上的人 / 游泳的人 / 溺水的人
DISPLAY_CLASSES = [
    "person",           # Stage1 class 1: 岸上人(安全)
    "swimming",         # Stage2: 游泳(正常)
    "drowning",         # Stage2: 溺水(确认红色告警)
    "drowning_possible", # Stage2: 疑似溺水(橙色预警)
    "person_in_water",  # Stage1 class 0: 水中人(未细分/低置信)
]

# Color map for display (BGR for OpenCV)
DISPLAY_BGR = {
    "drowning":                (0, 0, 255),       # BGR Red - confirmed drowning
    "drowning_possible":       (0, 165, 255),     # BGR Orange - warning level
    "swimming":                (0, 200, 0),       # BGR Green - normal swimming
    "person_in_water(未分类)":  (0, 215, 255),       # BGR Gold - 中性: 低置信/存疑, 非告警
    "person_in_water":         (255, 200, 0),       # BGR Cyan - 水中人粗粒度
    "person":                  (255, 0, 0),         # BGR Blue - 岸上人(安全)
}

# Default model paths (nc=2 Stage1 + nc=2 Stage2, v2)
DEFAULT_STAGE1 = (PROJECT_ROOT / "runs" /
                   "yolo26s_surveil_stage1_v2" / "best.pt")
DEFAULT_STAGE2 = (PROJECT_ROOT / "runs" /
                   "yolo26s_cls_surveil_stage2_v2" / "best.pt")

# Thresholds
DEFAULT_CONF = 0.5
DEFAULT_DROWNING_THRESHOLD = 0.5

# Window
WINDOW_TITLE = "Two-Stage Drowning Detection - Media"
DEFAULT_WIDTH = 1400
DEFAULT_HEIGHT = 800

# Output directories
OUTPUT_BASE = PROJECT_ROOT / "runs" / "media_detect"
PHOTO_SINGLE_DIR = OUTPUT_BASE / "photo_single"
PHOTO_BATCH_DIR = OUTPUT_BASE / "photo_batch"
VIDEO_OUTPUT_DIR = OUTPUT_BASE / "video"

# Supported image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Dark theme colors
BG_DARK = "#1e1e1e"
BG_PANEL = "#2d2d2d"
BG_STATUS = "#1a1a1a"
FG_LIGHT = "#e0e0e0"
FG_GREEN = "#00ff00"
FG_RED = "#E24B4A"
FG_YELLOW = "#ffff00"


# ===========================================================================
#  Model Finder
# ===========================================================================

def find_model(provided_path=None, default_path=None, search_dirs=None):
    """Find a trained model file with fallback search (newest match wins)."""
    if provided_path and Path(provided_path).exists():
        return Path(provided_path)

    if default_path and default_path.exists():
        return default_path

    candidates = []
    if search_dirs:
        for search_dir in search_dirs:
            search_dir = Path(search_dir)
            if not search_dir.exists():
                continue
            # 情况1: search_dir 自身就是 weights 目录
            if (search_dir / "best.pt").exists():
                candidates.append(search_dir / "best.pt")
                continue
            # 情况2: 遍历其子目录 (各次训练 run)
            for train_dir in sorted(search_dir.iterdir(), reverse=True):
                if not train_dir.is_dir():
                    continue
                for cand in [train_dir / "weights" / "best.pt",
                             train_dir / "best.pt"]:
                    if cand.exists():
                        candidates.append(cand)

    if candidates:
        # 取修改时间最新者（通常是最新一次训练产物）
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


# ===========================================================================
#  Media Detection GUI
# ===========================================================================

class MediaDetectGUI:
    """Main GUI application for two-stage photo/video drowning detection."""

    def __init__(self, stage1_path=None, stage2_path=None,
                 conf=DEFAULT_CONF, drowning_threshold=DEFAULT_DROWNING_THRESHOLD,
                 route_conf=ROUTE_CONF, drowning_confirm=DROWNING_CONFIRM,
                 min_class_conf=MIN_CLASS_CONF):
        # ---- Models ----
        self.stage1_path = stage1_path
        self.stage2_path = stage2_path
        self.stage1_model = None
        self.stage2_model = None

        # ---- Pipeline mode ----
        # "standard" = 单帧三分支判定 | "temporal" = 30帧滑窗投票 (DrowningTracker)
        self.pipeline_mode = "standard"
        self._tracker = None  # 时序跟踪器实例 (temporal 模式, 视频/摄像头场景)
        self._video_frame_idx = 0  # 当前视频帧序号

        # ---- Photo state ----
        self.photo_images = []          # list of image file paths
        self.photo_current_idx = -1     # currently displayed image index
        self.photo_results = {}         # {path: inference_results}
        self.photo_annotated = {}       # {path: annotated_frame ndarray}
        self.photo_originals = {}       # {path: original_frame ndarray}
        self.photo_zoom_level = 1.0
        self.photo_pan_offset = (0, 0)
        self._photo_drag_start = None

        # ---- Video state ----
        self.video_cap = None
        self.video_path = None
        self.video_running = False
        self.video_paused = False
        self.video_total_frames = 0
        self.video_current_frame = 0
        self.video_fps = 30.0
        self.video_writer = None
        self.video_recording = False
        self.video_record_path = None
        self.video_speed = 1            # frame skip = 0 (1x), 1 (2x), etc.
        self._video_seek_debounce = None

        # ---- Detection settings ----
        self.conf_threshold = conf
        self.drowning_threshold = drowning_threshold
        self.route_conf = route_conf
        self.drowning_confirm = drowning_confirm
        self.min_class_conf = min_class_conf
        self.drowning_detected = False
        self.drowning_possible_detected = False
        self._alert_flash = False

        # ---- Statistics ----
        self.class_counts = {name: 0 for name in DISPLAY_CLASSES}
        self._detect_time = 0.0
        self._s1_time = 0.0
        self._s2_time = 0.0

        # ---- Batch progress ----
        self._batch_running = False

        # ---- Create output dirs ----
        for d in [PHOTO_SINGLE_DIR, PHOTO_BATCH_DIR, VIDEO_OUTPUT_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        # ---- Build UI ----
        self._setup_ui()

        # Load models: Stage 1 必需, Stage 2 可选
        if self.stage1_path:
            self._load_models()

    # ------------------------------------------------------------------
    #  Model Management
    # ------------------------------------------------------------------

    def _load_models(self):
        """Load both Stage 1 and Stage 2 models."""
        from ultralytics import YOLO

        errors = []

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
            # Stage 2 未加载 → Stage1-only 模式（仅检测，不做 drowning/swimming 细分）
            self.stage2_model = None
            print("[Stage2] 未加载 → Stage1-only 模式")

        self._update_model_labels()

        if errors:
            from tkinter import messagebox
            messagebox.showwarning(
                "Model Load Warning",
                "Some models failed to load:\n\n" + "\n".join(errors)
            )
        elif self.stage2_model is None:
            self._update_status(
                "Stage1-only 模式：Stage 2 未加载，不做 drowning/swimming 细分")
        else:
            self._update_status("两阶段模型已就绪，可开始检测")

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

    def _remove_stage1(self):
        """Unload Stage 1 model from memory (keeps file on disk)."""
        self.stage1_model = None
        self.stage1_path = None  # 避免下次自动重载
        self._update_model_labels()
        self._update_status("Stage 1 已移除，请重新 Load 模型")
        print("[Stage1] Model removed from memory")

    def _remove_stage2(self):
        """Unload Stage 2 model from memory (keeps file on disk)."""
        self.stage2_model = None
        self.stage2_path = None
        self._update_model_labels()
        if self.stage1_model is not None:
            self._update_status("Stage1-only 模式：Stage 2 已移除")
        else:
            self._update_status("Stage 2 已移除")
        print("[Stage2] Model removed from memory")

    def _update_model_labels(self):
        import tkinter as tk
        if hasattr(self, 's1_label'):
            name = Path(self.stage1_path).name if self.stage1_path else "(未加载)"
            self.s1_label.config(text=name)
        if hasattr(self, 's2_label'):
            name = Path(self.stage2_path).name if self.stage2_path else "(未加载)"
            self.s2_label.config(text=name)
        if hasattr(self, 'root'):
            self.root.update_idletasks()

    # ------------------------------------------------------------------
    #  UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        import tkinter as tk
        from tkinter import ttk

        self.root = tk.Tk()
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}")
        self.root.configure(bg=BG_DARK)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ---- Main layout ----
        self.main_frame = tk.Frame(self.root, bg=BG_DARK)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left: content area (Notebook + canvas)
        self.content_frame = tk.Frame(self.main_frame, bg=BG_DARK)
        self.content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Notebook (Photo / Video tabs)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Dark.TNotebook', background=BG_DARK)
        style.configure('Dark.TNotebook.Tab', background=BG_PANEL,
                        foreground=FG_LIGHT, padding=[12, 4])
        style.map('Dark.TNotebook.Tab',
                  background=[('selected', '#378ADD')],
                  foreground=[('selected', '#ffffff')])

        self.notebook = ttk.Notebook(self.content_frame, style='Dark.TNotebook')
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Photo tab
        self.photo_tab = tk.Frame(self.notebook, bg=BG_DARK)
        self.notebook.add(self.photo_tab, text="  Photo Mode  ")
        self._setup_photo_tab()

        # Video tab
        self.video_tab = tk.Frame(self.notebook, bg=BG_DARK)
        self.notebook.add(self.video_tab, text="  Video Mode  ")
        self._setup_video_tab()

        # Right: control panel
        self.control_frame = tk.Frame(self.main_frame, bg=BG_PANEL, width=300)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        self.control_frame.pack_propagate(False)
        self._build_controls()

        # Bottom: status bar
        self.status_frame = tk.Frame(self.root, bg=BG_STATUS, height=30)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_frame.pack_propagate(False)

        self.status_label = tk.Label(
            self.status_frame,
            text="Status: 加载 Stage 1 模型后开始检测 (Stage 2 可选)",
            bg=BG_STATUS, fg=FG_YELLOW, font=("Consolas", 10), anchor="w"
        )
        self.status_label.pack(fill=tk.X, padx=10, pady=3)

        # Notebook tab change callback
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _setup_photo_tab(self):
        import tkinter as tk

        # ---- Photo display area ----
        display_frame = tk.Frame(self.photo_tab, bg="#000000", relief=tk.SUNKEN, bd=1)
        display_frame.pack(fill=tk.BOTH, expand=True)

        self.photo_canvas = tk.Canvas(display_frame, bg="#000000", highlightthickness=0)
        self.photo_canvas.pack(fill=tk.BOTH, expand=True)

        # Zoom / scroll bindings
        self.photo_canvas.bind("<MouseWheel>", self._on_photo_scroll)
        self.photo_canvas.bind("<Button-4>", self._on_photo_scroll)  # Linux scroll up
        self.photo_canvas.bind("<Button-5>", self._on_photo_scroll)  # Linux scroll down
        self.photo_canvas.bind("<ButtonPress-1>", self._on_photo_drag_start)
        self.photo_canvas.bind("<B1-Motion>", self._on_photo_drag)
        self.photo_canvas.bind("<ButtonRelease-1>", self._on_photo_drag_end)

        # ---- Thumbnails bar ----
        thumb_frame = tk.Frame(self.photo_tab, bg=BG_PANEL, height=80)
        thumb_frame.pack(fill=tk.X, pady=(2, 0))
        thumb_frame.pack_propagate(False)

        # Navigation buttons on left
        nav_frame = tk.Frame(thumb_frame, bg=BG_PANEL)
        nav_frame.pack(side=tk.LEFT, padx=5)

        tk.Button(nav_frame, text="< Prev", command=self._prev_photo,
                  bg="#444", fg=FG_LIGHT, font=("Consolas", 9)).pack(side=tk.LEFT, padx=2)
        tk.Button(nav_frame, text="Next >", command=self._next_photo,
                  bg="#444", fg=FG_LIGHT, font=("Consolas", 9)).pack(side=tk.LEFT, padx=2)

        self.photo_info_label = tk.Label(nav_frame, text="No image loaded",
                                          bg=BG_PANEL, fg=FG_YELLOW, font=("Consolas", 9))
        self.photo_info_label.pack(side=tk.LEFT, padx=10)

        # Scrollable thumbnail canvas
        thumb_container = tk.Frame(thumb_frame, bg=BG_PANEL)
        thumb_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.thumb_canvas = tk.Canvas(thumb_container, bg=BG_PANEL, height=70,
                                       highlightthickness=0)
        self.thumb_scrollbar = tk.Scrollbar(thumb_container, orient=tk.HORIZONTAL,
                                             command=self.thumb_canvas.xview)
        self.thumb_canvas.configure(xscrollcommand=self.thumb_scrollbar.set)
        self.thumb_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.thumb_canvas.pack(fill=tk.BOTH, expand=True)

        self.thumb_inner = tk.Frame(self.thumb_canvas, bg=BG_PANEL)
        self.thumb_canvas.create_window((0, 0), window=self.thumb_inner, anchor="nw")
        self.thumb_inner.bind("<Configure>",
            lambda e: self.thumb_canvas.configure(scrollregion=self.thumb_canvas.bbox("all")))

        # ---- Photo action buttons ----
        btn_frame = tk.Frame(self.photo_tab, bg=BG_DARK)
        btn_frame.pack(fill=tk.X, pady=5, padx=5)

        buttons = [
            ("Load Photo", self._load_photo, "#185FA5"),
            ("Load Folder", self._load_photo_folder, "#534AB7"),
            ("Detect", self._detect_current_photo, "#006600"),
            ("Detect All", self._detect_all_photos, "#008800"),
            ("Save", self._save_photo_result, "#8B4513"),
            ("Save All", self._save_all_results, "#A0522D"),
            ("Reset Zoom", self._reset_photo_zoom, "#444"),
        ]
        for text, cmd, color in buttons:
            tk.Button(btn_frame, text=text, command=cmd,
                      bg=color, fg="#fff", font=("Consolas", 9),
                      relief=tk.RAISED, padx=8, pady=2).pack(side=tk.LEFT, padx=3)

        # ---- Batch progress bar ----
        self.photo_progress_frame = tk.Frame(self.photo_tab, bg=BG_DARK)
        self.photo_progress_frame.pack(fill=tk.X, padx=5)

        self.photo_progress_label = tk.Label(self.photo_progress_frame, text="",
                                              bg=BG_DARK, fg=FG_LIGHT, font=("Consolas", 9))
        self.photo_progress_label.pack(side=tk.LEFT)

    def _setup_video_tab(self):
        import tkinter as tk
        from tkinter import ttk

        # ---- Video display area ----
        display_frame = tk.Frame(self.video_tab, bg="#000000", relief=tk.SUNKEN, bd=1)
        display_frame.pack(fill=tk.BOTH, expand=True)

        self.video_canvas = tk.Canvas(display_frame, bg="#000000", highlightthickness=0)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        # ---- Video info label ----
        info_frame = tk.Frame(self.video_tab, bg=BG_PANEL)
        info_frame.pack(fill=tk.X, pady=2, padx=5)

        self.video_info_label = tk.Label(info_frame, text="No video loaded",
                                          bg=BG_PANEL, fg=FG_YELLOW, font=("Consolas", 9))
        self.video_info_label.pack(side=tk.LEFT)

        self.video_frame_label = tk.Label(info_frame, text="Frame: --/--",
                                           bg=BG_PANEL, fg=FG_LIGHT, font=("Consolas", 9))
        self.video_frame_label.pack(side=tk.RIGHT)

        # ---- Progress slider ----
        self.video_progress_var = tk.IntVar(value=0)
        self.video_progress_slider = tk.Scale(
            self.video_tab, from_=0, to=1, orient=tk.HORIZONTAL,
            variable=self.video_progress_var,
            bg=BG_PANEL, fg=FG_LIGHT, troughcolor="#444",
            highlightthickness=0, showvalue=False,
            command=self._on_video_seek
        )
        self.video_progress_slider.pack(fill=tk.X, padx=5)

        # ---- Video control buttons ----
        ctrl_frame = tk.Frame(self.video_tab, bg=BG_DARK)
        ctrl_frame.pack(fill=tk.X, pady=5, padx=5)

        buttons_left = [
            ("Load Video", self._load_video, "#185FA5"),
            ("Play", self._play_video, "#006600"),
            ("Pause", self._pause_video, "#555"),
            ("Stop", self._stop_video, "#880000"),
        ]
        for text, cmd, color in buttons_left:
            tk.Button(ctrl_frame, text=text, command=cmd,
                      bg=color, fg="#fff", font=("Consolas", 9),
                      relief=tk.RAISED, padx=8, pady=2).pack(side=tk.LEFT, padx=3)

        ttk.Separator(ctrl_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Save controls
        self.video_save_btn = tk.Button(ctrl_frame, text="Start Save",
                                         command=self._toggle_video_save,
                                         bg="#8B4513", fg="#fff", font=("Consolas", 9),
                                         relief=tk.RAISED, padx=8, pady=2)
        self.video_save_btn.pack(side=tk.LEFT, padx=3)

        ttk.Separator(ctrl_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Speed control
        tk.Label(ctrl_frame, text="Speed:", bg=BG_DARK, fg=FG_LIGHT,
                 font=("Consolas", 9)).pack(side=tk.LEFT, padx=2)

        self.speed_var = tk.IntVar(value=0)
        speed_options = [("1x", 0), ("2x", 1), ("3x", 2), ("5x", 4)]
        for label, val in speed_options:
            tk.Radiobutton(ctrl_frame, text=label, variable=self.speed_var, value=val,
                           command=self._on_speed_change,
                           bg=BG_DARK, fg=FG_LIGHT, selectcolor=BG_PANEL,
                           activebackground=BG_DARK, activeforeground=FG_GREEN,
                           font=("Consolas", 9)).pack(side=tk.LEFT, padx=2)

    def _build_controls(self):
        import tkinter as tk
        from tkinter import ttk

        pad = {"padx": 8, "pady": 3}
        bg = BG_PANEL

        # ---- Title ----
        tk.Label(self.control_frame, text="Two-Stage Pipeline",
                 bg=bg, fg=FG_GREEN, font=("Consolas", 13, "bold")).pack(fill=tk.X, **pad)

        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, **pad)

        # ---- Stage 1 Model ----
        s1_group = tk.LabelFrame(self.control_frame, text="Stage 1 (Detect nc=2: person_in_water / person)",
                                  bg=bg, fg="#378ADD", font=("Consolas", 10, "bold"))
        s1_group.pack(fill=tk.X, **pad)

        s1_name = Path(self.stage1_path).name if self.stage1_path else "(未加载)"
        self.s1_label = tk.Label(s1_group, text=s1_name,
                                  bg=bg, fg="#85B7EB", font=("Consolas", 9),
                                  wraplength=260)
        self.s1_label.pack(fill=tk.X, **pad)

        s1_btn_row = tk.Frame(s1_group, bg=bg)
        s1_btn_row.pack(fill=tk.X, **pad)
        tk.Button(s1_btn_row, text="Load Stage1", command=self._browse_stage1,
                  bg="#185FA5", fg="#fff", font=("Consolas", 9)).pack(
                      side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        tk.Button(s1_btn_row, text="Remove", command=self._remove_stage1,
                  bg="#8B0000", fg="#fff", font=("Consolas", 9)).pack(
                      side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        # ---- Stage 2 Model ----
        s2_group = tk.LabelFrame(self.control_frame, text="Stage 2 (Classify nc=2)",
                                  bg=bg, fg="#7F77DD", font=("Consolas", 10, "bold"))
        s2_group.pack(fill=tk.X, **pad)

        s2_name = Path(self.stage2_path).name if self.stage2_path else "(未加载)"
        self.s2_label = tk.Label(s2_group, text=s2_name,
                                  bg=bg, fg="#AFA9EC", font=("Consolas", 9),
                                  wraplength=260)
        self.s2_label.pack(fill=tk.X, **pad)

        s2_btn_row = tk.Frame(s2_group, bg=bg)
        s2_btn_row.pack(fill=tk.X, **pad)
        tk.Button(s2_btn_row, text="Load Stage2", command=self._browse_stage2,
                  bg="#534AB7", fg="#fff", font=("Consolas", 9)).pack(
                      side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        tk.Button(s2_btn_row, text="Remove", command=self._remove_stage2,
                  bg="#8B0000", fg="#fff", font=("Consolas", 9)).pack(
                      side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, **pad)

        # ---- Pipeline Mode ----
        mode_group = tk.LabelFrame(self.control_frame, text="Pipeline Mode",
                                   bg=bg, fg="#FFD700", font=("Consolas", 10, "bold"))
        mode_group.pack(fill=tk.X, **pad)

        self.mode_var = tk.StringVar(value=self.pipeline_mode)
        mode_standard = tk.Radiobutton(mode_group, text="标准模式 (单帧判定)",
                                       variable=self.mode_var, value="standard",
                                       command=self._on_mode_change,
                                       bg=bg, fg=FG_LIGHT, font=("Consolas", 9),
                                       selectcolor=BG_DARK, activebackground=bg, activeforeground="#FFF")
        mode_standard.pack(anchor=tk.W, padx=12, pady=1)
        mode_temporal = tk.Radiobutton(mode_group, text="时序模式 (30帧投票, 抑制误报)",
                                       variable=self.mode_var, value="temporal",
                                       command=self._on_mode_change,
                                       bg=bg, fg=FG_LIGHT, font=("Consolas", 9),
                                       selectcolor=BG_DARK, activebackground=bg, activeforeground="#FFF")
        mode_temporal.pack(anchor=tk.W, padx=12, pady=1)

        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, **pad)

        # ---- Confidence Threshold ----
        conf_group = tk.LabelFrame(self.control_frame, text="Stage1 Conf Threshold",
                                   bg=bg, fg=FG_LIGHT, font=("Consolas", 10, "bold"))
        conf_group.pack(fill=tk.X, **pad)

        self.conf_var = tk.DoubleVar(value=DEFAULT_CONF)
        tk.Scale(conf_group, from_=0.1, to=0.9, resolution=0.05,
                 orient=tk.HORIZONTAL, variable=self.conf_var,
                 bg=bg, fg=FG_LIGHT, troughcolor="#444", highlightthickness=0,
                 command=self._on_conf_change).pack(fill=tk.X, **pad)

        self.conf_value_label = tk.Label(conf_group, text=f"{DEFAULT_CONF:.2f}",
                                         bg=bg, fg=FG_YELLOW, font=("Consolas", 11, "bold"))
        self.conf_value_label.pack()

        # ---- Drowning Warning Threshold ----
        drown_group = tk.LabelFrame(self.control_frame, text="Drowning Warning Threshold",
                                    bg=bg, fg="#FFA500", font=("Consolas", 10, "bold"))
        drown_group.pack(fill=tk.X, **pad)

        self.drown_var = tk.DoubleVar(value=DEFAULT_DROWNING_THRESHOLD)
        tk.Scale(drown_group, from_=0.1, to=0.9, resolution=0.05,
                 orient=tk.HORIZONTAL, variable=self.drown_var,
                 bg=bg, fg=FG_LIGHT, troughcolor="#444", highlightthickness=0,
                 command=self._on_drown_change).pack(fill=tk.X, **pad)

        self.drown_value_label = tk.Label(drown_group, text=f"{DEFAULT_DROWNING_THRESHOLD:.2f}",
                                          bg=bg, fg="#F09595", font=("Consolas", 11, "bold"))
        self.drown_value_label.pack()

        # ---- Route Conf (Stage1→Stage2 最低置信度闸门) ----
        # 低于此置信度的 Stage1 框不再送 Stage2, 直接标中性,
        # 从源头砍掉低质杂物误检 (缓解"杂物→溺水"误报的主闸门)
        route_group = tk.LabelFrame(self.control_frame, text="Route Conf (→Stage2 gate)",
                                   bg=bg, fg="#5DCAA5", font=("Consolas", 10, "bold"))
        route_group.pack(fill=tk.X, **pad)

        self.route_var = tk.DoubleVar(value=ROUTE_CONF)
        tk.Scale(route_group, from_=0.1, to=0.7, resolution=0.05,
                 orient=tk.HORIZONTAL, variable=self.route_var,
                 bg=bg, fg=FG_LIGHT, troughcolor="#444", highlightthickness=0,
                 command=self._on_route_change).pack(fill=tk.X, **pad)

        self.route_value_label = tk.Label(route_group, text=f"{ROUTE_CONF:.2f}",
                                          bg=bg, fg="#5DCAA5", font=("Consolas", 11, "bold"))
        self.route_value_label.pack()

        # ---- Drowning Confirm Threshold (红框确认阈值) ----
        # drowning_conf 需 >= 此值 且 > swimming_conf 才报红色确认溺水,
        # 抬高以压制杂物/边界 case 的误报红框
        confirm_group = tk.LabelFrame(self.control_frame, text="Drown Confirm (red gate)",
                                     bg=bg, fg="#F09595", font=("Consolas", 10, "bold"))
        confirm_group.pack(fill=tk.X, **pad)

        self.confirm_var = tk.DoubleVar(value=DROWNING_CONFIRM)
        tk.Scale(confirm_group, from_=0.5, to=0.9, resolution=0.05,
                 orient=tk.HORIZONTAL, variable=self.confirm_var,
                 bg=bg, fg=FG_LIGHT, troughcolor="#444", highlightthickness=0,
                 command=self._on_confirm_change).pack(fill=tk.X, **pad)

        self.confirm_value_label = tk.Label(confirm_group, text=f"{DROWNING_CONFIRM:.2f}",
                                            bg=bg, fg="#F09595", font=("Consolas", 11, "bold"))
        self.confirm_value_label.pack()

        # ---- Min Class Conf (存疑阈值) ----
        # Stage2 两类最大概率 < 此值时视为"存疑/非人", 不触发告警,
        # 杂物 crop 常落入此区间 → 不再被强制判成 drowning/swimming
        minc_group = tk.LabelFrame(self.control_frame, text="Min Class Conf (uncertain)",
                                   bg=bg, fg="#88AAFF", font=("Consolas", 10, "bold"))
        minc_group.pack(fill=tk.X, **pad)

        self.minc_var = tk.DoubleVar(value=MIN_CLASS_CONF)
        tk.Scale(minc_group, from_=0.4, to=0.8, resolution=0.05,
                 orient=tk.HORIZONTAL, variable=self.minc_var,
                 bg=bg, fg=FG_LIGHT, troughcolor="#444", highlightthickness=0,
                 command=self._on_minc_change).pack(fill=tk.X, **pad)

        self.minc_value_label = tk.Label(minc_group, text=f"{MIN_CLASS_CONF:.2f}",
                                         bg=bg, fg="#88AAFF", font=("Consolas", 11, "bold"))
        self.minc_value_label.pack()

        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, **pad)

        # ---- Detection Stats ----
        stats_group = tk.LabelFrame(self.control_frame, text="Detection Count",
                                    bg=bg, fg=FG_LIGHT, font=("Consolas", 10, "bold"))
        stats_group.pack(fill=tk.X, **pad)

        self.stat_labels = {}
        for name in DISPLAY_CLASSES:
            color_fg = "#cccccc"
            if name == "drowning":
                color_fg = "#F09595"
            elif name == "drowning_possible":
                color_fg = "#FFA500"
            elif name == "swimming":
                color_fg = "#5DCAA5"
            lbl = tk.Label(stats_group, text=f"{name:>14}: 0",
                           bg=bg, fg=color_fg, font=("Consolas", 9), anchor="w")
            lbl.pack(fill=tk.X, padx=12)
            self.stat_labels[name] = lbl

        # ---- Drowning Alert Indicator ----
        self.alert_label = tk.Label(self.control_frame, text="",
                                    bg=bg, fg=FG_RED, font=("Consolas", 14, "bold"))
        self.alert_label.pack(fill=tk.X, **pad)

    # ------------------------------------------------------------------
    #  UI Callbacks
    # ------------------------------------------------------------------

    def _on_conf_change(self, val):
        self.conf_threshold = float(val)
        self.conf_value_label.config(text=f"{self.conf_threshold:.2f}")
        if self.stage1_model:
            self.stage1_model.conf = self.conf_threshold

    def _on_drown_change(self, val):
        self.drowning_threshold = float(val)
        self.drown_value_label.config(text=f"{self.drowning_threshold:.2f}")

    def _on_route_change(self, val):
        self.route_conf = float(val)
        self.route_value_label.config(text=f"{self.route_conf:.2f}")

    def _on_confirm_change(self, val):
        self.drowning_confirm = float(val)
        self.confirm_value_label.config(text=f"{self.drowning_confirm:.2f}")

    def _on_minc_change(self, val):
        self.min_class_conf = float(val)
        self.minc_value_label.config(text=f"{self.min_class_conf:.2f}")

    def _on_tab_changed(self, event):
        """Pause video when switching away from Video tab."""
        current_tab = self.notebook.index(self.notebook.select())
        if current_tab != 1:  # Not Video tab
            if self.video_running and not self.video_paused:
                self._pause_video()

    # ------------------------------------------------------------------
    #  Two-Stage Detection & Drawing
    # ------------------------------------------------------------------

    def _on_mode_change(self):
        """Pipeline mode switched: reset tracker and update state."""
        old_mode = self.pipeline_mode
        self.pipeline_mode = self.mode_var.get()
        if old_mode != self.pipeline_mode:
            self._tracker = None
            self._video_frame_idx = 0
            self._update_status(f"Pipeline mode: {'Temporal (30f vote)' if self.pipeline_mode == 'temporal' else 'Standard (single-frame)'}")

    def _get_or_create_tracker(self):
        """Get or lazily create DrowningTracker for temporal mode."""
        if self._tracker is None:
            self._tracker = _pipeline_temporal.DrowningTracker(
                window_size=30, alarm_ratio=0.6, stale_frame_threshold=60)
        return self._tracker

    def _detect_frame(self, frame):
        """Run detection using selected pipeline mode."""
        if self.stage1_model is None:
            return [], 0.0, 0.0, 0.0

        t0 = time.time()
        self.stage1_model.conf = self.conf_threshold

        pipe = _PIPELINE_MODULES[self.pipeline_mode]

        # 时序模式: 传 tracker + frame_index
        if self.pipeline_mode == "temporal" and self._tracker is not None:
            results = pipe.two_stage_inference(
                frame, self.stage1_model, self.stage2_model,
                tracker=self._tracker,
                frame_index=self._video_frame_idx,
                drowning_threshold=self.drowning_threshold,
                route_conf=self.route_conf,
                drowning_confirm=self.drowning_confirm,
                min_class_conf=self.min_class_conf)
        else:
            results = pipe.two_stage_inference(
                frame, self.stage1_model, self.stage2_model,
                drowning_threshold=self.drowning_threshold,
                route_conf=self.route_conf,
                drowning_confirm=self.drowning_confirm,
                min_class_conf=self.min_class_conf)

        total_time = (time.time() - t0) * 1000  # ms
        if self.stage2_model is not None:
            self._s1_time = total_time * 0.6
            self._s2_time = total_time * 0.4
        else:
            self._s1_time = total_time
            self._s2_time = 0.0
        self._detect_time = total_time

        return results, self._s1_time, self._s2_time, total_time

    def _draw_pipeline_results(self, frame, results):
        """Draw two-stage pipeline results."""
        if not results:
            return frame
        pipe = _PIPELINE_MODULES[self.pipeline_mode]
        frame_out = pipe.draw_results(frame, results)
        self.drowning_detected = any(
            r.get("fine_class") == "drowning" for r in results
        )
        self.drowning_possible_detected = any(
            r.get("fine_class") == "drowning_possible" for r in results
        )
        return frame_out

    def _update_class_counts(self, results):
        """Update class count statistics from detection results."""
        self.class_counts = {name: 0 for name in DISPLAY_CLASSES}
        for r in results:
            fine = r.get("fine_class")
            coarse = r.get("coarse_class")
            if fine and fine in self.class_counts:
                self.class_counts[fine] += 1
            elif coarse == "person_in_water" and fine and fine not in ("drowning", "swimming"):
                self.class_counts["person_in_water"] += 1
            elif coarse in self.class_counts:
                self.class_counts[coarse] += 1

    def _update_stats_display(self):
        """Update stats labels in control panel."""
        for name, lbl in self.stat_labels.items():
            count = self.class_counts.get(name, 0)
            lbl.config(text=f"{name:>14}: {count}")

    def _update_alert(self):
        """Update drowning alert indicator (graded)."""
        if self.drowning_detected:
            self.alert_label.config(text="!! DROWNING DETECTED !!", fg=FG_RED)
        elif self.drowning_possible_detected:
            self.alert_label.config(text="POSSIBLE DROWNING", fg="#FFA500")
        else:
            self.alert_label.config(text="", fg=FG_RED)

    def _update_status(self, text):
        """Update status bar text."""
        self.status_label.config(text=text)

    # ------------------------------------------------------------------
    #  Photo Mode
    # ------------------------------------------------------------------

    def _load_photo(self):
        """Load a single photo via file dialog."""
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select Photo",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tif"),
                       ("All files", "*.*")],
            initialdir=str(PROJECT_ROOT),
        )
        if path:
            self._add_photo(path)
            self.photo_current_idx = len(self.photo_images) - 1
            self._show_current_photo()
            self._build_thumbnails()

    def _load_photo_folder(self):
        """Load all images from a folder."""
        from tkinter import filedialog
        folder = filedialog.askdirectory(
            title="Select Photo Folder",
            initialdir=str(PROJECT_ROOT),
        )
        if folder:
            folder_path = Path(folder)
            images = sorted([
                str(f) for f in folder_path.iterdir()
                if f.suffix.lower() in IMAGE_EXTS
            ])
            if not images:
                from tkinter import messagebox
                messagebox.showinfo("No Images", f"No supported images found in:\n{folder}")
                return

            # Clear previous state
            self._clear_photo_state()
            for img_path in images:
                self._add_photo(img_path)

            self.photo_current_idx = 0
            self._show_current_photo()
            self._build_thumbnails()
            self._update_status(f"Loaded {len(images)} images from {folder}")

    def _add_photo(self, path):
        """Add a single photo to the list."""
        self.photo_images.append(path)
        img = cv2.imread(path)
        if img is not None:
            self.photo_originals[path] = img

    def _clear_photo_state(self):
        """Reset all photo state."""
        self.photo_images = []
        self.photo_current_idx = -1
        self.photo_results = {}
        self.photo_annotated = {}
        self.photo_originals = {}
        self.photo_zoom_level = 1.0
        self.photo_pan_offset = (0, 0)

    def _detect_current_photo(self):
        """Run detection on the currently displayed photo."""
        if self.photo_current_idx < 0 or not self.photo_images:
            from tkinter import messagebox
            messagebox.showinfo("No Image", "Please load a photo first.")
            return

        if self.stage1_model is None:
            from tkinter import messagebox
            messagebox.showwarning("Stage 1 Not Loaded",
                                   "Load Stage 1 detection model first.")
            return

        path = self.photo_images[self.photo_current_idx]
        original = self.photo_originals.get(path)
        if original is None:
            original = cv2.imread(path)
            if original is None:
                return
            self.photo_originals[path] = original

        results, s1_time, s2_time, total = self._detect_frame(original)
        annotated = self._draw_pipeline_results(original.copy(), results)

        self.photo_results[path] = results
        self.photo_annotated[path] = annotated

        # Update thumbnail border color
        self._update_thumbnail_border(self.photo_current_idx)

        self._update_class_counts(results)
        self._update_stats_display()
        self._update_alert()
        self._update_status(f"Detected: {path} | {total:.0f}ms | "
                            f"Objects: {len(results)}")

        # Show annotated result
        self.photo_zoom_level = 1.0
        self.photo_pan_offset = (0, 0)
        self._display_photo_on_canvas(annotated)

    def _detect_all_photos(self):
        """Batch detect all loaded photos."""
        if not self.photo_images:
            from tkinter import messagebox
            messagebox.showinfo("No Images", "Please load photos first.")
            return

        if self.stage1_model is None:
            from tkinter import messagebox
            messagebox.showwarning("Stage 1 Not Loaded",
                                   "Load Stage 1 detection model first.")
            return

        self._batch_running = True
        total = len(self.photo_images)

        for i, path in enumerate(self.photo_images):
            if not self._batch_running:
                break

            original = self.photo_originals.get(path)
            if original is None:
                original = cv2.imread(path)
                if original is None:
                    continue
                self.photo_originals[path] = original

            results, _, _, total_ms = self._detect_frame(original)
            annotated = self._draw_pipeline_results(original.copy(), results)

            self.photo_results[path] = results
            self.photo_annotated[path] = annotated

            # Update progress
            self.photo_progress_label.config(
                text=f"Detecting: {i+1}/{total} | {total_ms:.0f}ms")
            self._update_thumbnail_border(i)

            # Force UI update
            self.root.update_idletasks()

        # Show current photo result
        if self.photo_current_idx >= 0:
            self._show_current_photo()

        self.photo_progress_label.config(
            text=f"Done: {total} images processed")
        self._update_status(f"Batch detection complete: {total} images")

        self._batch_running = False

    def _show_current_photo(self):
        """Display the current photo (annotated if available, otherwise original)."""
        if self.photo_current_idx < 0 or not self.photo_images:
            return

        path = self.photo_images[self.photo_current_idx]

        # Prefer annotated result if available
        if path in self.photo_annotated:
            frame = self.photo_annotated[path]
        elif path in self.photo_originals:
            frame = self.photo_originals[path]
        else:
            frame = cv2.imread(path)
            if frame is None:
                return
            self.photo_originals[path] = frame

        # Reset zoom when switching photos
        self.photo_zoom_level = 1.0
        self.photo_pan_offset = (0, 0)

        self._display_photo_on_canvas(frame)

        # Update info
        idx = self.photo_current_idx
        total = len(self.photo_images)
        detected = "Yes" if path in self.photo_results else "No"
        n_objects = len(self.photo_results.get(path, []))
        self.photo_info_label.config(
            text=f"Photo {idx+1}/{total} | Detected: {detected} | Objects: {n_objects}")

        # Update stats for current photo
        results = self.photo_results.get(path, [])
        self._update_class_counts(results)
        self._update_stats_display()

    def _display_photo_on_canvas(self, frame):
        """Display a frame on the photo canvas with zoom and pan support."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        canvas_w = self.photo_canvas.winfo_width()
        canvas_h = self.photo_canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            canvas_w, canvas_h = 800, 600

        fh, fw = frame_rgb.shape[:2]

        # Calculate base scale (fit to window)
        base_scale = min(canvas_w / fw, canvas_h / fh)

        # Apply zoom
        scale = base_scale * self.photo_zoom_level
        new_w = int(fw * scale)
        new_h = int(fh * scale)

        if new_w < 10 or new_h < 10:
            return

        # Resize frame
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h),
                                    interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

        img = Image.fromarray(frame_resized)
        self._photo_image = ImageTk.PhotoImage(image=img)

        self.photo_canvas.delete("all")

        # Center with pan offset
        x_offset = (canvas_w - new_w) // 2 + self.photo_pan_offset[0]
        y_offset = (canvas_h - new_h) // 2 + self.photo_pan_offset[1]
        self.photo_canvas.create_image(x_offset, y_offset, anchor="nw",
                                        image=self._photo_image)

    def _on_photo_scroll(self, event):
        """Mouse wheel zoom on photo."""
        if self.photo_current_idx < 0:
            return

        # Determine scroll direction
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            factor = 1.15  # zoom in
        elif event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            factor = 0.87  # zoom out
        else:
            return

        new_zoom = self.photo_zoom_level * factor
        new_zoom = max(0.3, min(5.0, new_zoom))
        self.photo_zoom_level = new_zoom

        self._show_current_photo()

    def _on_photo_drag_start(self, event):
        """Start dragging photo."""
        self._photo_drag_start = (event.x, event.y)

    def _on_photo_drag(self, event):
        """Drag photo with pan offset."""
        if self._photo_drag_start is None:
            return

        dx = event.x - self._photo_drag_start[0]
        dy = event.y - self._photo_drag_start[1]
        self._photo_drag_start = (event.x, event.y)

        self.photo_pan_offset = (
            self.photo_pan_offset[0] + dx,
            self.photo_pan_offset[1] + dy
        )

        self._show_current_photo()

    def _on_photo_drag_end(self, event):
        """End dragging photo."""
        self._photo_drag_start = None

    def _reset_photo_zoom(self):
        """Reset photo zoom to 1x and pan to center."""
        self.photo_zoom_level = 1.0
        self.photo_pan_offset = (0, 0)
        self._show_current_photo()

    def _prev_photo(self):
        """Show previous photo in batch."""
        if not self.photo_images or self.photo_current_idx <= 0:
            return
        self.photo_current_idx -= 1
        self._show_current_photo()

    def _next_photo(self):
        """Show next photo in batch."""
        if not self.photo_images or self.photo_current_idx >= len(self.photo_images) - 1:
            return
        self.photo_current_idx += 1
        self._show_current_photo()

    def _build_thumbnails(self):
        """Build thumbnail bar for batch images."""
        import tkinter as tk

        # Clear existing thumbnails
        for widget in self.thumb_inner.winfo_children():
            widget.destroy()

        self._thumb_buttons = []

        for i, path in enumerate(self.photo_images):
            # Create small thumbnail
            img = cv2.imread(path)
            if img is None:
                continue

            thumb_h = 60
            thumb_w = int(img.shape[1] * thumb_h / img.shape[0])
            thumb_img = cv2.resize(img, (thumb_w, thumb_h))
            thumb_rgb = cv2.cvtColor(thumb_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(thumb_rgb)
            tk_img = ImageTk.PhotoImage(image=pil_img)

            btn = tk.Button(self.thumb_inner, image=tk_img,
                            command=lambda idx=i: self._on_thumbnail_click(idx),
                            bg=BG_PANEL, relief=tk.FLAT, padx=2, pady=2)
            btn.image = tk_img  # prevent garbage collection
            btn.pack(side=tk.LEFT, padx=2)

            self._thumb_buttons.append(btn)

    def _on_thumbnail_click(self, idx):
        """Switch to clicked thumbnail."""
        if idx < 0 or idx >= len(self.photo_images):
            return
        self.photo_current_idx = idx
        self._show_current_photo()

    def _update_thumbnail_border(self, idx):
        """Update thumbnail border to show detection status."""
        if not hasattr(self, '_thumb_buttons') or idx >= len(self._thumb_buttons):
            return

        path = self.photo_images[idx]
        btn = self._thumb_buttons[idx]

        if path in self.photo_results:
            results = self.photo_results[path]
            has_drowning = any(r.get("fine_class") == "drowning" for r in results)
            has_possible = any(r.get("fine_class") == "drowning_possible" for r in results)
            if has_drowning:
                btn.config(relief="raised", bg="#880000")  # Red border
            elif has_possible:
                btn.config(relief="raised", bg="#8B4513")  # Dark orange/brown border
            elif len(results) > 0:
                btn.config(relief="raised", bg="#006600")  # Green border
            else:
                btn.config(relief="raised", bg="#444444")  # Gray
        else:
            btn.config(relief="flat", bg=BG_PANEL)

    def _save_photo_result(self):
        """Save current annotated photo."""
        if self.photo_current_idx < 0:
            from tkinter import messagebox
            messagebox.showinfo("No Image", "No photo to save.")
            return

        path = self.photo_images[self.photo_current_idx]
        annotated = self.photo_annotated.get(path)

        if annotated is None:
            from tkinter import messagebox
            messagebox.showinfo("No Result", "Detect first before saving.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(path).stem
        save_path = PHOTO_SINGLE_DIR / f"{stem}_detected_{timestamp}.jpg"
        cv2.imwrite(str(save_path), annotated)
        self._update_status(f"Saved: {save_path}")

    def _save_all_results(self):
        """Save all annotated batch photos."""
        if not self.photo_annotated:
            from tkinter import messagebox
            messagebox.showinfo("No Results", "Detect first before saving.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = PHOTO_BATCH_DIR / f"batch_{timestamp}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for path, annotated in self.photo_annotated.items():
            stem = Path(path).stem
            save_path = batch_dir / f"{stem}_detected.jpg"
            cv2.imwrite(str(save_path), annotated)
            count += 1

        self._update_status(f"Saved {count} images to: {batch_dir}")

    # ------------------------------------------------------------------
    #  Video Mode
    # ------------------------------------------------------------------

    def _load_video(self):
        """Load a video file via file dialog."""
        from tkinter import filedialog

        # Stop current video if running
        if self.video_running:
            self._stop_video()

        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                       ("All files", "*.*")],
            initialdir=str(PROJECT_ROOT),
        )
        if not path:
            return

        self.video_path = path
        self.video_cap = cv2.VideoCapture(path)

        if not self.video_cap.isOpened():
            from tkinter import messagebox
            messagebox.showerror("Error", f"Cannot open video:\n{path}")
            self.video_cap = None
            return

        self.video_total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.video_current_frame = 0
        self._video_frame_idx = 0
        self._tracker = None  # 新视频重置时序跟踪器

        # Update progress slider range
        self.video_progress_slider.config(from_=0, to=max(1, self.video_total_frames - 1))

        # Show video info
        width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = self.video_total_frames / self.video_fps
        self.video_info_label.config(
            text=f"{Path(path).name} | {width}x{height} | {self.video_fps:.1f}fps | "
                 f"{duration:.1f}s | {self.video_total_frames} frames")
        self.video_frame_label.config(text="Frame: 0/--")

        # Read and display first frame
        ret, frame = self.video_cap.read()
        if ret:
            self._update_video_display(frame)
            self.video_current_frame = 1
            self.video_frame_label.config(
                text=f"Frame: 1/{self.video_total_frames}")

        self._update_status(f"Video loaded: {Path(path).name}")

    def _play_video(self):
        """Start or resume video playback with detection."""
        if self.video_cap is None or not self.video_cap.isOpened():
            from tkinter import messagebox
            messagebox.showinfo("No Video", "Load a video first.")
            return

        if self.stage1_model is None:
            from tkinter import messagebox
            messagebox.showwarning("Stage 1 Not Loaded",
                                   "Load Stage 1 detection model first.")
            return

        self.video_running = True
        self.video_paused = False
        self._update_status("Video: Playing...")
        self.root.after(10, self._process_video_frame)

    def _pause_video(self):
        """Pause video playback."""
        self.video_paused = True
        self._update_status("Video: Paused")

    def _stop_video(self):
        """Stop video playback and release capture."""
        self.video_running = False
        self.video_paused = False

        if self.video_recording:
            self._toggle_video_save()

        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None

        self._update_status("Video: Stopped")

    def _process_video_frame(self):
        """Process a single video frame in the main loop."""
        if not self.video_running or self.video_cap is None:
            return

        if self.video_paused:
            # Keep scheduling but don't process
            self.root.after(50, self._process_video_frame)
            return

        # Read frame(s), considering speed (frame skip)
        skip = self.speed_var.get()
        ret = True
        raw_frame = None

        for _ in range(skip + 1):
            ret, raw_frame = self.video_cap.read()
            if not ret:
                break
            self.video_current_frame += 1

        if not ret or raw_frame is None:
            # Video ended
            self.video_running = False
            self._tracker = None
            if self.video_recording:
                self._toggle_video_save()
            self._update_status("Video: Completed")
            self.video_frame_label.config(
                text=f"Frame: {self.video_current_frame}/{self.video_total_frames} (Done)")
            return

        # Two-stage detection
        if self.pipeline_mode == "temporal":
            self._get_or_create_tracker()
        results, s1_time, s2_time, total_time = self._detect_frame(raw_frame)
        self._video_frame_idx += 1

        # 时序模式: 清理过期 track
        if self.pipeline_mode == "temporal" and self._tracker is not None:
            active_ids = {r.get("track_id") for r in results if r.get("track_id") is not None}
            self._tracker.cleanup(active_ids, self._video_frame_idx)

        display_frame = self._draw_pipeline_results(raw_frame.copy(), results)

        # Draw timing overlay
        h, w = display_frame.shape[:2]
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)

        fps_text = (f"S1: {s1_time:.0f}ms + S2: {s2_time:.0f}ms = "
                    f"{total_time:.0f}ms | Frame: {self.video_current_frame}/{self.video_total_frames}")
        cv2.putText(display_frame, fps_text, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.putText(display_frame,
                    f"conf={self.conf_threshold:.2f} warn>{self.drowning_threshold:.2f}",
                    (w - 220, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Drowning alert overlay (graded)
        if self.drowning_detected:
            self._alert_flash = not self._alert_flash
            if self._alert_flash:
                cv2.rectangle(display_frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
                cv2.putText(display_frame, "DROWNING ALERT!", (w // 2 - 130, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        elif self.drowning_possible_detected:
            # Possible drowning: orange border (steady, no flash)
            cv2.rectangle(display_frame, (0, 0), (w - 1, h - 1), (0, 165, 255), 3)
            cv2.putText(display_frame, "POSSIBLE DROWNING", (w // 2 - 130, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        # Recording indicator
        if self.video_recording and self.video_writer:
            cv2.circle(display_frame, (w - 80, 18), 6, (0, 0, 255), -1)
            cv2.putText(display_frame, "REC", (w - 65, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Update UI
        self._update_video_display(display_frame)
        self._update_class_counts(results)
        self._update_stats_display()
        self._update_alert()
        self.video_frame_label.config(
            text=f"Frame: {self.video_current_frame}/{self.video_total_frames}")

        # Update progress slider
        self.video_progress_var.set(self.video_current_frame)

        # Write to output if recording
        if self.video_recording and self.video_writer:
            self.video_writer.write(display_frame)

        # Schedule next frame
        self.root.after(10, self._process_video_frame)

    def _update_video_display(self, frame):
        """Display a frame on the video canvas."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        canvas_w = self.video_canvas.winfo_width()
        canvas_h = self.video_canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            canvas_w, canvas_h = 800, 600

        fh, fw = frame_rgb.shape[:2]
        scale = min(canvas_w / fw, canvas_h / fh)
        new_w, new_h = int(fw * scale), int(fh * scale)
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))

        img = Image.fromarray(frame_resized)
        self._video_photo = ImageTk.PhotoImage(image=img)

        self.video_canvas.delete("all")
        x_offset = (canvas_w - new_w) // 2
        y_offset = (canvas_h - new_h) // 2
        self.video_canvas.create_image(x_offset, y_offset, anchor="nw",
                                        image=self._video_photo)

    def _on_video_seek(self, val):
        """Seek to a specific frame in the video."""
        if self.video_cap is None or not self.video_cap.isOpened():
            return

        # Debounce: only seek after slider settles
        if self._video_seek_debounce:
            self.root.after_cancel(self._video_seek_debounce)

        self._video_seek_debounce = self.root.after(
            100, lambda: self._do_video_seek(int(val)))

    def _do_video_seek(self, frame_idx):
        """Actually perform the video seek."""
        if self.video_cap is None or not self.video_cap.isOpened():
            return

        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self.video_current_frame = frame_idx

        ret, frame = self.video_cap.read()
        if ret:
            # If paused or stopped, just display the raw frame
            if not self.video_running or self.video_paused:
                self._update_video_display(frame)

            self.video_current_frame = frame_idx + 1
            self.video_frame_label.config(
                text=f"Frame: {self.video_current_frame}/{self.video_total_frames}")

    def _on_speed_change(self):
        """Handle speed change from radio buttons."""
        speed = self.speed_var.get()
        self._update_status(f"Speed: {speed + 1}x (skip {speed} frames)")

    def _toggle_video_save(self):
        """Start or stop video recording."""
        if self.video_recording:
            self._stop_video_save()
        else:
            self._start_video_save()

    def _start_video_save(self):
        """Start recording output video."""
        if self.video_cap is None or not self.video_cap.isOpened():
            return

        # Need a frame to get dimensions
        current_pos = self.video_cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = self.video_cap.read()
        if not ret:
            return

        h, w = frame.shape[:2]

        # Reset position
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(self.video_path).stem if self.video_path else "video"
        self.video_record_path = VIDEO_OUTPUT_DIR / f"{stem}_detected_{timestamp}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            str(self.video_record_path), fourcc, self.video_fps, (w, h))
        self.video_recording = True
        self.video_save_btn.config(text="Stop Save", bg="#008800")
        self._update_status(f"Recording: {self.video_record_path}")

    def _stop_video_save(self):
        """Stop recording and save output video."""
        self.video_recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.video_save_btn.config(text="Start Save", bg="#8B4513")
        if self.video_record_path:
            self._update_status(f"Saved: {self.video_record_path}")

    # ------------------------------------------------------------------
    #  Window Management
    # ------------------------------------------------------------------

    def _on_close(self):
        """Handle window close event."""
        self._batch_running = False
        self.video_running = False
        if self.video_recording:
            self._stop_video_save()
        if self.video_cap:
            self.video_cap.release()
        self.root.destroy()

    def run(self):
        """Start GUI main loop."""
        self.root.mainloop()


# ===========================================================================
#  Main
# ===========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Two-Stage Media Detection GUI")
    parser.add_argument("--stage1", type=str, default=str(DEFAULT_STAGE1),
                        help="Stage 1 detection model path (.pt)")
    parser.add_argument("--stage2", type=str, default=str(DEFAULT_STAGE2),
                        help="Stage 2 classification model path (.pt)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help="Stage 1 confidence threshold")
    parser.add_argument("--drowning-threshold", type=float, default=DEFAULT_DROWNING_THRESHOLD,
                        help="Drowning warning threshold (drowning_conf>=此值报drowning_possible)")
    parser.add_argument("--route-conf", type=float, default=ROUTE_CONF,
                        help="Stage1 低于此置信度的框不送 Stage2 (源头砍低质误检)")
    parser.add_argument("--drowning-confirm", type=float, default=DROWNING_CONFIRM,
                        help="drowning 确认阈值 (drowning_conf>=此值且>swimming_conf 才报红框)")
    parser.add_argument("--min-class-conf", type=float, default=MIN_CLASS_CONF,
                        help="Stage2 最小分类置信度, max(d,s)<此值视为存疑/非人, 不告警")
    args = parser.parse_args()

    # Auto-find models if defaults don't exist
    stage1_path = find_model(
        args.stage1, DEFAULT_STAGE1,
        [PROJECT_ROOT / "runs" / "stage1_detect",
         PROJECT_ROOT / "runs" / "stage1_detect" / "yolo26n_stage1_pure_v1",
         PROJECT_ROOT / "runs" / "stage1_pure",
         PROJECT_ROOT / "runs" / "stage1_v1",
         PROJECT_ROOT / "runs" / "stage1_coarse"]
    )
    stage2_path = find_model(
        args.stage2, DEFAULT_STAGE2,
        [PROJECT_ROOT / "runs" / "stage2_classify",
         PROJECT_ROOT / "runs" / "stage2_classify" / "yolo26n_stage2_cls_v1",
         PROJECT_ROOT / "runs" / "stage2_v1",
         PROJECT_ROOT / "runs" / "stage2_classify" / "yolo26s_cls_stage2_v1",
         PROJECT_ROOT / "runs" / "stage2_classify" / "yolo26n_cls_stage2_v1"]
    )

    app = MediaDetectGUI(
        stage1_path=stage1_path,
        stage2_path=stage2_path,
        conf=args.conf,
        drowning_threshold=args.drowning_threshold,
        route_conf=args.route_conf,
        drowning_confirm=args.drowning_confirm,
        min_class_conf=args.min_class_conf,
    )
    app.run()


if __name__ == "__main__":
    main()
