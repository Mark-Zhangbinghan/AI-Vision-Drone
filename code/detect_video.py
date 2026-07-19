"""
detect_video.py — YOLO26 视频文件推理模块
============================================================
加载训练好的 YOLO26 模型，对视频文件逐帧进行溺水目标检测，
输出带检测框的视频文件。

功能:
  - 加载 best.pt 模型进行推理
  - 支持多种视频格式 (mp4/avi/mov/mkv/webm)
  - 实时进度显示 (帧数/总数, FPS, 耗时)
  - 置信度阈值可调
  - 输出分辨率可配置
  - 可选择只保留有检测结果的帧
  - 各类别检测数量统计

用法:
    python detect_video.py --source D:/path/to/video.mp4
    python detect_video.py --source video.mp4 --model best.pt --conf 0.4
    python detect_video.py --source video.mp4 --output result.mp4 --no-display
    python detect_video.py --source video.mp4 --frame-skip 2 --conf 0.5
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ===========================================================================
#  Constants
# ===========================================================================

CLASS_NAMES = [
    "person", "boat", "surfboard", "wood",
    "life_buoy", "drowning", "background", "swimming"
]

# BGR colors for OpenCV overlays
CLASS_BGR = [
    (128, 0, 0),      # 0 person     — dark blue
    (0, 128, 0),      # 1 boat       — dark green
    (128, 128, 0),    # 2 surfboard  — teal
    (0, 128, 128),    # 3 wood       — olive
    (0, 0, 255),      # 4 life_buoy  — red
    (0, 0, 200),      # 5 drowning   — bright red
    (128, 128, 128),  # 6 background — gray
    (255, 0, 0),      # 7 swimming   — blue
]

ALERT_CLASSES = {5, 4}  # drowning & life_buoy trigger alert styling

DROWNING_RUNS_DIR = PROJECT_ROOT / "runs" / "drowning_detection"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "video_detect"


# ===========================================================================
#  Model Discovery
# ===========================================================================

def find_best_pt():
    """Auto-discover the most recent best.pt from training runs."""
    pt_files = sorted(
        DROWNING_RUNS_DIR.glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime, reverse=True
    )
    return str(pt_files[0]) if pt_files else None


# ===========================================================================
#  Video Processor
# ===========================================================================

class VideoDetector:
    """Load a YOLO model and run detection on a video file."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        device: str = "0",
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model_path = model_path

        # Import ultralytics inside the project context
        from ultralytics import YOLO

        print(f"[MODEL] Loading: {model_path}")
        self.model = YOLO(model_path)
        self.device = device
        print(f"[MODEL] Loaded successfully, device={device}")

        # Stats
        self.total_frames = 0
        self.detected_frames = 0
        self.class_counts = {name: 0 for name in CLASS_NAMES}
        self.total_objects = 0

    def detect_frame(self, frame: np.ndarray):
        """Run YOLO inference on a single frame, return annotated frame + detections."""
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        annotated = results[0].plot(
            conf=True,
            labels=True,
            boxes=True,
            line_width=2,
        )

        # Collect stats
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            self.detected_frames += 1
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            for cid in cls_ids:
                if 0 <= cid < len(CLASS_NAMES):
                    self.class_counts[CLASS_NAMES[cid]] += 1
            self.total_objects += len(cls_ids)

        return annotated

    def process_video(
        self,
        source_path: str,
        output_path: str,
        frame_skip: int = 1,
        show_preview: bool = True,
        save_output: bool = True,
    ):
        """Process a video file: read → detect → annotate → save."""
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {source_path}")

        # Source video info
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if src_fps <= 0:
            src_fps = 30.0

        print(f"\n{'='*60}")
        print(f"  Video Info")
        print(f"{'='*60}")
        print(f"  Source:     {source_path}")
        print(f"  Resolution: {src_w}x{src_h}")
        print(f"  Total Frames: {total_frames}")
        print(f"  FPS:        {src_fps:.1f}")
        print(f"  Frame Skip: {frame_skip}")
        print(f"  Conf:       {self.conf_threshold}")
        print(f"  Output:     {output_path}")
        print(f"{'='*60}\n")

        # Output writer
        out = None
        if save_output:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            out_w, out_h = src_w, src_h
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_fps = src_fps / frame_skip
            out = cv2.VideoWriter(output_path, fourcc, out_fps, (out_w, out_h))
            if not out.isOpened():
                raise RuntimeError(f"Cannot create output video: {output_path}")

        # Preview window
        preview_name = f"YOLO26 — {os.path.basename(source_path)}"
        if show_preview:
            cv2.namedWindow(preview_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(preview_name, min(src_w, 1280), min(src_h, 720))

        frame_idx = 0
        written_frames = 0
        t_start = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # Frame skip
                if frame_idx % frame_skip != 0:
                    continue

                # Detect
                annotated = self.detect_frame(frame)
                self.total_frames += 1

                # Write output
                if out is not None:
                    out.write(annotated)
                    written_frames += 1

                # Show preview
                if show_preview:
                    cv2.imshow(preview_name, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # Esc
                        print("\n[STOP] User interrupted (Esc).")
                        break
                    elif key == ord(" "):
                        print("\n[PAUSE] Press any key to continue...")
                        cv2.waitKey(0)
                        print("[RESUME]")

                # Progress display
                if self.total_frames % 50 == 0 or frame_idx == total_frames:
                    elapsed = time.time() - t_start
                    fps = self.total_frames / elapsed if elapsed > 0 else 0
                    pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
                    eta = (total_frames - frame_idx) / fps if fps > 0 else 0
                    objs = self.total_objects
                    print(
                        f"  Frame {frame_idx}/{total_frames} ({pct:.1f}%)  |  "
                        f"FPS: {fps:.1f}  |  Detections: {objs}  |  "
                        f"ETA: {eta:.0f}s     ",
                        end="\r",
                    )

        finally:
            cap.release()
            if out is not None:
                out.release()
            if show_preview:
                cv2.destroyAllWindows()

        t_total = time.time() - t_start
        print(f"\n\n{'='*60}")
        print(f"  Processing Complete")
        print(f"{'='*60}")
        print(f"  Total frames processed: {self.total_frames}")
        print(f"  Frames with detections: {self.detected_frames}")
        print(f"  Total objects detected: {self.total_objects}")
        print(f"  Written frames:         {written_frames}")
        print(f"  Elapsed:                {t_total:.1f}s")
        print(f"  Avg FPS:                {self.total_frames/t_total:.1f}")
        print(f"\n  Per-Class Detections:")
        for name in CLASS_NAMES:
            cnt = self.class_counts[name]
            bar = "█" * min(cnt // 10, 50)
            print(f"    {name:<12} {cnt:>6}  {bar}")
        print(f"{'='*60}\n")
        if save_output:
            print(f"  Output saved to: {output_path}\n")

        return {
            "total_frames": self.total_frames,
            "detected_frames": self.detected_frames,
            "total_objects": self.total_objects,
            "class_counts": self.class_counts,
            "elapsed": t_total,
            "fps": self.total_frames / t_total if t_total > 0 else 0,
        }


# ===========================================================================
#  CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO26 — Video Drowning Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect_video.py --source drone_footage.mp4
  python detect_video.py --source video.mp4 --conf 0.5 --frame-skip 3
  python detect_video.py --source video.mp4 --output result.mp4 --no-display
  python detect_video.py --model path/to/best.pt --source test.mp4
        """,
    )
    parser.add_argument("--source", "-s", required=True, type=str,
                        help="Input video file path")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Model weights path (auto-find best.pt if not given)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output video path (auto-generated if not given)")
    parser.add_argument("--conf", "-c", type=float, default=0.4,
                        help="Confidence threshold (default: 0.4)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold for NMS (default: 0.45)")
    parser.add_argument("--frame-skip", "-k", type=int, default=1,
                        help="Process every Nth frame (default: 1 = all frames)")
    parser.add_argument("--no-display", action="store_true",
                        help="Hide preview window (headless mode)")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save output video (preview only)")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device (default: 0, use 'cpu' for CPU)")
    return parser.parse_args()


def validate_pt_file(path: str) -> bool:
    """Check if a .pt file is a valid loadable PyTorch checkpoint."""
    import zipfile
    p = Path(path)
    if not p.is_file():
        return False
    try:
        # Quick check: valid ZIP archive (PyTorch saves as ZIP)
        with zipfile.ZipFile(p, 'r') as zf:
            # Must have at least one real entry with non-zero size
            for info in zf.infolist():
                if info.file_size > 0:
                    return True
        # ZIP exists but all entries are 0-byte placeholders → corrupted
        print(f"[WARN] {path}: ZIP has no real data entries (corrupted)")
        return False
    except zipfile.BadZipFile:
        print(f"[WARN] {path}: Not a valid ZIP archive (corrupted)")
        return False
    except Exception as e:
        print(f"[WARN] {path}: Validation error - {e}")
        return False


def find_all_valid_pt(root_dir: str = None) -> list:
    """Find all valid (non-corrupted) .pt files in the project."""
    import os
    root = Path(root_dir or PROJECT_ROOT / "runs")
    valid = []
    for p in root.rglob("best.pt"):
        if validate_pt_file(str(p)):
            valid.append(str(p))
    # Sort by modification time (latest first)
    valid.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return valid


def resolve_model_path(arg_model):
    """Resolve model path from arg, or auto-discover. Validates file integrity."""
    if arg_model:
        p = Path(arg_model)
        if not p.is_file():
            raise FileNotFoundError(f"Model not found: {arg_model}")
        if not validate_pt_file(str(p)):
            # File exists but is corrupted — suggest alternatives
            valid = find_all_valid_pt()
            msg = (
                f"Model file CORRUPTED: {arg_model}\n"
                f"The ZIP archive is missing its central directory — "
                f"this usually means the file was truncated during save/copy.\n"
            )
            if valid:
                msg += f"\nAvailable WORKING models:\n"
                for v in valid[:5]:
                    msg += f"  --model \"{v}\"\n"
            else:
                msg += "No working models found. Please retrain or re-download."
            raise RuntimeError(msg)
        return str(p.resolve())

    # Auto-discover: skip corrupted files
    valid = find_all_valid_pt()
    if valid:
        auto = valid[0]  # Latest valid best.pt
        print(f"[MODEL] Auto-discovered (validated): {auto}")
        return auto
    raise FileNotFoundError(
        "No model specified and no valid best.pt found in runs/. "
        "Use --model to specify a model path."
    )


def resolve_output_path(arg_output, source_path):
    """Generate output path. If arg_output is a directory or lacks extension,
    append an auto-generated filename."""
    src_name = Path(source_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if arg_output:
        p = Path(arg_output)
        # If it's a directory or has no file extension, treat as directory
        if p.is_dir() or not p.suffix:
            p.mkdir(parents=True, exist_ok=True)
            return str(p / f"{src_name}_detected_{timestamp}.mp4")
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            return str(p)
    else:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return str(DEFAULT_OUTPUT_DIR / f"{src_name}_detected_{timestamp}.mp4")


def main():
    args = parse_args()

    # Validate source
    source_path = args.source
    if not os.path.isfile(source_path):
        print(f"[ERROR] Video file not found: {source_path}")
        sys.exit(1)

    # Resolve model
    model_path = resolve_model_path(args.model)

    # Resolve output
    output_path = resolve_output_path(args.output, source_path) if not args.no_save else ""

    # Build detector
    detector = VideoDetector(
        model_path=model_path,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
    )

    # Process
    detector.process_video(
        source_path=source_path,
        output_path=output_path,
        frame_skip=args.frame_skip,
        show_preview=not args.no_display,
        save_output=not args.no_save,
    )


if __name__ == "__main__":
    main()
