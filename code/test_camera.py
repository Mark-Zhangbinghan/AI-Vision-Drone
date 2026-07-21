"""
test_camera.py
=========================
Camera real-time drowning detection with microphone fusion toggle
1. Open USB camera
2. Stage1 YOLO detect person_in_water/person
3. Stage2 YOLO classify drowning/swimming (fine_class)
4. ByteTrack + sliding window state judgment (alarm_class / is_alarm)
5. Real-time render detection & warning results
6. Press key 'm' to enable/disable microphone drowning signal fusion
    - Microphone ON: Alarm when camera alarm OR mic drowning signal triggered, hydrophone thread auto start
    - Microphone OFF: Alarm only depends on vision pipeline result, stop reading audio signal
"""
import cv2
import threading
from pathlib import Path
from ultralytics import YOLO

import hydrophone_testing
from pipeline_inference import (
    two_stage_inference,
    draw_results,
    DrowningTracker
)

# ==============================
# Global Control Flags
# ==============================
use_microphone = False
hydrophone_running = False  # Track if hydrophone background thread launched
mic_drowning = False

# ==============================
# Model Path Configuration
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parent
STAGE1_WEIGHTS = (PROJECT_ROOT / "runs" / "yolo26s_surveil_stage1_v2" / "best.pt")
STAGE2_WEIGHTS = (PROJECT_ROOT / "runs" / "yolo26s_cls_surveil_stage2_v2" / "best.pt")

# ==============================
# Inference Hyperparameters
# ==============================
CAMERA_ID = 0
STAGE1_CONF = 0.35
DROWNING_THRESHOLD = 0.5
DROWNING_CONFIRM = 0.65
MIN_CLASS_CONF = 0.60
ROUTE_CONF = 0.35


def start_hydrophone():
    """Start hydrophone drowning detection in background daemon thread."""
    global hydrophone_running
    if hydrophone_running:
        print("[INFO] Hydrophone thread already running, skip launch")
        return
    thread = threading.Thread(
        target=hydrophone_testing.main,
        daemon=True
    )
    thread.start()
    hydrophone_running = True
    print("[INFO] Hydrophone detection thread started")


def main():
    global use_microphone, mic_drowning, hydrophone_running

    print("[INFO] Loading Stage1 detection model...")
    stage1 = YOLO(str(STAGE1_WEIGHTS))
    print("[INFO] Loading Stage2 classification model...")
    stage2 = YOLO(str(STAGE2_WEIGHTS))
    stage1.conf = STAGE1_CONF

    # ByteTrack Sliding Window Tracker Init
    tracker = DrowningTracker(
        window_size=90,
        alarm_ratio=0.6,
        stale_frame_threshold=60
    )

    # USB Camera Initialization
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("[ERROR] Failed to open camera device")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("[INFO] Camera stream started")
    print("[INFO] Press 'q' to exit program")
    print("[INFO] Press 'm' to toggle microphone fusion mode")

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read camera frame")
            break

        # Sync hydrophone drowning signal ONLY when mic fusion is enabled
        if use_microphone and hydrophone_running:
            mic_drowning = hydrophone_testing.drowning
        else:
            mic_drowning = False

        # Two-stage YOLO Inference Pipeline
        results = two_stage_inference(
            frame,
            stage1,
            stage2,
            tracker=tracker,
            frame_index=frame_id,
            drowning_threshold=DROWNING_THRESHOLD,
            drowning_confirm=DROWNING_CONFIRM,
            min_class_conf=MIN_CLASS_CONF,
            route_conf=ROUTE_CONF
        )

        # Clean up expired lost track IDs
        active_ids = {
            r["track_id"]
            for r in results
            if r.get("track_id") is not None
        }
        tracker.cleanup(active_ids, frame_id)

        # Draw detection bounding boxes & labels on frame
        output = draw_results(frame, results)

        # Get vision pipeline alarm state
        vision_alarm_active = any(r.get("is_alarm", False) for r in results)

        # Fusion Alarm Logic (Vision + Microphone Toggle)
        final_alarm = False
        if use_microphone:
            final_alarm = vision_alarm_active or mic_drowning
        else:
            final_alarm = vision_alarm_active

        # Draw global alarm text on screen
        if final_alarm:
            cv2.putText(
                output,
                "!!! DROWNING ALERT !!!",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3
            )

        # Draw frame number counter
        cv2.putText(
            output,
            f"Frame: {frame_id}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Draw current microphone fusion mode status
        mic_status_text = f"Microphone Fusion: {'ON' if use_microphone else 'OFF'} | MicDrown:{mic_drowning}"
        mic_color = (0, 255, 0) if use_microphone else (180, 180, 180)
        cv2.putText(
            output,
            mic_status_text,
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            mic_color,
            2
        )

        # Render video window
        cv2.imshow("Drowning Detection Camera", output)

        frame_id += 1

        # Keyboard Input Handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('m'):
            # Toggle microphone fusion switch
            use_microphone = not use_microphone
            state = "ENABLED" if use_microphone else "DISABLED"
            print(f"[INFO] Microphone fusion mode switched to {state}")
            # If turn mic ON, start hydrophone thread if not launched yet
            if use_microphone:
                start_hydrophone()

    # Release all resources before exit
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()