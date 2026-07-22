# 🚁 AI-Vision-Drone: Intelligent Drowning Detection & Vision System

## 📖 Project Overview

**AI-Vision-Drone** is an end-to-end intelligent computer vision system designed specifically for real-time drowning behavior detection and safety monitoring. Built upon the **YOLO26** framework, the system features a **two-stage cascaded inference architecture**, covering the complete pipeline from *in-water person detection* to *fine-grained drowning/swimming action classification*.

Additionally, the project integrates **hydrophone-assisted acoustic recognition** to address visual challenges in complex environments, such as glare, heavy rain, adverse weather, or long-distance monitoring.

## 💡 Key Features

- **Two-Stage Cascaded Detection Engine**
  - **Stage 1 (In-Water Target Detection):** Accurately locates human targets in water bodies while filtering out surface interference like floating debris, buoys, and water ripples.
  - **Stage 2 (Fine-Grained Classification):** Performs secondary classification on cropped human targets to distinguish between **normal swimming, drowning struggling, and floating/stationary states**.
- **Multimodal Fusion (Visual-Acoustic)**
  - Combines underwater hydrophone audio signals with computer vision streams, significantly reducing false positives and missed detections in harsh conditions (rainy days, intense glare, partial occlusions, distant targets).
- **Multi-Terminal GUI Interfaces**
  - Supports three input streams: **local USB cameras**, **offline image/video files**, and **real-time feed from DJI Tello drones**, enabling one-click real-time visualization and alerting.
- **End-to-End Training & Evaluation Toolchain**
  - Includes a complete suite for dataset preparation, online weather data augmentation, joint/distributed DDP training, ablation studies, and automated performance plotting.

## 🛠️ Hardware & Platform Compatibility

| **Component**           | **Description**                                              |
| ----------------------- | ------------------------------------------------------------ |
| **Drone Hardware**      | DJI Tello Series (powered by `djitellopy` library)           |
| **Auxiliary Sensors**   | Hydrophone (real-time audio acquisition via `pyserial` serial communication) |
| **Compute & Training**  | Single-GPU & Multi-GPU DDP (Distributed Data Parallel) training |
| **Video/Image Sources** | Local USB Cameras / RTSP drone video streams / Offline media files |

## 🚀 Quick Start

### 1. Environment Setup

Python **3.8.20** is recommended:

```bash
python --version  # Recommended: Python 3.8.20
```

### 2. Dependency Installation

Choose one of the following three methods:

```bash
# first choice
pip install djitellopy==2.5.0 lap==0.5.13 matplotlib==3.7.5 numpy==1.24.4 \
            opencv-python==4.13.0.92 pillow==10.4.0 psutil==7.2.2 pygame==2.6.1 \
            pyserial==3.5 pyyaml==6.0.3 scipy==1.10.1 torch==2.4.1 \
            torchvision==0.19.1 ultralytics==8.4.70 websockets==13.1
```

```bash
# second choice
conda env create -f environment.yaml
conda activate ai_vision_drone
```

```bash
# third choice
pip install -r requirements.txt
```

### 3. Run the Integrated Test

Execute the 3-in-1 integrated GUI test script (recommended for first-time runs):

```bash
python test_camera.py
```

## 📂 Directory Structure

```Plaintext
code/
├── 📄 pipeline_inference.py          # [Core] Two-stage cascaded inference engine
├── 📄 detect_video.py                # [Core] Single-model video inference script
│
├── 🖥️ detect_media_gui.py            # [GUI] Main image/video detection interface
├── 🖥️ detect_gui.py                  # [GUI] Real-time USB camera detection interface
├── 🖥️ Drone_GUI.py                   # [GUI] DJI Tello drone remote control interface
├── 🖥️ test_camera.py                 # [GUI] 3-in-1 integrated test entry point (Recommended)
│
├── 🚀 train_pipeline.py              # [Training] Main entry point for joint training
├── 📄 train_surveil_stage1.py        # [Training] Stage 1 in-water detection (v2)
├── 📄 train_surveil_stage2.py        # [Training] Stage 2 behavior classification (v2)
├── 📄 train.py                       # [Training] Multi-GPU DDP training (Legacy)
├── 📄 train_drowning.py              # [Training] Single-stage training (Legacy)
├── 📄 train_stage1.py                # [Training] Stage 1 5-class training (Legacy)
├── 📄 train_stage2.py                # [Training] Stage 2 standalone training (Legacy)
│
├── ⚙️ config_surveil.py              # [Config] Modern hyperparameters + freeze callbacks
├── ⚙️ config_drowning.py             # [Config] Legacy hyperparameters + freeze callbacks
├── ⚙️ weather_augment.py             # [Config] Online weather augmentation module
│
├── 📦 build_stage1_pure.py           # [Data] Stage 1 dataset builder
├── 📦 build_stage2_dataset.py        # [Data] Stage 2 dataset builder
├── 📦 strip_ignore_labels.py         # [Data] Label cleaning & filtering utilities
│
├── 📊 eval_stage1.py                 # [Eval] Stage 1 performance evaluation
├── 📊 analyze_stage2.py              # [Eval] Stage 2 dataset distribution analysis
├── 📊 analyze_stage2_p2.py           # [Eval] Annotation quality & consistency check
├── 📊 compare_optimizations.py       # [Eval] Ablation study analysis
├── 📊 custom_plots.py                # [Eval] Metrics visualization engine
│
├── 🛠️ hydrophone_testing.py          # [Hardware] Hydrophone serial communication test
├── 🛠️ repair_ckpt.py                 # [Tools] Checkpoint repair utility
│
├── 📂 ultralytics/                   # [Third-party] YOLO26 framework source code
└── 📂 runs/                          # [Outputs] Weights, validation plots & output videos
    ├── 📁 yolo26s_surveil_stage1_v2/ #   Stage 1 best.pt + validation results
    ├── 📁 yolo26s_cls_surveil_stage2_v2/ # Stage 2 best.pt + validation results
    └── 📁 media_detect/video/        #   GUI detection output videos
```

