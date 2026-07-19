"""
train_stage2.py - Stage 2 细粒度分类器训练脚本
================================================
两阶段架构的第二阶段: drowning/swimming 二分类

类别定义:
  0: drowning   # 溺水人员
  1: swimming   # 游泳人员

训练策略:
  - YOLOv26 分类模式 (yolo26s-cls)
  - 较小图像尺寸 (256x256), 因为输入是裁剪区域
  - 较大 batch (64), 分类任务内存消耗小
  - drowning 优先: 训练数据中 drowning 过采样 20%

Usage:
    python train_stage2.py                    # 默认训练
    python train_stage2.py --epochs 150       # 更多 epochs
    python train_stage2.py --variant m        # 更大模型
"""

import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

DATASET_PATH = PROJECT_ROOT.parent / "picture_process" / "stage2_cls_dataset"

# ===========================================================================
#  Configuration
# ===========================================================================

STAGE2_CONFIG = {
    "variant": "s",
    "pretrained_model": "yolo26s-cls.pt",
    "data": str(DATASET_PATH),
    "epochs": 100,
    "imgsz": 256,
    "batch": 64,
    "device": 0,
    "workers": 4,
    "optimizer": "AdamW",
    "lr0": 0.01,
    "lrf": 0.01,
    "cos_lr": True,
    "warmup_epochs": 3,
    "amp": True,
    "patience": 30,
    "save_period": 10,
    "project": str(PROJECT_ROOT / "runs" / "stage2_classify"),
    "name": "yolo26s_cls_stage2_v1",
}


# ===========================================================================
#  Drowning Oversampling
# ===========================================================================

def oversample_drowning(dataset_path, ratio=1.2, seed=42):
    """
    对 drowning 类做过采样: 复制 drowning 图片, 使 drowning 数量约为 swimming 的 1.2x
    安全优先: 宁可多训练 drowning 样本, 提高溺水 recall
    """
    import random
    random.seed(seed)

    drowning_dir = Path(dataset_path) / "train" / "drowning"
    swimming_dir = Path(dataset_path) / "train" / "swimming"

    if not drowning_dir.exists() or not swimming_dir.exists():
        print("[WARN] 分类数据集目录不存在, 跳过过采样")
        return

    drowning_files = list(drowning_dir.glob("*.jpg"))
    swimming_files = list(swimming_dir.glob("*.jpg"))

    target_drowning = int(len(swimming_files) * ratio)
    current_drowning = len(drowning_files)

    if current_drowning >= target_drowning:
        print(f"[INFO] drowning 已有 {current_drowning} 张, 目标 {target_drowning}, 无需过采样")
        return

    need_copy = target_drowning - current_drowning
    print(f"[INFO] drowning 过采样: {current_drowning} → {target_drowning} (复制 {need_copy} 张)")

    import shutil
    random.shuffle(drowning_files)
    for i in range(need_copy):
        src = drowning_files[i % current_drowning]
        dst = drowning_dir / f"{src.stem}_oversample_{i}.jpg"
        shutil.copy2(src, dst)


# ===========================================================================
#  Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 2 细粒度分类器训练")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--variant", type=str, default=None, choices=["n", "s", "m"])
    parser.add_argument("--data", type=str, default=None,
                        help="数据集根目录绝对路径 (分类模式, 含 train/val 子目录)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--oversample", type=float, default=1.2,
                        help="drowning 过采样比例 (相对于 swimming)")
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()

    config = STAGE2_CONFIG.copy()
    if args.data: config["data"] = args.data
    if args.epochs: config["epochs"] = args.epochs
    if args.batch: config["batch"] = args.batch
    if args.variant: config["variant"] = args.variant
    if args.device: config["device"] = args.device

    # Environment check
    print("=" * 60)
    print("  Stage 2 细粒度分类器训练 (drowning vs swimming)")
    print("=" * 60)
    print(f"  Python:  {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU:     {props.name}")
    print("=" * 60)

    # Oversample drowning
    oversample_drowning(config["data"], ratio=args.oversample)

    # Initialize model
    from ultralytics import YOLO

    variant = config.get("variant", "s")
    pt_name = f"yolo26{variant}-cls.pt"

    model = None
    # Try local first
    local_path = PROJECT_ROOT / pt_name
    if local_path.exists():
        model = YOLO(str(local_path))
        print(f"[MODEL] 本地分类权重: {local_path}")
    else:
        # Try framework download
        try:
            model = YOLO(pt_name)
            print(f"[MODEL] 下载分类权重: {pt_name}")
        except Exception as e:
            print(f"[INFO] 下载失败, 从零训练: {e}")
            model_cfg = str(PROJECT_ROOT / "ultralytics" / "cfg" / "models" / "26" / f"yolo26{variant}-cls.yaml")
            model = YOLO(model_cfg)
            print(f"[MODEL] 从零训练: {model_cfg}")

    # Build training args
    train_args = {
        "data": config["data"],
        "epochs": config["epochs"],
        "imgsz": config["imgsz"],
        "batch": config["batch"],
        "device": config["device"],
        "workers": config.get("workers", 4),
        "optimizer": config.get("optimizer", "AdamW"),
        "lr0": config.get("lr0", 0.01),
        "lrf": config.get("lrf", 0.01),
        "cos_lr": config.get("cos_lr", True),
        "warmup_epochs": config.get("warmup_epochs", 3),
        "amp": config.get("amp", True),
        "patience": config.get("patience", 30),
        "save_period": config.get("save_period", 10),
        "project": config.get("project", str(PROJECT_ROOT / "runs" / "stage2_classify")),
        "name": config.get("name", "yolo26s_cls_stage2_v1"),
        "resume": config.get("resume", False),
        "plots": True,
        "exist_ok": True,
    }

    # Start training
    print("\n" + "=" * 60)
    print("  开始训练")
    print("=" * 60)
    print(f"  类别:   2 (drowning, swimming)")
    print(f"  Epochs: {train_args['epochs']}")
    print(f"  Batch:  {train_args['batch']}")
    print(f"  ImgSz:  {train_args['imgsz']}")
    print(f"  Model:  yolo26{variant}-cls")
    print("=" * 60)

    start_time = time.time()
    try:
        results = model.train(**train_args)
    except torch.cuda.OutOfMemoryError:
        print("\n[ERROR] CUDA OOM! 尝试降低 batch: --batch 32")
        sys.exit(1)

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    print(f"\n训练完成! 耗时: {hours}h {minutes}m")
    print("Done!")


if __name__ == "__main__":
    main()
