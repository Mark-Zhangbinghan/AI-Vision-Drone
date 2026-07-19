"""
train_stage1.py - Stage 1 粗粒度检测器训练脚本
================================================
两阶段架构的第一阶段: 5类粗粒度目标检测

类别定义:
  0: person           # 合并 person + background(不在水中的人)
  1: person_in_water   # 合并 drowning + swimming = "水中人"
  2: boat
  3: floating_object   # 合并 surfboard + wood
  4: life_buoy

训练策略:
  - 只冻结前 10 层 (P1/P2 浅特征), 第 5 epoch 后全解冻
  - 适度增强: mosaic=0.3, mixup=0, copy_paste=0.2
  - cls=2.0 (替代cls_pw逆频率加权效果, ultralytics不支持cls_pw)
  - yolo26s 变体 (5类粗粒度, s 够用)

Usage:
    python train_stage1.py                    # 默认训练
    python train_stage1.py --epochs 200       # 更多 epochs
    python train_stage1.py --variant m        # 更大模型
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from config_drowning import ProgressiveFreezeCallback

# ===========================================================================
#  Configuration
# ===========================================================================

STAGE1_CONFIG = {
    # ---- Model ----
    "variant": "s",
    "model": str(PROJECT_ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26s.yaml"),
    "pretrained_model": str(PROJECT_ROOT / "yolo26s.pt"),

    # ---- Data ----
    "data": str(PROJECT_ROOT.parent / "picture_process" / "stage1_dataset" / "data.yaml"),

    # ---- Training Core ----
    "epochs": 150,
    "batch": 16,
    "imgsz": 640,
    "device": 0,
    "workers": 4,

    # ---- Optimizer ----
    "optimizer": "AdamW",
    "lr0": 0.002,
    "lrf": 0.01,
    "cos_lr": True,
    "warmup_epochs": 5,

    # ---- Loss Weights ----
    "box": 7.5,
    "cls": 2.0,
    "dfl": 1.5,
    # cls_pw 不被 ultralytics 支持, 用 cls=2.0 + copy_paste=0.2 替代逆频率加权效果

    # ---- Data Augmentation ----
    "mosaic": 0.3,       # 降低 mosaic, 保留目标完整性
    "mixup": 0.0,        # 禁用 mixup, 防止模糊水中人特征
    "copy_paste": 0.2,   # 将稀缺类别实例粘贴到其他图
    "close_mosaic": 30,  # 最后 30 epoch 关闭 mosaic

    "hsv_h": 0.01,       # 减少色调变化 (水面颜色很重要)
    "hsv_s": 0.5,
    "hsv_v": 0.3,
    "degrees": 5.0,
    "translate": 0.1,
    "scale": 0.3,        # 小目标不宜大幅缩放
    "fliplr": 0.5,
    "multi_scale": 0.3,

    # ---- Mixed Precision ----
    "amp": True,

    # ---- Freeze ----
    "freeze_stages": [
        {"start_epoch": 0, "freeze": 10},      # 前10层浅特征冻结
        {"start_epoch": 5, "freeze": None},     # 第5 epoch 后全解冻
    ],

    # ---- Early Stopping & Saving ----
    "patience": 50,
    "save_period": 10,

    # ---- Output ----
    "project": str(PROJECT_ROOT / "runs" / "stage1_coarse"),
    "name": "yolo26s_stage1_v1",
}

# Stage 1 类名 (nc=5)
STAGE1_CLASS_NAMES = ["person", "person_in_water", "boat", "floating_object", "life_buoy"]


# ===========================================================================
#  Model Initialization
# ===========================================================================

def setup_stage1_model(config):
    """
    三级模型初始化: 本地权重 → 框架下载 → 从零训练
    """
    from ultralytics import YOLO

    variant = config.get("variant", "s")
    model = None
    weights_source = "unknown"

    # Tier 1: 本地预训练权重
    pretrained_path = Path(config.get("pretrained_model", ""))
    if pretrained_path.exists():
        try:
            model = YOLO(str(pretrained_path))
            weights_source = f"local: {pretrained_path.name}"
            print(f"[MODEL] 加载本地预训练权重: {pretrained_path}")
        except Exception as e:
            print(f"[WARN] 本地权重加载失败: {e}")

    # Tier 2: 框架下载
    if model is None:
        pt_name = f"yolo26{variant}.pt"
        try:
            model = YOLO(pt_name)
            weights_source = f"downloaded: {pt_name}"
            print(f"[MODEL] 下载权重: {pt_name}")
        except Exception as e:
            print(f"[INFO] 下载失败: {e}")

    # Tier 3: 从 YAML 从零训练
    if model is None:
        model_cfg = config.get("model", "")
        if not model_cfg or not Path(model_cfg).exists():
            model_cfg = str(PROJECT_ROOT / "ultralytics" / "cfg" / "models" / "26" / f"yolo26{variant}.yaml")
        model = YOLO(str(model_cfg))
        weights_source = f"scratch ({variant})"
        print(f"[MODEL] 从零训练: {model_cfg}")

    print(f"[MODEL] 权重来源: {weights_source}")
    return model, weights_source


# ===========================================================================
#  Main
# ===========================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1 粗粒度检测器训练")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--variant", type=str, default=None, choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--data", type=str, default=None,
                        help="数据集 data.yaml 的绝对路径")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-freeze", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()

    # Apply CLI overrides
    config = STAGE1_CONFIG.copy()
    if args.data: config["data"] = args.data
    if args.epochs: config["epochs"] = args.epochs
    if args.batch: config["batch"] = args.batch
    if args.variant:
        config["variant"] = args.variant
        config["model"] = str(PROJECT_ROOT / "ultralytics" / "cfg" / "models" / "26" / f"yolo26{args.variant}.yaml")
        config["pretrained_model"] = str(PROJECT_ROOT / f"yolo26{args.variant}.pt")
    if args.device: config["device"] = args.device
    if args.no_freeze: config["freeze_stages"] = []

    # Environment check
    print("=" * 60)
    print("  Stage 1 粗粒度检测器训练")
    print("=" * 60)
    print(f"  Python:  {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU:     {props.name}")
        print(f"  VRAM:    {props.total_memory / (1024**3):.1f} GB")
    print("=" * 60)

    # Initialize model
    model, weights_source = setup_stage1_model(config)

    # Build training arguments
    train_args = {
        "data": config["data"],
        "epochs": config["epochs"],
        "batch": config["batch"],
        "imgsz": config["imgsz"],
        "device": config["device"],
        "workers": config.get("workers", 4),
        "optimizer": config["optimizer"],
        "lr0": config["lr0"],
        "lrf": config["lrf"],
        "cos_lr": config.get("cos_lr", True),
        "warmup_epochs": config.get("warmup_epochs", 5),
        "box": config.get("box", 7.5),
        "cls": config.get("cls", 2.0),
        "dfl": config.get("dfl", 1.5),
        "mosaic": config.get("mosaic", 0.3),
        "mixup": config.get("mixup", 0.0),
        "copy_paste": config.get("copy_paste", 0.2),
        "close_mosaic": config.get("close_mosaic", 30),
        "hsv_h": config.get("hsv_h", 0.01),
        "hsv_s": config.get("hsv_s", 0.5),
        "hsv_v": config.get("hsv_v", 0.3),
        "degrees": config.get("degrees", 5.0),
        "translate": config.get("translate", 0.1),
        "scale": config.get("scale", 0.3),
        "fliplr": config.get("fliplr", 0.5),
        "multi_scale": config.get("multi_scale", 0.3),
        "amp": config.get("amp", True),
        "patience": config.get("patience", 50),
        "save_period": config.get("save_period", 10),
        "project": config.get("project", str(PROJECT_ROOT / "runs" / "stage1_coarse")),
        "name": config.get("name", "yolo26s_stage1_v1"),
        "resume": config.get("resume", False),
        "pretrained": True,
        "plots": True,
        "val": True,
        "save": True,
        "exist_ok": True,
    }

    # Register progressive freeze callback
    freeze_stages = config.get("freeze_stages", [])
    if freeze_stages and not config.get("resume", False):
        freeze_cb = ProgressiveFreezeCallback(freeze_stages)
        model.add_callback("on_train_epoch_start", freeze_cb)
        print(f"[FREEZE] 渐进式冻结启用: {len(freeze_stages)} 个阶段")

    # Start training
    print("\n" + "=" * 60)
    print("  开始训练")
    print("=" * 60)
    print(f"  类别:   5 (person, person_in_water, boat, floating_object, life_buoy)")
    print(f"  Epochs: {train_args['epochs']}")
    print(f"  Batch:  {train_args['batch']}")
    print(f"  Model:  yolo26{config.get('variant', 's')}")
    print(f"  cls:    {train_args['cls']} (加权替代cls_pw)")
    print("=" * 60)

    start_time = time.time()
    try:
        results = model.train(**train_args)
    except torch.cuda.OutOfMemoryError:
        print("\n[ERROR] CUDA OOM! 尝试降低 batch: --batch 8 或 --batch 4")
        sys.exit(1)

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    print(f"\n训练完成! 耗时: {hours}h {minutes}m")

    # Test evaluation
    if results:
        print("\n" + "=" * 60)
        print("  测试集评估")
        print("=" * 60)
        try:
            model.val(data=config["data"], split="test", plots=True)
        except Exception as e:
            print(f"[WARN] 测试评估失败: {e}")

    print("Done!")


if __name__ == "__main__":
    main()
