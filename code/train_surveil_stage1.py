"""
train_surveil_stage1.py - Stage1 监控视角检测器训练 (nc=2)
==========================================================
类别: 0=person_in_water(水中人)  1=person(岸上/安全)

对应已定稿方案:
  - radiant-pulse-curie.md: 两阶段+3分支, Stage1 判「是否在水里」
  - 优化器 AdamW 方案A, 修正冻结回调, 在线天气增强

输入数据: picture_process/unified_surveillance_v1/stage1_detect/ (images+labels, 80/10/10 划分)

Usage (服务器单卡4090):
    python train_surveil_stage1.py
    python train_surveil_stage1.py --epochs 250 --batch 16 --device 0
    python train_surveil_stage1.py --no-weather        # 关闭在线天气增强
    python train_surveil_stage1.py --no-freeze         # 关闭渐进冻结
    python train_surveil_stage1.py --data /path/to/stage1_detect/data.yaml
"""

import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from config_surveil import (
    SURVEIL_STAGE1_CONFIG,
    SurveilFreezeCallback,
    build_train_args,
)
from weather_augment import WeatherBatchCallback

STAGE1_CLASS_NAMES = ["person_in_water", "person"]


def setup_model(config):
    """三级权重加载: 本地 -> 下载 -> 从零。"""
    from ultralytics import YOLO

    model, src = None, "unknown"
    p = Path(config.get("pretrained_model", ""))
    if p.exists():
        try:
            model = YOLO(str(p))
            src = f"local: {p.name}"
        except Exception as e:
            print(f"[WARN] 本地权重加载失败: {e}")

    if model is None:
        pt = f"yolo26{config.get('variant', 's')}.pt"
        try:
            model = YOLO(pt)
            src = f"download: {pt}"
        except Exception as e:
            print(f"[INFO] 下载失败: {e}")

    if model is None:
        cfg = config.get("model") or str(
            PROJECT_ROOT / "ultralytics" / "cfg" / "models" / "26" / f"yolo26{config.get('variant', 's')}.yaml"
        )
        model = YOLO(cfg)
        src = f"scratch: {Path(cfg).name}"

    print(f"[MODEL] 权重来源: {src}")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Stage1 监控视角检测器训练 (nc=3)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--variant", type=str, default=None, choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--data", type=str, default=None, help="stage1_detect/data.yaml 路径")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-freeze", action="store_true", default=False)
    parser.add_argument("--no-weather", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    config = SURVEIL_STAGE1_CONFIG.copy()
    if args.data: config["data"] = args.data
    if args.epochs: config["epochs"] = args.epochs
    if args.batch: config["batch"] = args.batch
    if args.device: config["device"] = args.device
    if args.variant:
        config["variant"] = args.variant
        config["model"] = str(PROJECT_ROOT / "ultralytics" / "cfg" / "models" / "26" / f"yolo26{args.variant}.yaml")
        config["pretrained_model"] = str(PROJECT_ROOT / f"yolo26{args.variant}.pt")
    if args.no_freeze: config["freeze_epochs"] = 0
    if args.no_weather: config["weather_aug"] = False
    if args.resume: config["resume"] = True

    # 环境信息
    print("=" * 64)
    print("  Stage1 监控视角检测器训练 (nc=2)")
    print("=" * 64)
    print(f"  Python:  {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU:     {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    print(f"  Data:    {config['data']}")
    print("=" * 64)

    model = setup_model(config)

    # 过滤为 ultralytics 接受的训练参数
    train_args = build_train_args(config, task="detect")
    train_args.update({
        "val": True,
        "plots": True,
        "save": True,
        "exist_ok": True,
        "pretrained": True,
        "resume": config.get("resume", False),
    })

    # 注册渐进冻结回调（修正版）
    if config["freeze_epochs"] > 0:
        freeze_cb = SurveilFreezeCallback(
            freeze_epochs=config["freeze_epochs"],
            freeze_n=config["freeze_n"],
        )
        model.add_callback("on_train_epoch_start", freeze_cb)
        print(f"[FREEZE] 渐进冻结: 前 {config['freeze_epochs']} epoch 冻 backbone 前 {config['freeze_n']} 层 + 始终冻 .dfl")
    else:
        print("[FREEZE] 渐进冻结已禁用")

    # 注册在线天气增强回调
    if config.get("weather_aug"):
        weather_cb = WeatherBatchCallback(p=config.get("weather_p", 0.3), enabled=True)
        model.add_callback("on_train_batch_start", weather_cb)
        print(f"[WEATHER] 在线天气增强已启用 (p={config.get('weather_p', 0.3)})")
    else:
        print("[WEATHER] 在线天气增强已禁用")

    print("\n" + "=" * 64)
    print("  开始训练")
    print("=" * 64)
    print(f"  类别:   {STAGE1_CLASS_NAMES}")
    print(f"  Epochs: {train_args['epochs']}  Batch: {train_args['batch']}  ImgSz: {train_args['imgsz']}")
    print(f"  Optim:  {config['optimizer']}  lr0={config['lr0']}  wd={config['weight_decay']}")
    print(f"  Model:  yolo26{config.get('variant', 's')}")
    print("=" * 64)

    start = time.time()
    try:
        model.train(**train_args)
    except torch.cuda.OutOfMemoryError:
        print("\n[ERROR] CUDA OOM! 尝试 --batch 8 或 --batch 4")
        sys.exit(1)

    elapsed = time.time() - start
    print(f"\n训练完成! 耗时: {int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m")

    # 测试集评估
    try:
        model.val(data=config["data"], split="test", plots=True)
    except Exception as e:
        print(f"[WARN] 测试评估失败: {e}")

    print("Done!")


if __name__ == "__main__":
    main()
