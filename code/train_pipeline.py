"""
train_pipeline.py - 两阶段联合训练脚本
=======================================
Stage1 (detect, nc=2): person_in_water / person
Stage2 (classify, nc=2): drowning / swimming

一键完成: Stage1 训练 → Stage2 训练 → 端到端测试集评估

Usage (服务器单卡4090):
    python train_pipeline.py
    python train_pipeline.py --epochs1 200 --epochs2 150 --batch1 16 --batch2 64
    python train_pipeline.py --skip-stage1                    # 只训 Stage2
    python train_pipeline.py --skip-stage2                    # 只训 Stage1
    python train_pipeline.py --data /path/to/unified_surveillance_v1
"""

import sys
import time
import argparse
import shutil
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from config_surveil import (
    SURVEIL_STAGE1_CONFIG,
    SURVEIL_STAGE2_CONFIG,
    SurveilFreezeCallback,
    build_train_args,
    DATA_ROOT,
)
from weather_augment import WeatherBatchCallback

STAGE1_CLASSES = ["person_in_water", "person"]
STAGE2_CLASSES = ["drowning", "swimming"]


# ===========================================================================
#  工具函数
# ===========================================================================
def print_banner(title: str):
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


def print_env():
    print(f"  Python:   {sys.version.split()[0]}")
    print(f"  PyTorch:  {torch.__version__}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU:      {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    print(f"  DataRoot: {DATA_ROOT}")


def setup_detect_model(config):
    from ultralytics import YOLO
    model, src = None, "unknown"
    p = Path(config.get("pretrained_model", ""))
    if p.exists():
        try:
            model = YOLO(str(p)); src = f"local: {p.name}"
        except Exception as e:
            print(f"[WARN] 本地权重加载失败: {e}")
    if model is None:
        pt = f"yolo26{config.get('variant', 's')}.pt"
        try:
            model = YOLO(pt); src = f"download: {pt}"
        except Exception as e:
            print(f"[INFO] 下载失败: {e}")
    if model is None:
        cfg = config.get("model") or str(
            PROJECT_ROOT / "ultralytics" / "cfg" / "models" / "26" / f"yolo26{config.get('variant', 's')}.yaml"
        )
        model = YOLO(cfg); src = f"scratch: {Path(cfg).name}"
    print(f"[MODEL] 权重来源: {src}")
    return model


def setup_cls_model(config):
    from ultralytics import YOLO
    model, src = None, "unknown"
    pt = f"yolo26{config.get('variant', 's')}-cls.pt"
    local = PROJECT_ROOT / pt
    if local.exists():
        try:
            model = YOLO(str(local)); src = f"local: {pt}"
        except Exception as e:
            print(f"[WARN] 本地分类权重失败: {e}")
    if model is None:
        try:
            model = YOLO(pt); src = f"download: {pt}"
        except Exception as e:
            print(f"[INFO] 下载失败: {e}")
    if model is None:
        cfg = str(PROJECT_ROOT / "ultralytics" / "cfg" / "models" / "26" / f"yolo26{config.get('variant','s')}-cls.yaml")
        model = YOLO(cfg); src = f"scratch: {Path(cfg).name}"
    print(f"[MODEL] 权重来源: {src}")
    return model


def oversample_drowning(data_dir: str, ratio: float = 1.0, seed: int = 42):
    if ratio <= 1.0:
        print(f"[INFO] oversample={ratio} <= 1.0，跳过(已平衡)")
        return
    rng = random.Random(seed)
    d_dir = Path(data_dir) / "train" / "drowning"
    s_dir = Path(data_dir) / "train" / "swimming"
    if not d_dir.exists() or not s_dir.exists():
        print("[WARN] 分类目录不存在, 跳过过采样")
        return
    d_files = list(d_dir.glob("*.jpg"))
    s_files = list(s_dir.glob("*.jpg"))
    target = int(len(s_files) * ratio)
    need = target - len(d_files)
    if need <= 0:
        print(f"[INFO] drowning {len(d_files)} >= 目标 {target}, 无需过采样")
        return
    rng.shuffle(d_files)
    print(f"[INFO] drowning 过采样: {len(d_files)} -> {target} (复制 {need} 张)")
    for i in range(need):
        src = d_files[i % len(d_files)]
        dst = d_dir / f"{src.stem}_os{i}.jpg"
        if not dst.exists():
            shutil.copy2(src, dst)


# ===========================================================================
#  训练入口
# ===========================================================================
def train_stage1(config):
    print_banner("Stage 1: 检测器训练 (nc=2)")
    print_env()
    print(f"  类别:  {STAGE1_CLASSES}")
    print(f"  Data:  {config['data']}")

    model = setup_detect_model(config)

    train_args = build_train_args(config, task="detect")
    train_args.update({
        "val": True, "plots": True, "save": True,
        "exist_ok": True, "pretrained": True,
        "resume": config.get("resume", False),
    })

    # 渐进冻结
    if config["freeze_epochs"] > 0:
        freeze_cb = SurveilFreezeCallback(
            freeze_epochs=config["freeze_epochs"],
            freeze_n=config["freeze_n"],
        )
        model.add_callback("on_train_epoch_start", freeze_cb)
        print(f"[FREEZE] 前 {config['freeze_epochs']} epoch 冻 backbone 前 {config['freeze_n']} 层 + .dfl")

    # 在线天气增强
    if config.get("weather_aug"):
        weather_cb = WeatherBatchCallback(p=config.get("weather_p", 0.3), enabled=True)
        model.add_callback("on_train_batch_start", weather_cb)
        print(f"[WEATHER] 已启用 (p={config.get('weather_p', 0.3)})")

    print(f"  Epochs: {train_args['epochs']}  Batch: {train_args['batch']}  ImgSz: {train_args['imgsz']}")
    print(f"  Optim:  {config['optimizer']}  lr0={config['lr0']}  wd={config['weight_decay']}")
    print("=" * 64)

    start = time.time()
    try:
        model.train(**train_args)
    except torch.cuda.OutOfMemoryError:
        print("\n[ERROR] CUDA OOM! 减小 --batch1 重试")
        sys.exit(1)
    elapsed = time.time() - start
    print(f"\n[Stage1] 训练完成: {int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m")

    # 测试集评估
    try:
        model.val(data=config["data"], split="test", plots=True)
    except Exception as e:
        print(f"[WARN] Stage1 测试评估失败: {e}")

    # 返回 best.pt 路径
    save_dir = Path(train_args["project"]) / train_args["name"]
    best = save_dir / "weights" / "best.pt"
    if best.exists():
        print(f"[Stage1] best.pt → {best}")
    return best


def train_stage2(config, stage1_best=None):
    print_banner("Stage 2: 分类器训练 (nc=2)")
    print(f"  类别:  {STAGE2_CLASSES}")
    print(f"  Data:  {config['data']}")

    # 可选过采样
    oversample_drowning(config["data"], ratio=config.get("oversample", 1.0))

    model = setup_cls_model(config)

    train_args = build_train_args(config, task="classify")
    train_args.update({
        "val": True, "plots": True, "save": True,
        "exist_ok": True, "pretrained": True,
        "resume": config.get("resume", False),
    })

    print(f"  Epochs: {train_args['epochs']}  Batch: {train_args['batch']}  ImgSz: {train_args['imgsz']}")
    print(f"  Optim:  {config['optimizer']}  lr0={config['lr0']}  wd={config['weight_decay']}")
    print("=" * 64)

    start = time.time()
    try:
        model.train(**train_args)
    except torch.cuda.OutOfMemoryError:
        print("\n[ERROR] CUDA OOM! 减小 --batch2 重试")
        sys.exit(1)
    elapsed = time.time() - start
    print(f"\n[Stage2] 训练完成: {int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m")

    # 测试集评估
    try:
        model.val(data=config["data"], split="test", plots=True)
    except Exception as e:
        print(f"[WARN] Stage2 测试评估失败: {e}")

    save_dir = Path(train_args["project"]) / train_args["name"]
    best = save_dir / "weights" / "best.pt"
    if best.exists():
        print(f"[Stage2] best.pt → {best}")
    return best


# ===========================================================================
#  CLI
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="两阶段联合训练 (nc=2 + nc=2)")
    p.add_argument("--epochs1", type=int, default=None, help="Stage1 epochs")
    p.add_argument("--epochs2", type=int, default=None, help="Stage2 epochs")
    p.add_argument("--batch1", type=int, default=None, help="Stage1 batch size")
    p.add_argument("--batch2", type=int, default=None, help="Stage2 batch size")
    p.add_argument("--variant", type=str, default=None, choices=["n", "s", "m"])
    p.add_argument("--data", type=str, default=None, help="数据集根目录 (含 stage1_detect/ 和 stage2_classify/)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--skip-stage1", action="store_true")
    p.add_argument("--skip-stage2", action="store_true")
    p.add_argument("--no-freeze", action="store_true")
    p.add_argument("--no-weather", action="store_true")
    p.add_argument("--oversample", type=float, default=None)
    p.add_argument("--stage1-name", type=str, default=None, help="Stage1 输出目录名")
    p.add_argument("--stage2-name", type=str, default=None, help="Stage2 输出目录名")
    return p.parse_args()


def main():
    args = parse_args()
    s1_cfg = SURVEIL_STAGE1_CONFIG.copy()
    s2_cfg = SURVEIL_STAGE2_CONFIG.copy()

    # 覆盖参数
    if args.data:
        data_root = Path(args.data)
        s1_cfg["data"] = str(data_root / "stage1_detect" / "data.yaml")
        s2_cfg["data"] = str(data_root / "stage2_classify")
    if args.device:
        s1_cfg["device"] = args.device
        s2_cfg["device"] = args.device
    if args.variant:
        for c in (s1_cfg, s2_cfg):
            c["variant"] = args.variant
        s1_cfg["model"] = str(PROJECT_ROOT / "ultralytics" / "cfg" / "models" / "26" / f"yolo26{args.variant}.yaml")
        s1_cfg["pretrained_model"] = str(PROJECT_ROOT / f"yolo26{args.variant}.pt")
    for c, e, b, name_key in [
        (s1_cfg, args.epochs1, args.batch1, "stage1_name"),
        (s2_cfg, args.epochs2, args.batch2, "stage2_name"),
    ]:
        if e: c["epochs"] = e
        if b: c["batch"] = b
    if args.no_freeze: s1_cfg["freeze_epochs"] = 0
    if args.no_weather: s1_cfg["weather_aug"] = False
    if args.oversample is not None: s2_cfg["oversample"] = args.oversample
    if args.stage1_name: s1_cfg["name"] = args.stage1_name
    if args.stage2_name: s2_cfg["name"] = args.stage2_name

    # 总览
    print_banner("Two-Stage Training Pipeline")
    print_env()
    print(f"  Stage1: epochs={s1_cfg['epochs']} batch={s1_cfg['batch']} imgsz={s1_cfg['imgsz']} freeze={s1_cfg['freeze_epochs']}")
    print(f"  Stage2: epochs={s2_cfg['epochs']} batch={s2_cfg['batch']} imgsz={s2_cfg['imgsz']} oversample={s2_cfg['oversample']}")
    print(f"  Skip:   Stage1={args.skip_stage1}  Stage2={args.skip_stage2}")
    print("=" * 64)

    total_start = time.time()
    stage1_best = None

    # ---- Stage 1 ----
    if not args.skip_stage1:
        stage1_best = train_stage1(s1_cfg)
    else:
        print("[SKIP] Stage1")

    # ---- Stage 2 ----
    if not args.skip_stage2:
        train_stage2(s2_cfg, stage1_best)
    else:
        print("[SKIP] Stage2")

    total_elapsed = time.time() - total_start
    print_banner("Pipeline Complete")
    print(f"  Total: {int(total_elapsed // 3600)}h {int((total_elapsed % 3600) // 60)}m")
    print("=" * 64)


if __name__ == "__main__":
    main()
