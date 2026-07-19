"""
train_surveil_stage2.py - Stage2 细粒度分类器训练 (nc=2)
=========================================================
类别: 0=drowning(溺水)  1=swimming(游泳)

对应已定稿方案:
  - radiant-pulse-curie.md: 两阶段+3分支, Stage2 对 person_in_water 裁剪区判「是否溺水」
  - 优化器 AdamW 方案A
  - 整合后 drowning:swimming ≈ 50.8:49.2 已基本平衡 -> 默认关闭过采样

输入数据: picture_process/unified_surveillance_v1/stage2_classify/ (train/val/test/{drowning,swimming})

Usage (服务器单卡4090):
    python train_surveil_stage2.py
    python train_surveil_stage2.py --epochs 200 --batch 64 --device 0
    python train_surveil_stage2.py --oversample 1.2   # 若首轮 drowning 召回不足
    python train_surveil_stage2.py --data /path/to/stage2_classify
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
from config_surveil import SURVEIL_STAGE2_CONFIG, build_train_args

STAGE2_CLASS_NAMES = ["drowning", "swimming"]


def oversample_drowning(data_dir: str, ratio: float = 1.0, seed: int = 42):
    """
    对 drowning 类做过采样(写回 train/drowning，便于 resume/复现)。
    ratio=1.0 表示不操作(已平衡)。ratio>1.0 时 drowing 复制到 swimming*ratio 张。
    """
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


def setup_model(config):
    """分类权重三级加载: 本地 -> 下载 -> 从零。"""
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


def parse_args():
    parser = argparse.ArgumentParser(description="Stage2 细粒度分类器训练 (nc=2)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--variant", type=str, default=None, choices=["n", "s", "m"])
    parser.add_argument("--data", type=str, default=None, help="stage2_classify 目录")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--oversample", type=float, default=None, help="drowning 过采样比例(相对swimming)")
    parser.add_argument("--resume", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    config = SURVEIL_STAGE2_CONFIG.copy()
    if args.data: config["data"] = args.data
    if args.epochs: config["epochs"] = args.epochs
    if args.batch: config["batch"] = args.batch
    if args.device: config["device"] = args.device
    if args.variant: config["variant"] = args.variant
    if args.oversample is not None: config["oversample"] = args.oversample
    if args.resume: config["resume"] = True

    print("=" * 64)
    print("  Stage2 细粒度分类器训练 (drowning vs swimming)")
    print("=" * 64)
    print(f"  Python:  {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU:     {props.name}")
    print(f"  Data:    {config['data']}")
    print("=" * 64)

    # 可选过采样
    oversample_drowning(config["data"], ratio=config.get("oversample", 1.0))

    model = setup_model(config)

    train_args = build_train_args(config, task="classify")
    train_args.update({
        "val": True,
        "plots": True,
        "save": True,
        "exist_ok": True,
        "pretrained": True,
        "resume": config.get("resume", False),
    })

    print("\n" + "=" * 64)
    print("  开始训练")
    print("=" * 64)
    print(f"  类别:   {STAGE2_CLASS_NAMES}")
    print(f"  Epochs: {train_args['epochs']}  Batch: {train_args['batch']}  ImgSz: {train_args['imgsz']}")
    print(f"  Optim:  {config['optimizer']}  lr0={config['lr0']}  wd={config['weight_decay']}")
    print(f"  Model:  yolo26{config.get('variant','s')}-cls")
    print("=" * 64)

    start = time.time()
    try:
        model.train(**train_args)
    except torch.cuda.OutOfMemoryError:
        print("\n[ERROR] CUDA OOM! 尝试 --batch 32")
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
