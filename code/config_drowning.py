"""
config_drowning.py - YOLO26 Drowning Detection Training Configuration
============================================================================
Single source of truth for all training hyperparameters.
Supports three-layer override: Defaults < YAML file < CLI arguments.

Usage:
    from config_drowning import DROWNING_CONFIG, parse_cli_args, build_config

    args = parse_cli_args()
    config = build_config(args)
"""

import argparse
import sys
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Progressive Freeze Callback
# ---------------------------------------------------------------------------
class ProgressiveFreezeCallback:
    """
    YOLO 训练回调: 按 epoch 渐进式冻结/解冻 backbone 层。

    freeze_stages 格式:
        [
            {"start_epoch": 0, "freeze": 10},   # epoch 0-4: 冻结前 10 层
            {"start_epoch": 5, "freeze": None},  # epoch 5+: 全解冻
        ]

    freeze=None 表示全部解冻; freeze=0 表示不冻结任何层。
    """

    def __init__(self, freeze_stages):
        self.freeze_stages = freeze_stages
        self._current_freeze = None

    def __call__(self, trainer):
        """on_train_epoch_start 回调入口"""
        epoch = trainer.epoch

        # 找到当前 epoch 应该使用的冻结配置
        active_freeze = None
        for stage in self.freeze_stages:
            if epoch >= stage["start_epoch"]:
                active_freeze = stage["freeze"]

        # 如果冻结配置没变, 不重复操作
        if active_freeze == self._current_freeze:
            return

        self._current_freeze = active_freeze

        if active_freeze is None or active_freeze == 0:
            # 全解冻
            for param in trainer.model.parameters():
                param.requires_grad = True
            print(f"[FREEZE] Epoch {epoch}: 全部解冻 ✓")
        else:
            # 冻结前 N 层
            model = trainer.model
            # 获取 backbone 的层 (按参数分组)
            frozen = 0
            for i, (name, param) in enumerate(model.named_parameters()):
                if frozen < active_freeze:
                    param.requires_grad = False
                    frozen += 1
                else:
                    param.requires_grad = True

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"[FREEZE] Epoch {epoch}: 冻结前 {frozen} 层, "
                  f"可训练 {trainable}/{total} 参数 "
                  f"({trainable/total*100:.1f}%)")



# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
# Auto-detect dataset location: server uses datasets/, local uses picture_process/
_DATASETS_DIR = PROJECT_ROOT / "datasets"
if not _DATASETS_DIR.exists():
    _DATASETS_DIR = PROJECT_ROOT.parent / "picture_process"
STAGE1_DATA = _DATASETS_DIR / "stage1_pure_dataset"
STAGE2_DATA = _DATASETS_DIR / "stage2_cls_dataset"

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DROWNING_CONFIG = {
    # ---- Task ----
    "stage": 1,                            # 1=Stage1 detection, 2=Stage2 classification
    "task": "detect",                      # ultralytics task: detect / classify

    # ---- Model ----
    "model": str(PROJECT_ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26n.yaml"),
    "pretrained_model": str(PROJECT_ROOT / "yolo26n.pt"),
    "variant": "n",                        # YOLO26 variant: n, s, m, l, x

    # ---- Data ----
    "data": str(STAGE1_DATA / "data.yaml"),

    # ---- Training Core ----
    "epochs": 300,
    "batch": 16,
    "imgsz": 640,
    "device": 0,                           # GPU device ID; None for CPU
    "workers": 4,                          # dataloader workers (lower for laptop)

    # ---- Optimizer ----
    "optimizer": "AdamW",
    "lr0": 0.001,                          # AdamW: 1e-3; SGD: 1e-2
    "lrf": 0.01,                           # final_lr = lr0 * lrf = 1e-5
    "momentum": 0.9,                       # AdamW beta1
    "weight_decay": 0.0005,
    "cos_lr": True,                        # cosine LR schedule
    "warmup_epochs": 5,                    # extended warmup for scratch training

    # ---- Class Imbalance Handling ----
    "cls_pw": 0.5,                         # 0=disable, 0.5=sqrt, 1.0=inverse freq

    # ---- Loss Weights ----
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,

    # ---- Data Augmentation ----
    "mosaic": 1.0,
    "mixup": 0.1,                          # light mixup for rare classes
    "close_mosaic": 15,                    # disable mosaic in last 15 epochs
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "fliplr": 0.5,
    "multi_scale": 0.5,                    # +/-50% multi-scale

    # ---- Mixed Precision ----
    "amp": True,

    # ---- Early Stopping & Checkpoints ----
    "patience": 0,                         # 0 = disable early stopping
    "save_period": 10,                     # save checkpoint every N epochs

    # ---- Layer Freeze (progressive unfreeze) ----
    "freeze_stages": [
        {"start_epoch": 0, "freeze": 10},   # epoch 0-4: freeze first 10 backbone layers
        {"start_epoch": 5, "freeze": None}, # epoch 5+: unfreeze all
    ],

    # ---- Reproducibility ----
    "seed": 42,
    "deterministic": True,

    # ---- Output ----
    "project": str(PROJECT_ROOT / "runs" / "drowning_detection"),
    "name": "yolo26n_drowning_v1",

    # ---- Validation & Testing ----
    "val": True,
    "test": True,

    # ---- Resume ----
    "resume": False,
    "pretrained": True,
}


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments. All override defaults when provided."""
    parser = argparse.ArgumentParser(
        description="YOLO26 Drowning Detection Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Stage Selection ----
    parser.add_argument("--stage", type=int, default=None,
                        choices=[0, 1, 2],
                        help="0=pipeline (Stage1→Stage2), 1=Stage1 detection, 2=Stage2 classification")

    # ---- Model ----
    parser.add_argument("--model", type=str, default=None,
                        help="Model path (.pt or .yaml)")
    parser.add_argument("--variant", type=str, default="n",
                        choices=["n", "s", "m", "l", "x"],
                        help="YOLO26 variant")

    # ---- Training Core ----
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default=None,
                        help="GPU device (0, 1, 'cpu')")
    parser.add_argument("--workers", type=int, default=None)

    # ---- Optimizer ----
    parser.add_argument("--optimizer", type=str, default=None,
                        choices=["SGD", "Adam", "AdamW", "RMSProp", "auto"])
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--lrf", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None, dest="weight_decay")

    # ---- Class Imbalance ----
    parser.add_argument("--cls-pw", type=float, default=None, dest="cls_pw",
                        help="Class weight power (0=off, 0.5=sqrt, 1.0=inv-freq)")

    # ---- State Control ----
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume from latest checkpoint")
    parser.add_argument("--no-val", action="store_false", default=True, dest="val",
                        help="Skip validation")
    parser.add_argument("--no-test", action="store_false", default=True, dest="test",
                        help="Skip test-set evaluation")
    parser.add_argument("--no-freeze", action="store_true", default=False,
                        help="Disable progressive layer freezing")
    parser.add_argument("--no-amp", action="store_false", default=True, dest="amp",
                        help="Disable AMP mixed precision")

    # ---- Config File ----
    parser.add_argument("--cfg", type=str, default=None,
                        help="YAML config file (CLI overrides YAML values)")

    # ---- Output ----
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    """
    Merge configuration sources with priority: defaults < YAML file < CLI args.

    Args:
        args: parsed CLI arguments from parse_cli_args()

    Returns:
        dict: final merged configuration
    """
    import yaml

    config = DROWNING_CONFIG.copy()

    # ---- Layer 2: YAML config file ----
    if args.cfg:
        cfg_path = Path(args.cfg)
        if not cfg_path.is_absolute():
            cfg_path = PROJECT_ROOT / cfg_path
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f) or {}
            config.update(yaml_config)
            print(f"[CONFIG] Loaded YAML overrides from: {cfg_path}")
        else:
            print(f"[WARN] Config file not found: {cfg_path}")

    # ---- Layer 3: CLI overrides (only non-None values) ----
    cli_map = {
        "stage": args.stage,
        "model": args.model,
        "variant": args.variant if args.variant != "n" else None,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "cls_pw": args.cls_pw,
        "resume": args.resume if args.resume else None,
        "val": args.val if not args.val else None,
        "test": args.test if not args.test else None,
        "amp": args.amp if not args.amp else None,
        "project": args.project,
        "name": args.name,
    }
    for key, value in cli_map.items():
        if value is not None:
            config[key] = value

    # ---- Post-processing ----
    # Stage selection: adjust data, task, and model paths
    stage = config.get("stage", 1)
    if stage == 2:
        config["task"] = "classify"
        config["data"] = str(STAGE2_DATA)
        config["cls_pw"] = 0  # classification doesn't use cls_pw
        config["name"] = "yolo26n_stage2_cls_v1"
        config["project"] = str(PROJECT_ROOT / "runs" / "stage2_classify")
        config["freeze_stages"] = []  # no freeze for classification
        config["imgsz"] = 256         # crops are 256x256
        config["multi_scale"] = 0     # classification no multi-scale
        config["mosaic"] = 0          # classification no mosaic
        config["mixup"] = 0
        config["patience"] = 0        # disable early stop
    else:
        config["task"] = "detect"
        config["data"] = str(STAGE1_DATA / "data.yaml")
        config["name"] = "yolo26n_stage1_pure_v1"
        config["project"] = str(PROJECT_ROOT / "runs" / "stage1_detect")

    # Disable freeze stages if --no-freeze
    if args.no_freeze:
        config["freeze_stages"] = []

    # Handle variant: update model paths
    if args.variant and args.variant != config.get("variant"):
        config["variant"] = args.variant

    # Auto-compute model paths based on variant
    variant = config.get("variant", "n")
    model_yaml = PROJECT_ROOT / "ultralytics" / "cfg" / "models" / "26" / f"yolo26{variant}.yaml"
    config["model"] = str(model_yaml)
    config["pretrained_model"] = str(PROJECT_ROOT / f"yolo26{variant}.pt")

    # ---- Validation ----
    _validate_config(config)

    return config


def _validate_config(config: dict) -> None:
    """Validate critical config values."""
    model_path = Path(config["model"])
    data_path = Path(config["data"])

    if not data_path.exists():
        print(f"[WARN] Dataset config not found: {data_path}")
        print(f"       Expected at: {data_path}")

    if config["batch"] <= 0:
        raise ValueError(f"batch must be > 0, got {config['batch']}")

    if config["epochs"] <= 0:
        raise ValueError(f"epochs must be > 0, got {config['epochs']}")

    if config["cls_pw"] < 0:
        print("[WARN] cls_pw < 0, disabling class weighting")

    print(f"[CONFIG] Stage: {config.get('stage', 1)}, Task: {config.get('task', 'detect')}")
    print(f"[CONFIG] Model: {config['model']}")
    print(f"[CONFIG] Data:  {config['data']}")
    print(f"[CONFIG] Epochs: {config['epochs']}, Batch: {config['batch']}, ImgSz: {config['imgsz']}")
    print(f"[CONFIG] Optimizer: {config['optimizer']}, LR0: {config['lr0']}, LRF: {config['lrf']}")
    print(f"[CONFIG] cls_pw: {config['cls_pw']}, AMP: {config['amp']}, Resume: {config['resume']}")
    print(f"[CONFIG] Freeze stages: {len(config.get('freeze_stages', []))} stage(s)")


def get_compare_configs() -> dict:
    """
    Generate configuration variants for ablation study (compare_optimizations.py).
    Returns a dict of {experiment_name: config_overrides}.
    """
    base = DROWNING_CONFIG.copy()
    base["epochs"] = 50  # quick comparison

    return {
        "Baseline": {
            **base,
            "name": "00_baseline",
            "freeze_stages": [],
        },
        "Freeze": {
            **base,
            "name": "01_freeze",
            "freeze_stages": [
                {"start_epoch": 0, "freeze": 10},
                {"start_epoch": 5, "freeze": None},
            ],
        },
        "FocalLoss": {
            **base,
            "name": "02_focal_loss",
            "freeze_stages": [],
            "cls_pw": 1.0,  # stronger weighting simulates focal effect
        },
        "LabelSmooth": {
            **base,
            "name": "03_label_smooth",
            "freeze_stages": [],
        },
        "OneCycleLR": {
            **base,
            "name": "04_one_cycle_lr",
            "freeze_stages": [],
            "cos_lr": True,
        },
        "TTA": {
            **base,
            "name": "05_tta",
            "freeze_stages": [],
        },
        "ALL": {
            **base,
            "name": "06_all_combined",
            "cos_lr": True,
            "freeze_stages": [
                {"start_epoch": 0, "freeze": 10},
                {"start_epoch": 5, "freeze": None},
            ],
            "cls_pw": 1.0,
        },
    }


# ---------------------------------------------------------------------------
# Standalone usage: python config_drowning.py --help
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_cli_args()
    config = build_config(args)
    print("\n" + "=" * 60)
    print("Final merged configuration:")
    print("=" * 60)
    for k, v in sorted(config.items()):
        print(f"  {k:25s} = {v}")
