"""
train_drowning.py - YOLO26 Drowning Detection Training Pipeline
===================================================================
Main training entry point for YOLO26-based drowning person detection.

Features:
  - YOLO26 model with 3-tier weight initialization (local → download → scratch)
  - Progressive layer freezing (freeze backbone, then unfreeze)
  - Seamless checkpoint resume (restores optimizer, EMA, scaler, epoch)
  - Class imbalance handling via cls_pw and optional Focal Loss
  - Multi-scale training for mixed-resolution dataset
  - Automatic visualization on training completion
  - NaN recovery and training stability safeguards

Usage:
    python train_drowning.py                          # Default training
    python train_drowning.py --resume                 # Resume from latest checkpoint
    python train_drowning.py --variant s --epochs 200 # Custom variant & epochs
    python train_drowning.py --cfg custom_config.yaml # YAML config override
    python train_drowning.py --no-freeze              # Disable layer freezing
    python train_drowning.py --no-test                # Skip test evaluation
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from config_drowning import parse_cli_args, build_config

try:
    from custom_plots import DrowningVisualizer
    _HAS_PLOTS = True
except ImportError:
    DrowningVisualizer = None
    _HAS_PLOTS = False


# ===========================================================================
#  Progressive Freeze Callback
# ===========================================================================

class ProgressiveFreezeCallback:
    """
    Implements progressive layer freezing strategy for YOLO training.

    Strategy:
        Stage 0 (epoch 0-4): Freeze backbone layers (first 10) to protect
                              pretrained weights. Only train neck + head.
        Stage 1 (epoch 5+):  Unfreeze all layers for full fine-tuning.

    Usage:
        cb = ProgressiveFreezeCallback(freeze_stages)
        model.add_callback("on_train_epoch_start", cb)
    """

    def __init__(self, freeze_stages):
        """
        Args:
            freeze_stages: list of dicts, each with:
                start_epoch (int): when this stage begins
                freeze (int|list|None): layers to freeze (int=N=range(N),
                       list=specific indices, None=unfreeze all)
                Example: [
                    {"start_epoch": 0, "freeze": 10},
                    {"start_epoch": 5, "freeze": None},
                ]
        """
        self.stages = sorted(freeze_stages, key=lambda s: s["start_epoch"])
        self.current_stage_idx = -1

    def __call__(self, trainer):
        """Called at on_train_epoch_start by ultralytics callback system."""
        current_epoch = trainer.epoch
        new_stage = None

        for stage in self.stages:
            if current_epoch >= stage["start_epoch"]:
                new_stage = stage
            else:
                break

        if new_stage is None:
            return

        new_idx = self.stages.index(new_stage)
        if new_idx == self.current_stage_idx:
            return  # same stage, nothing to change

        self.current_stage_idx = new_idx
        self._apply_freeze(trainer, new_stage["freeze"])

    def _apply_freeze(self, trainer, freeze_value):
        """Apply freeze/unfreeze to model parameters."""
        model = trainer.model
        # Handle DataParallel/DDP wrapping
        unwrapped = model.module if hasattr(model, "module") else model

        if freeze_value is None:
            # Unfreeze all
            for param in unwrapped.parameters():
                param.requires_grad = True
            trainable = sum(p.numel() for p in unwrapped.parameters() if p.requires_grad)
            total = sum(p.numel() for p in unwrapped.parameters())
            print(f"[FREEZE] Epoch {trainer.epoch}: All layers unfrozen "
                  f"({trainable:,}/{total:,} trainable params)")
            return

        if isinstance(freeze_value, int):
            freeze_list = list(range(freeze_value))
        else:
            freeze_list = list(freeze_value)

        # Build freeze layer name patterns
        # Ultralytics layer naming: model.0, model.1, ..., model.22, etc.
        always_frozen = [".dfl"]  # DFL always frozen
        freeze_patterns = [f"model.{i}." for i in freeze_list] + always_frozen

        frozen = 0
        total_params = 0
        for name, param in unwrapped.named_parameters():
            total_params += 1
            if any(pattern in name for pattern in freeze_patterns):
                param.requires_grad = False
                frozen += 1
            elif param.dtype.is_floating_point:
                param.requires_grad = True

        trainable = sum(p.numel() for p in unwrapped.parameters() if p.requires_grad)
        total = sum(p.numel() for p in unwrapped.parameters())
        print(f"[FREEZE] Epoch {trainer.epoch}: Frozen {frozen}/{total_params} param groups "
              f"(layers {freeze_list}), trainable: {trainable:,}/{total:,}")


# ===========================================================================
#  Model Initialization
# ===========================================================================

def setup_model(config):
    """
    Three-tier model initialization strategy.

    Tier 1: Load local pretrained weights (e.g., yolo26n.pt)
    Tier 2: Attempt framework download (may fail on custom forks)
    Tier 3: Train from scratch using model YAML config

    Note: yolov8x.pt CANNOT be loaded into yolo26 architecture
          (different backbone: C3k2 vs C2f, end2end head vs standard)
    """
    from ultralytics import YOLO

    variant = config.get("variant", "n")
    task = config.get("task", "detect")
    model = None
    weights_source = "unknown"

    # Classification: try cls-specific weights first
    if task == "classify":
        cls_pt = f"yolo26{variant}-cls.pt"
        if Path(cls_pt).exists():
            try:
                model = YOLO(cls_pt)
                weights_source = f"local: {cls_pt}"
                print(f"[MODEL] Loaded classification weights: {cls_pt}")
                return model, weights_source
            except:
                pass

    # Classification: use cls-specific YAML (detection weights incompatible)
    if task == "classify":
        cls_yaml = PROJECT_ROOT / "ultralytics" / "cfg" / "models" / "26" / f"yolo26-cls.yaml"
        if cls_yaml.exists():
            try:
                model = YOLO(str(cls_yaml), task="classify")
                weights_source = f"cls yaml ({variant})"
                print(f"[MODEL] Created classification model from: {cls_yaml}")
                return model, weights_source
            except Exception as e:
                print(f"[WARN] Classification YAML failed: {e}")

    # Tier 1: Local pretrained weights
    pretrained_path = Path(config.get("pretrained_model", ""))
    if pretrained_path.exists():
        try:
            model = YOLO(str(pretrained_path), task=task)
            weights_source = f"local: {pretrained_path.name}"
            print(f"[MODEL] Loaded local pretrained weights: {pretrained_path}")
        except Exception as e:
            print(f"[WARN] Local weights failed: {e}")

    # Tier 2: Framework download
    if model is None:
        pt_name = f"yolo26{variant}.pt"
        try:
            model = YOLO(pt_name, task=task)
            weights_source = f"downloaded: {pt_name}"
            print(f"[MODEL] Downloaded weights: {pt_name}")
        except Exception as e:
            print(f"[INFO] Download failed (expected for custom fork): {e}")

    # Tier 3: Scratch from YAML
    if model is None:
        model_cfg = config.get("model", "")
        if not model_cfg or not Path(model_cfg).exists():
            model_cfg = str(PROJECT_ROOT / "ultralytics" / "cfg" / "models" / "26" / f"yolo26{variant}.yaml")
        model = YOLO(str(model_cfg), task=task)
        weights_source = f"scratch from yaml ({variant})"
        print(f"[MODEL] Training from scratch: {model_cfg}")

    print(f"[MODEL] Weight source: {weights_source}")
    # Force task override — yolo26n.pt defaults to detect even with task= arg
    if model is not None and hasattr(model, 'task'):
        model.task = task
        print(f"[MODEL] Task forced to: {task}")
    return model, weights_source


# ===========================================================================
#  Training Argument Builder
# ===========================================================================

def build_train_args(config):
    """
    Extract ultralytics-compatible training arguments from our config dict.
    Only includes keys recognized by model.train().
    """
    return {
        # Core
        "data": config["data"],
        "epochs": config["epochs"],
        "batch": config["batch"],
        "imgsz": config["imgsz"],
        "device": config["device"],
        "workers": config.get("workers", 4),

        # Optimizer
        "optimizer": config["optimizer"],
        "lr0": config["lr0"],
        "lrf": config["lrf"],
        "momentum": config["momentum"],
        "weight_decay": config["weight_decay"],
        "cos_lr": config.get("cos_lr", True),
        "warmup_epochs": config["warmup_epochs"],

        # Class imbalance (removed cls_pw — not in standard ultralytics)

        # Loss weights
        "box": config["box"],
        "cls": config["cls"],
        "dfl": config["dfl"],

        # Augmentation
        "mosaic": config["mosaic"],
        "mixup": config["mixup"],
        "close_mosaic": config["close_mosaic"],
        "hsv_h": config["hsv_h"],
        "hsv_s": config["hsv_s"],
        "hsv_v": config["hsv_v"],
        "degrees": config["degrees"],
        "translate": config["translate"],
        "scale": config["scale"],
        "fliplr": config["fliplr"],
        "multi_scale": config["multi_scale"],

        # Mixed precision
        "amp": config["amp"],

        # Early stopping & saving
        "patience": config["patience"],
        "save_period": config["save_period"],

        # Reproducibility
        "seed": config["seed"],
        "deterministic": config.get("deterministic", True),

        # Output
        "project": config["project"],
        "name": config["name"],

        # Resume
        "resume": config.get("resume", False),
        "pretrained": config.get("pretrained", True),

        # Misc
        "plots": True,
        "val": config.get("val", True),
        "save": True,
        "exist_ok": False,
    }


# ===========================================================================
#  Environment Check
# ===========================================================================

def check_environment(config):
    """Print environment info and validate GPU availability."""
    print("=" * 60)
    print("  Environment Check")
    print("=" * 60)
    print(f"  Python:      {sys.version.split()[0]}")
    print(f"  PyTorch:     {torch.__version__}")
    print(f"  CUDA:        {torch.version.cuda or 'N/A'}")

    if torch.cuda.is_available():
        device_id = config.get("device", 0)
        if device_id is not None and device_id >= 0 and device_id < torch.cuda.device_count():
            props = torch.cuda.get_device_properties(device_id)
            print(f"  GPU:         {props.name}")
            print(f"  VRAM:        {props.total_memory / (1024**3):.1f} GB")
            print(f"  Compute:     {props.major}.{props.minor}")
    else:
        print("  GPU:         None (CPU-only mode)")

    print("=" * 60)


# ===========================================================================
#  Post-Training Callback
# ===========================================================================

def create_post_training_callback(visualizer, config, weights_source):
    """Returns a callback function to generate plots after training."""

    def on_train_end(trainer):
        print("\n" + "=" * 60)
        print("  Generating Visualization Reports...")
        print("=" * 60)

        save_dir = Path(trainer.save_dir)
        visualizer.save_dir = save_dir

        results_csv = save_dir / "results.csv"
        best_pt = save_dir / "weights" / "best.pt"

        if results_csv.exists():
            visualizer.create_training_plots(
                results_csv=str(results_csv),
                best_pt=str(best_pt) if best_pt.exists() else None,
                config=config,
                weights_source=weights_source,
            )
        else:
            print("[WARN] results.csv not found, skipping plots")

        # Save config snapshot
        config_snapshot = {k: str(v) if isinstance(v, Path) else v
                           for k, v in config.items()}
        config_path = save_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(config_snapshot, f, indent=2, default=str)
        print(f"[CONFIG] Saved to: {config_path}")

        print("=" * 60)
        print(f"  Training Complete! Output: {save_dir}")
        print("=" * 60)

    return on_train_end


# ===========================================================================
#  Main
# ===========================================================================


# ===========================================================================
#  Single Stage Runner
# ===========================================================================
def run_stage(config, stage_label="", test=True):
    """Run a single training stage. Returns save_dir Path."""
    start_time = time.time()

    # 1. Initialize model
    print("\n" + "=" * 60)
    print(f"  Initializing Model ({stage_label})...")
    print("=" * 60)
    model, weights_source = setup_model(config)

    # 2. Build training arguments
    train_args = build_train_args(config)

    # 3. Register progressive freeze callback
    freeze_stages = config.get("freeze_stages", [])
    if freeze_stages and not config.get("resume", False):
        freeze_cb = ProgressiveFreezeCallback(freeze_stages)
        model.add_callback("on_train_epoch_start", freeze_cb)
        print(f"[FREEZE] Progressive freezing: {len(freeze_stages)} stage(s)")

    # 4. Register post-training visualization (optional)
    if _HAS_PLOTS:
        if config.get("stage") == 2:
            class_names = ["drowning", "swimming"]
        else:
            class_names = ["person_in_water"]
        visualizer = DrowningVisualizer(
            save_dir=None, class_names=class_names, dataset_path=config["data"],
        )
        post_cb = create_post_training_callback(visualizer, config, weights_source)
        model.add_callback("on_train_end", post_cb)

    # 5. Start training
    print("\n" + "=" * 60)
    print(f"  {stage_label} Training")
    print("=" * 60)
    print(f"  Epochs: {train_args['epochs']}, Batch: {train_args['batch']}, ImgSz: {train_args['imgsz']}")
    print(f"  Task: {config.get('task', 'detect')}")
    print("=" * 60)

    try:
        results = model.train(**train_args)
    except torch.cuda.OutOfMemoryError:
        print("[ERROR] CUDA OOM! Try --batch 8")
        sys.exit(1)

    save_dir = Path(model.trainer.save_dir) if hasattr(model, 'trainer') and model.trainer else \
               Path(config["project"]) / config["name"]

    # Test evaluation
    if test and hasattr(model, 'trainer') and model.trainer:
        try:
            best_path = save_dir / "weights" / "best.pt"
            if best_path.exists():
                model.val(data=config["data"], split="test", plots=True)
        except Exception as e:
            print(f"[WARN] Test eval failed: {e}")

    elapsed = time.time() - start_time
    print(f"\n[{stage_label}] Done in {elapsed/3600:.1f}h")
    return save_dir


_DATASETS_DIR = PROJECT_ROOT / "datasets"
if not _DATASETS_DIR.exists():
    _DATASETS_DIR = PROJECT_ROOT.parent / "picture_process"
STAGE1_DATA_DIR = _DATASETS_DIR / "stage1_pure_dataset"
STAGE2_DATA_DIR = _DATASETS_DIR / "stage2_cls_dataset"


def main():
    args = parse_cli_args()
    config = build_config(args)
    check_environment(config)

    stage = config.get("stage", 1)
    do_test = config.get("test", True)

    if stage == 0:
        # ── Pipeline: Stage1 → Stage2 ──
        print("\n" + "=" * 60)
        print("  PIPELINE: Stage1 (detect) → Stage2 (classify)")
        print("=" * 60)

        import copy
        s1 = copy.deepcopy(config)
        s1.update(stage=1, task="detect", data=str(STAGE1_DATA_DIR / "data.yaml"),
                  name="yolo26n_stage1_pure_v1", project=str(PROJECT_ROOT / "runs" / "stage1_detect"),
                  imgsz=640)
        s1_dir = run_stage(s1, "Stage1", test=do_test)

        s2 = copy.deepcopy(config)
        s2.update(stage=2, task="classify", data=str(STAGE2_DATA_DIR),
                  name="yolo26n_stage2_cls_v1", project=str(PROJECT_ROOT / "runs" / "stage2_classify"),
                  imgsz=256, cls_pw=0, freeze_stages=[], multi_scale=0, mosaic=0, mixup=0)
        s2_dir = run_stage(s2, "Stage2", test=do_test)

        print("\n" + "=" * 60)
        print("  PIPELINE COMPLETE")
        print(f"  Stage1: {s1_dir}")
        print(f"  Stage2: {s2_dir}")
        print("=" * 60)
    else:
        run_stage(config, f"Stage{stage}", test=do_test)


if __name__ == "__main__":
    main()
