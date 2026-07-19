"""
config_surveil.py - 监控视角溺水/游泳识别 训练配置（新一版 YOLOv26）
========================================================================

依据已定稿方案：
  - radiant-pulse-curie.md  (训练方法学: 两阶段+3分支 / AdamW方案A / 在线天气增强)
  - swift-vortex-einstein.md (数据集整合: archive+internet+new+internet_v2)

相对于旧 config_drowning.py 的修正：
  1. 类别改为 nc=3(Stage1) / nc=2(Stage2)，数据指向新整合集 unified_surveillance_v1
  2. 冻结回调改为「按 backbone 模块冻结前 N 层 + 始终冻结 .dfl」
     —— 旧 ProgressiveFreezeCallback 按参数出现顺序冻结前10个参数组，会误冻 neck/head
  3. cls_pw 失效(ultralytics 不支持) -> 改用 cls=2.0 + copy_paste=0.2 + 已平衡的 50/50 数据集
  4. 优化器锁定 AdamW 方案A (lr0=0.0015, cos_lr, warmup=5, wd=0.0005)
  5. 在线天气增强开关 weather_aug / weather_p

数据路径自动解析优先级：
  1. 环境变量 SURVEIL_DATA_ROOT
  2. <PROJECT_ROOT>/datasets/unified_surveillance_v1   (服务器软链)
  3. <PROJECT_ROOT>/../picture_process/unified_surveillance_v1 (本地)

Usage:
  from config_surveil import SURVEIL_STAGE1_CONFIG, SURVEIL_STAGE2_CONFIG, SurveilFreezeCallback
"""

import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 数据集根目录自动检测
# ---------------------------------------------------------------------------
def _find_data_root() -> Path:
    env = os.environ.get("SURVEIL_DATA_ROOT")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(PROJECT_ROOT / "datasets" / "unified_surveillance_v1")
    candidates.append(PROJECT_ROOT.parent / "picture_process" / "unified_surveillance_v1")
    for c in candidates:
        if c and (c / "stage1_detect" / "data.yaml").exists():
            return c
    # 都没命中则返回本地默认(便于报错时给出明确预期路径)
    return PROJECT_ROOT.parent / "picture_process" / "unified_surveillance_v1"


DATA_ROOT = _find_data_root()
STAGE1_YAML = str(DATA_ROOT / "stage1_detect" / "data.yaml")
STAGE2_DIR = str(DATA_ROOT / "stage2_classify")


# ---------------------------------------------------------------------------
# Stage1 (检测, nc=2) 配置
# ---------------------------------------------------------------------------
SURVEIL_STAGE1_CONFIG = {
    # ---- Model ----
    "variant": "s",
    "model": str(PROJECT_ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26s.yaml"),
    "pretrained_model": str(PROJECT_ROOT / "yolo26s.pt"),

    # ---- Data ----
    "data": STAGE1_YAML,

    # ---- Training Core ----
    "epochs": 200,
    "batch": 16,
    "imgsz": 640,
    "device": 0,
    "workers": 8,

    # ---- Optimizer (方案A: AdamW) ----
    "optimizer": "AdamW",
    "lr0": 0.0015,            # AdamW 推荐 1e-3 ~ 2e-3
    "lrf": 0.01,              # final_lr = lr0 * lrf = 1.5e-5
    "weight_decay": 0.0005,
    "cos_lr": True,
    "warmup_epochs": 5,

    # ---- Loss Weights ----
    # cls_pw 不被 ultralytics 支持；用 cls=2.0 提高分类关注度 + copy_paste 提召回
    "box": 7.5,
    "cls": 2.0,
    "dfl": 1.5,

    # ---- Data Augmentation (检测) ----
    "mosaic": 0.6,            # 比旧 0.3 略升，兼顾目标完整性与小目标
    "mixup": 0.0,             # 禁用 mixup，避免模糊水中人特征
    "copy_paste": 0.2,        # 稀缺/小目标粘贴增强，提 person_in_water 召回
    "close_mosaic": 20,       # 最后 20 epoch 关闭 mosaic 稳定收敛
    "hsv_h": 0.01,            # 减少色调变化(水面颜色关键)
    "hsv_s": 0.5,
    "hsv_v": 0.3,
    "degrees": 5.0,
    "translate": 0.1,
    "scale": 0.3,             # 小目标不宜大幅缩放
    "fliplr": 0.5,
    "multi_scale": 0.3,

    # ---- Mixed Precision ----
    "amp": True,

    # ---- Progressive Freeze (修正版) ----
    "freeze_epochs": 5,       # 前 5 epoch 冻结 backbone 前 10 层
    "freeze_n": 10,           # 冻结 backbone 前 10 个顶层模块
    # 注: 始终冻结 .dfl（见 SurveilFreezeCallback）

    # ---- Weather Augmentation (在线, 可选) ----
    "weather_aug": True,      # 是否启用在线天气增强回调
    "weather_p": 0.3,         # 每个 batch 施加天气增强的概率

    # ---- Early Stopping & Saving ----
    "patience": 50,
    "save_period": 10,

    # ---- Output ----
    "project": str(PROJECT_ROOT / "runs" / "surveil_stage1"),
    "name": "yolo26s_surveil_stage1_v1",
}


# ---------------------------------------------------------------------------
# Stage2 (分类, nc=2) 配置
# ---------------------------------------------------------------------------
SURVEIL_STAGE2_CONFIG = {
    "variant": "s",
    "pretrained_model": "yolo26s-cls.pt",
    "data": STAGE2_DIR,

    "epochs": 150,
    "imgsz": 256,             # 裁剪输入 256x256
    "batch": 64,              # 分类任务显存占用小，大 batch
    "device": 0,
    "workers": 8,

    # ---- Optimizer (方案A: AdamW) ----
    "optimizer": "AdamW",
    "lr0": 0.001,             # 分类 lr0 略低于检测
    "lrf": 0.01,
    "weight_decay": 0.0005,
    "cos_lr": True,
    "warmup_epochs": 3,

    "amp": True,

    # ---- 类别平衡 ----
    # 整合后 drowning:swimming ≈ 50.8:49.2，已基本平衡 -> 默认关闭过采样(1.0)
    # 若首轮后 drowning 召回不足，可调大 --oversample (如 1.2)
    "oversample": 1.0,

    # 分类不启用在线天气增强(离线增强子目录已覆盖雨/雾/夜/亮度)
    "weather_aug": False,

    "patience": 30,
    "save_period": 10,

    "project": str(PROJECT_ROOT / "runs" / "surveil_stage2"),
    "name": "yolo26s_cls_surveil_stage2_v1",
}


# ---------------------------------------------------------------------------
# 修正版渐进冻结回调
# ---------------------------------------------------------------------------
class SurveilFreezeCallback:
    """
    按 backbone 模块（而非参数顺序）冻结前 N 层 + 始终冻结 .dfl。

    旧 ProgressiveFreezeCallback 遍历 model.named_parameters() 按出现顺序冻结前 N 个
    参数张量，容易误冻 neck/head 的参数。本实现改为：
      - 找到 ultralytics DetectionModel 的 nn.Sequential 主干 (model.model)
      - 冻结其前 freeze_n 个顶层模块（backbone 浅层特征）
      - 在 freeze_epochs 之后逐步解冻所有层（除 .dfl）
      - 整个训练过程中，名字含 'dfl' 的参数始终 requires_grad=False
    """

    def __init__(self, freeze_epochs: int = 5, freeze_n: int = 10):
        self.freeze_epochs = freeze_epochs
        self.freeze_n = freeze_n
        self._applied_freeze = None   # 缓存上一次状态，避免重复操作

    @staticmethod
    def _get_backbone_sequential(model):
        """从 ultralytics model 中取出主干的 nn.Sequential。"""
        m = model
        # DetectionModel -> .model 是 nn.Sequential(backbone, neck, head)
        for _ in range(3):  # 最多下钻 3 层，兼容不同封装
            if isinstance(m, torch.nn.Sequential):
                return m
            if hasattr(m, "model"):
                m = m.model
            else:
                break
        return None

    def __call__(self, trainer):
        epoch = trainer.epoch
        should_freeze = epoch < self.freeze_epochs

        # 状态未变化则跳过
        if should_freeze == self._applied_freeze:
            self._freeze_dfl(trainer.model)
            return
        self._applied_freeze = should_freeze

        seq = self._get_backbone_sequential(trainer.model)
        if should_freeze and seq is not None:
            # 冻结 backbone 前 freeze_n 个顶层模块，其余可训练
            for idx, child in enumerate(seq):
                grad = idx >= self.freeze_n
                for p in child.parameters():
                    p.requires_grad = grad
            trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in trainer.model.parameters())
            print(f"[FREEZE] Epoch {epoch}: 冻结 backbone 前 {self.freeze_n} 层, "
                  f"可训练 {trainable/total*100:.1f}%")
        else:
            # 全解冻
            for p in trainer.model.parameters():
                p.requires_grad = True
            print(f"[FREEZE] Epoch {epoch}: 全部解冻 ✓")

        # 始终冻结 DFL
        self._freeze_dfl(trainer.model)

    @staticmethod
    def _freeze_dfl(model):
        for name, p in model.named_parameters():
            if "dfl" in name.lower():
                p.requires_grad = False


# ---------------------------------------------------------------------------
# 将 config dict 转为 ultralytics train() 支持的参数
# ---------------------------------------------------------------------------
# 仅用于本脚本控制逻辑、不传给 model.train 的键
_CTRL_KEYS = {
    "variant", "model", "pretrained_model", "freeze_epochs", "freeze_n",
    "weather_aug", "weather_p", "oversample",
}

# 不同 task 下 ultralytics 接受的训练参数白名单
# （分类任务不支持 box/cls/dfl/mosaic/mixup/copy_paste/hsv/... 等检测专属参数）
_DETECT_KEYS = {
    "data", "epochs", "batch", "imgsz", "device", "workers",
    "optimizer", "lr0", "lrf", "weight_decay", "cos_lr", "warmup_epochs",
    "box", "cls", "dfl", "mosaic", "mixup", "copy_paste", "close_mosaic",
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "fliplr",
    "multi_scale", "amp", "patience", "save_period", "project", "name",
}
_CLASSIFY_KEYS = {
    "data", "epochs", "batch", "imgsz", "device", "workers",
    "optimizer", "lr0", "lrf", "weight_decay", "cos_lr", "warmup_epochs",
    "amp", "patience", "save_period", "project", "name",
}


def build_train_args(config: dict, task: str = "detect") -> dict:
    """按 task 过滤出 ultralytics model.train() 接受的参数。"""
    allowed = _CLASSIFY_KEYS if task == "classify" else _DETECT_KEYS
    return {k: v for k, v in config.items() if k in allowed}


def resolve_data_yaml(stage: int = 1) -> str:
    return STAGE1_YAML if stage == 1 else STAGE2_DIR


if __name__ == "__main__":
    print("DATA_ROOT   :", DATA_ROOT)
    print("STAGE1 yaml :", STAGE1_YAML, "| exists:", Path(STAGE1_YAML).exists())
    print("STAGE2 dir  :", STAGE2_DIR, "| exists:", Path(STAGE2_DIR).exists())
    print("\n[Stage1] epochs=%d batch=%d imgsz=%d lr0=%s optimizer=%s" % (
        SURVEIL_STAGE1_CONFIG["epochs"], SURVEIL_STAGE1_CONFIG["batch"],
        SURVEIL_STAGE1_CONFIG["imgsz"], SURVEIL_STAGE1_CONFIG["lr0"],
        SURVEIL_STAGE1_CONFIG["optimizer"]))
    print("[Stage2] epochs=%d batch=%d imgsz=%d lr0=%s" % (
        SURVEIL_STAGE2_CONFIG["epochs"], SURVEIL_STAGE2_CONFIG["batch"],
        SURVEIL_STAGE2_CONFIG["imgsz"], SURVEIL_STAGE2_CONFIG["lr0"]))
