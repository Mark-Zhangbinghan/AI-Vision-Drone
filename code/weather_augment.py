"""
weather_augment.py - 在线天气增强（雨 / 雾 / 夜）
================================================

为监控视角模型补充恶劣天气鲁棒性。计划要求把雨/雾/夜作为**在线增强**接入训练，
覆盖所有源，而不只依赖 internet_v2 的离线 450 张。

实现方式：
  - 用 albumentations 定义三种变换：RandomRain / RandomFog / 亮度压暗(模拟夜)
  - 提供 WeatherBatchCallback，注册为 ultralytics 的 on_train_batch_start 回调
  - 每个 batch 以概率 weather_p 随机施加一种天气（在 batch 图像张量上原地变换）

注意：
  - 默认仅作用于训练阶段（on_train_batch_start 只在训练触发）
  - 变换在 numpy uint8 空间完成，再转回 ultralytics 的 CHW float 张量
  - 与 ultralytics 内置 mosaic/hsv 等增强叠加使用，互不影响
  - 若服务器未安装 albumentations，回调自动降级为 no-op（打印一次警告），不影响训练

依赖：
  albumentations>=1.3
"""

import random

import numpy as np
import torch

try:
    import albumentations as A
    _HAS_ALB = True
except Exception:  # pragma: no cover
    _HAS_ALB = False


# 在 numpy(uint8, HWC) 上施加的天气变换。p=1 表示每次被选中就必做。
def _build_transforms():
    """构建一个随机抽取的天气变换列表。"""
    transforms = []

    # 雨：斜线雨丝 + 轻微模糊
    transforms.append(
        A.Compose([
            A.RandomRain(
                slant_lower=-10, slant_upper=10,
                drop_length=15, drop_width=1, drop_color=(150, 150, 150),
                blur_value=3, brightness_coefficient=0.85,
                rain_type="drizzle", p=1.0,
            ),
        ])
    )

    # 雾：中等浓度雾，降低能见度
    transforms.append(
        A.Compose([
            A.RandomFog(
                fog_coef_lower=0.3, fog_coef_upper=0.6,
                alpha_coef=0.06, p=1.0,
            ),
        ])
    )

    # 夜：亮度压暗 + 轻微加噪（模拟低照度监控）
    transforms.append(
        A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=(-0.55, -0.35),
                contrast_limit=(-0.2, 0.0), p=1.0,
            ),
            A.OneOf([
                A.GaussNoise(stddev_range=(10.0, 30.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.4), p=1.0),
            ], p=0.5),
        ])
    )
    return transforms


_TRANSFORMS = _build_transforms() if _HAS_ALB else []


def _augment_image_np(img_np: np.ndarray) -> np.ndarray:
    """对单张 HWC uint8 图像施加一种随机天气。"""
    if not _TRANSFORMS:
        return img_np
    tf = random.choice(_TRANSFORMS)
    return tf(image=img_np)["image"]


class WeatherBatchCallback:
    """
    ultralytics on_train_batch_start 回调：对当前 batch 的图像施加天气增强。

    trainer.batch['img'] 形状 [B, 3, H, W]，值范围 0~1（未做均值方差标准化）。
    处理流程：CHW float -> HWC uint8 -> albumentations -> HWC uint8 -> CHW float。
    """

    def __init__(self, p: float = 0.3, enabled: bool = True):
        self.p = p
        self.enabled = enabled and _HAS_ALB
        self._warned = False
        if enabled and not _HAS_ALB:
            print("[WEATHER] 警告: 未安装 albumentations，在线天气增强已禁用(no-op)。"
                  " 安装: pip install albumentations")

    def __call__(self, trainer):
        if not self.enabled:
            return
        batch = getattr(trainer, "batch", None)
        if batch is None:
            return
        imgs = batch.get("img")
        if imgs is None or not isinstance(imgs, torch.Tensor):
            return

        # 只对当前 batch 的一部分施加（按概率 p）
        if random.random() > self.p:
            return

        try:
            # 转为 numpy (B,3,H,W) -> (B,H,W,3) uint8
            arr = imgs.detach().cpu().float().clamp_(0, 1).numpy()
            arr = (arr * 255.0).astype(np.uint8)
            arr = np.transpose(arr, (0, 2, 3, 1))  # B,H,W,3

            out = np.empty_like(arr)
            for i in range(arr.shape[0]):
                out[i] = _augment_image_np(arr[i])

            # 转回 CHW float 0~1，写回原 tensor（in-place）
            out = np.transpose(out, (0, 3, 1, 2)).astype(np.float32) / 255.0
            device = imgs.device
            imgs.copy_(torch.from_numpy(out).to(device))
        except Exception as e:  # pragma: no cover - 不阻断训练
            if not self._warned:
                print(f"[WEATHER] 回调异常已忽略: {e}")
                self._warned = True


def apply_weather_offline(image_dir: str, output_dir: str,
                          copies_per_image: int = 1, seed: int = 42):
    """
    离线批处理：对目录下图像生成雨/雾/夜变体（供补充数据集用，非训练必需）。
    """
    if not _HAS_ALB:
        raise RuntimeError("需要 albumentations: pip install albumentations")
    from pathlib import Path
    import shutil
    rng = random.Random(seed)
    src = Path(image_dir)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    for img_path in [p for p in src.iterdir() if p.suffix.lower() in exts]:
        img = np.array(__import__("cv2").imread(str(img_path)))
        if img is None:
            continue
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        for k in range(copies_per_image):
            aug = _augment_image_np(img)
            out_name = f"{img_path.stem}_weather{k}{img_path.suffix}"
            __import__("cv2").imwrite(str(dst / out_name), aug)
    print(f"[WEATHER] 离线增强完成: {image_dir} -> {output_dir}")
