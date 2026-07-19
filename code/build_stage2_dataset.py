"""
build_stage2_dataset.py - 构建 Stage 2 细粒度分类数据集
=========================================================
两阶段架构的第二阶段数据集构建脚本

逻辑:
  1. 遍历原始 unified_dataset 的图片和标签
  2. 对每个标注中包含 drowning(5) 或 swimming(7) 的 bbox
  3. 从原图裁剪该 bbox 区域 (padding 20%, resize 256x256)
  4. 按 class_id 分为 drowning / swimming 两个分类目录
  5. 处理重叠: 同一 bbox 区域有 drowning+swimming → 取 drowning (安全优先)
  6. 同时处理 background(6) 类 (映射为 person, 不进入 Stage 2)

输出目录结构:
  stage2_cls_dataset/
    train/
      drowning/
        img_001_crop_0.jpg
      swimming/
        img_002_crop_0.jpg
    val/
      drowning/
      swimming/

Usage:
    python build_stage2_dataset.py
    python build_stage2_dataset.py --padding 0.3
    python build_stage2_dataset.py --resize 128
"""

import sys
import argparse
import random
import shutil
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT.parent / "picture_process" / "unified_dataset"
OUTPUT_ROOT = PROJECT_ROOT.parent / "picture_process" / "stage2_cls_dataset"

# 原始数据集中的 class_id → Stage 2 分类
# 只有 drowning(5) 和 swimming(7) 进入 Stage 2
STAGE2_CLASS_MAP = {
    5: "drowning",
    7: "swimming",
}


def parse_args():
    parser = argparse.ArgumentParser(description="构建 Stage 2 分类数据集")
    parser.add_argument("--padding", type=float, default=0.2,
                        help="裁剪 bbox 的 padding 比例 (0.2 = 20%%)")
    parser.add_argument("--resize", type=int, default=256,
                        help="裁剪后 resize 尺寸")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (用于 train/val 拆分)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="验证集比例")
    return parser.parse_args()


def crop_bbox(image, cx, cy, w, h, padding=0.2):
    """
    从原始图片中裁剪 YOLO bbox 区域 (cx, cy, w, h 为归一化坐标)
    padding: 在 bbox 四周额外扩展的比例

    Returns:
        crop: 裁剪后的图片区域 (numpy array)
    """
    img_h, img_w = image.shape[:2]

    # 归一化坐标 → 绝对像素坐标
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)

    # 添加 padding
    pad_w = int((x2 - x1) * padding)
    pad_h = int((y2 - y1) * padding)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(img_w, x2 + pad_w)
    y2 = min(img_h, y2 + pad_h)

    # 裁剪
    crop = image[y1:y2, x1:x2]

    # 如果裁剪区域太小 (目标太小), 至少保证 32x32
    if crop.shape[0] < 32 or crop.shape[1] < 32:
        # 扩大裁剪区域到最小尺寸
        cx_abs = int(cx * img_w)
        cy_abs = int(cy * img_h)
        x1 = max(0, cx_abs - 64)
        y1 = max(0, cy_abs - 64)
        x2 = min(img_w, cx_abs + 64)
        y2 = min(img_h, cy_abs + 64)
        crop = image[y1:y2, x1:x2]

    return crop


def read_label_file(label_path):
    """读取 YOLO 格式标签文件, 返回 [(class_id, cx, cy, w, h), ...]"""
    annotations = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            annotations.append((cls_id, cx, cy, w, h))
    return annotations


def build_dataset(padding=0.2, resize=256, seed=42, val_ratio=0.15):
    """
    构建 Stage 2 分类数据集
    """
    random.seed(seed)

    # 收集所有裁剪样本
    samples = defaultdict(list)  # class_name → [crop_img, ...]

    # 统计
    total_crops = 0
    skipped = 0
    multi_label_count = 0

    # 遍历 train 和 val 目录
    for split in ["train", "val"]:
        img_dir = DATASET_ROOT / "images" / split
        lbl_dir = DATASET_ROOT / "labels" / split

        if not img_dir.exists() or not lbl_dir.exists():
            print(f"[WARN] 目录不存在: {split}")
            continue

        img_files = sorted(img_dir.glob("*"))
        print(f"[INFO] {split}: {len(img_files)} 张图片")

        for img_path in tqdm(img_files, desc=f"裁剪 {split}", unit="img"):
            stem = img_path.stem
            lbl_path = lbl_dir / f"{stem}.txt"

            if not lbl_path.exists():
                continue

            # 读取图片
            image = cv2.imread(str(img_path))
            if image is None:
                skipped += 1
                continue

            # 读取标注
            annotations = read_label_file(lbl_path)

            # 对每个 drowning/swimming bbox 裁剪
            for cls_id, cx, cy, w, h in annotations:
                if cls_id not in STAGE2_CLASS_MAP:
                    continue  # 只处理 drowning/swimming

                class_name = STAGE2_CLASS_MAP[cls_id]

                # 裁剪 bbox
                crop = crop_bbox(image, cx, cy, w, h, padding)

                # Resize
                if resize > 0:
                    crop = cv2.resize(crop, (resize, resize),
                                     interpolation=cv2.INTER_LINEAR)

                # 保存到临时列表 (稍后写入目录)
                # 提取来源前缀 (arc_N) 用于按源拆分 train/val, 防止数据泄露
                src_prefix = stem.split("_", 2)[:2]  # e.g. ['arc', '10'] → 'arc_10'
                src_key = "_".join(src_prefix) if len(src_prefix) >= 2 else stem
                samples[class_name].append((crop, stem, split, src_key))

                total_crops += 1

    # 处理多标签图片统计
    print(f"\n[统计] 裁剪总数: {total_crops}")
    print(f"[统计] 跳过: {skipped}")
    for cls_name, items in samples.items():
        print(f"  {cls_name}: {len(items)} 个裁剪样本")

    # 创建输出目录并写入
    for sub in ["train", "val"]:
        for cls_name in STAGE2_CLASS_MAP.values():
            (OUTPUT_ROOT / sub / cls_name).mkdir(parents=True, exist_ok=True)

    # 拆分 train/val — 按来源 (arc_N) 分组, 防止数据泄露
    written = 0
    leak_check = {}
    for cls_name, items in samples.items():
        # 按来源前缀分组
        src_groups = defaultdict(list)
        for item in items:
            src_key = item[3]  # src_key extracted during collection
            src_groups[src_key].append(item)

        # 打乱来源顺序
        src_keys = list(src_groups.keys())
        random.shuffle(src_keys)

        n_val_src = max(1, int(len(src_keys) * val_ratio))
        train_srcs = set(src_keys[:-n_val_src]) if n_val_src > 0 else set(src_keys)
        val_srcs = set(src_keys[-n_val_src:]) if n_val_src > 0 else set()

        train_count = sum(len(src_groups[k]) for k in train_srcs)
        val_count = sum(len(src_groups[k]) for k in val_srcs)
        print(f"  {cls_name}: {len(src_keys)} sources → train={train_count} crops ({len(train_srcs)} srcs), "
              f"val={val_count} crops ({len(val_srcs)} srcs)")

        # 写入
        for split_name, src_set in [("train", train_srcs), ("val", val_srcs)]:
            for src_key in src_set:
                for crop_idx, (crop, stem, _, _) in enumerate(src_groups[src_key]):
                    filename = f"{stem}_crop_{crop_idx}.jpg"
                    out_path = OUTPUT_ROOT / split_name / cls_name / filename
                    cv2.imwrite(str(out_path), crop,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    written += 1

        # 验证无泄露: train 和 val 的 src 不应有交集
        overlap = train_srcs & val_srcs
        if overlap:
            print(f"[ERROR] Leak detected! {len(overlap)} sources in both train/val: {list(overlap)[:5]}")

    print(f"\n[完成] Stage 2 分类数据集已构建")
    print(f"  输出: {OUTPUT_ROOT}")
    print(f"  写入: {written} 个裁剪样本")
    print(f"  drowning: {len(samples.get('drowning', []))} 总裁剪")
    print(f"  swimming: {len(samples.get('swimming', []))} 总裁剪")

    # 保存统计
    stats = {
        "total_crops": total_crops,
        "skipped": skipped,
        "class_counts": {cls: len(items) for cls, items in samples.items()},
        "train_ratio": 1 - val_ratio,
        "val_ratio": val_ratio,
        "padding": padding,
        "resize": resize,
    }
    import json
    stats_path = OUTPUT_ROOT / "statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  统计: {stats_path}")


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        padding=args.padding,
        resize=args.resize,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )
