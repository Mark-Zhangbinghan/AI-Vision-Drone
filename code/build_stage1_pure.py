"""
T2: Build stage1_pure_dataset — single-class person_in_water detection dataset.
80/10/10 train/val/test split with 46% drone-view + 10% background images.

Sources:
  - DJI person:  4,500 (drone-view, person→class0)
  - DJI bg:      1,240 (drone-view, empty labels)
  - arc_:        4,568 (ground-view, drowning+swimming→class0)
  - new_:          876 (iPhone, swimming→class0)
  - inet_:       1,220 (web, drowning+swimming→class0)
  ─────────────────────
  Total:        12,404
"""
import json, os, random, shutil, sys, cv2
from pathlib import Path
from collections import defaultdict, Counter
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "picture_process" / "pipeline"))
from common import (
    load_coco, build_image_lookup, coco_bbox_to_yolo,
    read_image_from_zip, close_zip_reader,
    CATEGORY_MAP, IMAGES_ZIP_PATH, ANNOTATIONS_PATH,
)

SEED = 42
random.seed(SEED)

# ─── Paths ───
ROOT = Path(__file__).resolve().parent.parent  # D:/AI_Drone_Vision
OUTPUT = ROOT / "picture_process" / "stage1_pure_dataset"
UNIFIED_LABELS = ROOT / "picture_process" / "unified_dataset" / "labels"
UNIFIED_IMAGES = ROOT / "picture_process" / "unified_dataset" / "images"

# ─── Target counts ───
DJI_PERSON_TOTAL = 4500
DJI_BG_TOTAL = 1240
ARC_TOTAL = 4568
NEW_TOTAL = 876
INET_TOTAL = 1220

def split_counts(total):
    """80/10/10 split"""
    t = int(total * 0.8)
    v = int(total * 0.1)
    ts = total - t - v
    return t, v, ts


# ================================================================
#  T2-A & T2-B: DJI data from images.zip
# ================================================================
print("=" * 60)
print("  T2-A/B: Sampling DJI images from COCO annotations")
print("=" * 60)

images, annotations, categories = load_coco()
img_lookup = build_image_lookup(images)

# Group images by has_person
person_img_ids = set()
img_id_to_person_anns = defaultdict(list)
for ann in annotations:
    if ann["category_id"] == 1:  # person
        person_img_ids.add(ann["image_id"])
        img_id_to_person_anns[ann["image_id"]].append(ann)

no_person_ids = [img["id"] for img in images if img["id"] not in person_img_ids]
person_ids_pool = list(person_img_ids)

print(f"  Person images: {len(person_ids_pool)}")
print(f"  No-person images: {len(no_person_ids)}")

# Sample
sampled_person_ids = set(random.sample(person_ids_pool, DJI_PERSON_TOTAL))
sampled_bg_ids = set(random.sample(no_person_ids, DJI_BG_TOTAL))

print(f"  Sampled DJI person: {len(sampled_person_ids)}")
print(f"  Sampled DJI bg:     {len(sampled_bg_ids)}")

# Create output directories
for split_ in ["train", "val", "test"]:
    (OUTPUT / "images" / split_).mkdir(parents=True, exist_ok=True)
    (OUTPUT / "labels" / split_).mkdir(parents=True, exist_ok=True)

# Split DJI person IDs
dji_p_ids = list(sampled_person_ids)
random.shuffle(dji_p_ids)
dji_p_train_n, dji_p_val_n, dji_p_test_n = split_counts(DJI_PERSON_TOTAL)
dji_p_splits = {
    "train": set(dji_p_ids[:dji_p_train_n]),
    "val":   set(dji_p_ids[dji_p_train_n:dji_p_train_n+dji_p_val_n]),
    "test":  set(dji_p_ids[dji_p_train_n+dji_p_val_n:]),
}

# Split DJI bg IDs
dji_bg_ids = list(sampled_bg_ids)
random.shuffle(dji_bg_ids)
dji_bg_train_n, dji_bg_val_n, dji_bg_test_n = split_counts(DJI_BG_TOTAL)
dji_bg_splits = {
    "train": set(dji_bg_ids[:dji_bg_train_n]),
    "val":   set(dji_bg_ids[dji_bg_train_n:dji_bg_train_n+dji_bg_val_n]),
    "test":  set(dji_bg_ids[dji_bg_train_n+dji_bg_val_n:]),
}

# ─── Extract DJI person images + labels ───
print("\n  Extracting DJI person images + converting labels...")
stats = defaultdict(lambda: defaultdict(int))
person_annotations_count = 0

for split_ in ["train", "val", "test"]:
    ids = list(dji_p_splits[split_])
    done = 0
    skipped = 0
    failed = 0
    for i, img_id in enumerate(ids):
        img_info = img_lookup[img_id]
        fname = img_info["file_name"]
        base = Path(fname).stem
        
        out_img = OUTPUT / "images" / split_ / f"{base}.png"
        out_label = OUTPUT / "labels" / split_ / f"{base}.txt"
        
        # Resumability: skip if both image and label already exist
        if out_img.exists() and out_label.exists():
            skipped += 1
            stats["DJI_person"][split_] += 1
            continue
        
        img = read_image_from_zip(fname)
        if img is None:
            failed += 1
            continue
        
        out_img.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_img), img)
        
        yolo_lines = []
        for ann in img_id_to_person_anns.get(img_id, []):
            cx, cy, nw, nh = coco_bbox_to_yolo(
                ann["bbox"], img_info["width"], img_info["height"]
            )
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            person_annotations_count += 1
        
        out_label.write_text("\n".join(yolo_lines))
        done += 1
        stats["DJI_person"][split_] += 1
        
        if (i + 1) % 500 == 0:
            print(f"    {split_}: {i+1}/{len(ids)} (done={done}, skip={skipped}, fail={failed})")

    print(f"    {split_}: {len(ids)} total, {done} new, {skipped} skipped, {failed} failed")

close_zip_reader()
print(f"    DJI_person total: {sum(stats['DJI_person'].values())} images, {person_annotations_count} annotations")

# ─── Extract DJI background images + empty labels ───
print("\n  Extracting DJI background images...")
for split_ in ["train", "val", "test"]:
    ids = list(dji_bg_splits[split_])
    done = 0
    skipped = 0
    failed = 0
    for i, img_id in enumerate(ids):
        img_info = img_lookup[img_id]
        fname = img_info["file_name"]
        base = Path(fname).stem
        
        out_img = OUTPUT / "images" / split_ / f"{base}.png"
        out_label = OUTPUT / "labels" / split_ / f"{base}.txt"
        
        if out_img.exists() and out_label.exists():
            skipped += 1
            stats["DJI_bg"][split_] += 1
            continue
        
        img = read_image_from_zip(fname)
        if img is None:
            failed += 1
            continue
        
        out_img.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_img), img)
        out_label.write_text("")
        done += 1
        stats["DJI_bg"][split_] += 1
        
        if (i + 1) % 300 == 0:
            print(f"    {split_}: {i+1}/{len(ids)} (done={done}, skip={skipped}, fail={failed})")

    print(f"    {split_}: {len(ids)} total, {done} new, {skipped} skipped, {failed} failed")

close_zip_reader()
print(f"    DJI_bg total: {sum(stats['DJI_bg'].values())} images")

# ================================================================
#  T2-C & T2-D: Ground-view data from unified_dataset
# ================================================================
print("\n" + "=" * 60)
print("  T2-C/D: Ground-view data from unified_dataset")
print("=" * 60)

# Collect arc_ files (from all splits in unified)
arc_labels = list(UNIFIED_LABELS.glob("train/arc_*.txt")) + \
             list(UNIFIED_LABELS.glob("val/arc_*.txt")) + \
             list(UNIFIED_LABELS.glob("test/arc_*.txt"))
print(f"  arc_ files available: {len(arc_labels)}")

new_labels = list(UNIFIED_LABELS.glob("train/new_*.txt")) + \
             list(UNIFIED_LABELS.glob("val/new_*.txt"))
print(f"  new_ files available: {len(new_labels)}")

inet_labels = list(UNIFIED_LABELS.glob("train/inet_*.txt")) + \
              list(UNIFIED_LABELS.glob("val/inet_*.txt"))
print(f"  inet_ files available: {len(inet_labels)}")

# Sample arc_
random.shuffle(arc_labels)
arc_sampled = arc_labels[:ARC_TOTAL]
# Split
arc_train_n, arc_val_n, arc_test_n = split_counts(ARC_TOTAL)
arc_by_split = {
    "train": arc_sampled[:arc_train_n],
    "val":   arc_sampled[arc_train_n:arc_train_n+arc_val_n],
    "test":  arc_sampled[arc_train_n+arc_val_n:],
}

# Split new_ (all of them)
random.shuffle(new_labels)
new_sampled = new_labels[:NEW_TOTAL]  # should be exact
new_train_n, new_val_n, new_test_n = split_counts(NEW_TOTAL)
new_by_split = {
    "train": new_sampled[:new_train_n],
    "val":   new_sampled[new_train_n:new_train_n+new_val_n],
    "test":  new_sampled[new_train_n+new_val_n:],
}

# Split inet_ (all of them)
random.shuffle(inet_labels)
inet_sampled = inet_labels[:INET_TOTAL]
inet_train_n, inet_val_n, inet_test_n = split_counts(INET_TOTAL)
inet_by_split = {
    "train": inet_sampled[:inet_train_n],
    "val":   inet_sampled[inet_train_n:inet_train_n+inet_val_n],
    "test":  inet_sampled[inet_train_n+inet_val_n:],
}


def copy_ground_data(label_list, split_, prefix):
    """Copy images + remapped labels from unified_dataset. Resumable."""
    count = 0
    annotations_count = 0
    skipped = 0
    for label_path in label_list:
        base = label_path.stem
        
        out_label = OUTPUT / "labels" / split_ / f"{base}.txt"
        
        # Find corresponding image
        img_path = None
        for ext in [".jpg", ".jpeg", ".PNG", ".png"]:
            for s in ["train", "val", "test"]:
                p = UNIFIED_IMAGES / s / f"{base}{ext}"
                if p.exists():
                    img_path = p
                    break
            if img_path:
                break
        
        if img_path is None:
            continue
        
        out_ext = img_path.suffix
        out_img = OUTPUT / "images" / split_ / f"{base}{out_ext}"
        
        # Resumability: skip if both exist
        if out_img.exists() and out_label.exists():
            skipped += 1
            count += 1
            continue
        
        # Clean up partial files from previous crashed run
        try:
            if out_img.exists():
                out_img.unlink()
            if out_label.exists():
                out_label.unlink()
        except PermissionError:
            skipped += 1
            continue
        
        try:
            shutil.copy2(str(img_path), str(out_img))
        except (PermissionError, OSError):
            skipped += 1
            continue
        
        # Remap: drowning(5)→0, swimming(7)→0, discard others
        yolo_lines = []
        for line in label_path.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            parts = line.strip().split()
            cls = int(float(parts[0]))
            if cls in (5, 7):
                yolo_lines.append("0 " + " ".join(parts[1:]))
                annotations_count += 1
        
        out_label.write_text("\n".join(yolo_lines))
        count += 1
    
    print(f"    {split_}: {count} images, {skipped} skipped")
    return count, annotations_count


print("\n  Copying arc_ data...")
for split_ in ["train", "val", "test"]:
    n, a = copy_ground_data(arc_by_split[split_], split_, "arc")
    stats["arc"][split_] = n
    print(f"    {split_}: {n} images, {a} annotations")

print("\n  Copying new_ data...")
for split_ in ["train", "val", "test"]:
    n, a = copy_ground_data(new_by_split[split_], split_, "new")
    stats["new"][split_] = n
    print(f"    {split_}: {n} images, {a} annotations")

print("\n  Copying inet_ data...")
for split_ in ["train", "val", "test"]:
    n, a = copy_ground_data(inet_by_split[split_], split_, "inet")
    stats["inet"][split_] = n
    print(f"    {split_}: {n} images, {a} annotations")


# ================================================================
#  T2-E: Generate data.yaml + statistics.json + verify
# ================================================================
print("\n" + "=" * 60)
print("  T2-E: Generating configs + verifying")
print("=" * 60)

# data.yaml
data_yaml = {
    "path": ".",
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": 1,
    "names": ["person_in_water"],
}
import yaml
yaml_path = OUTPUT / "data.yaml"
with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
print(f"  Written: {yaml_path}")

# Count everything
summary = {}
total = 0
for split_ in ["train", "val", "test"]:
    img_count = len(list((OUTPUT / "images" / split_).glob("*")))
    lbl_count = len(list((OUTPUT / "labels" / split_).glob("*.txt")))
    summary[split_] = {"images": img_count, "labels": lbl_count}
    total += img_count

# Per-source breakdown
for src in ["DJI_person", "DJI_bg", "arc", "new", "inet"]:
    src_total = sum(stats[src].values())
    splits_detail = {s: stats[src][s] for s in ["train", "val", "test"]}
    src_pct = src_total / total * 100 if total > 0 else 0
    print(f"  {src:<15}: {src_total:>5} ({src_pct:5.1f}%) {splits_detail}")

print(f"  {'TOTAL':<15}: {total:>5}")

# Count empty labels (background images)
empty_count = 0
for split_ in ["train", "val", "test"]:
    for lbl in (OUTPUT / "labels" / split_).glob("*.txt"):
        content = lbl.read_text(encoding="utf-8").strip()
        if not content:
            empty_count += 1

# Count all person_in_water annotations
piw_count = 0
for split_ in ["train", "val", "test"]:
    for lbl in (OUTPUT / "labels" / split_).glob("*.txt"):
        for line in lbl.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                piw_count += 1

print(f"\n  person_in_water annotations: {piw_count}")
print(f"  empty (background) images:   {empty_count}")
print(f"  bg ratio:                    {empty_count/total*100:.1f}%")

# statistics.json
statistics = {
    "dataset": "stage1_pure_dataset",
    "nc": 1,
    "class_names": ["person_in_water"],
    "total_images": total,
    "total_annotations": piw_count,
    "background_images": empty_count,
    "background_ratio": round(empty_count/total*100, 2),
    "split_ratio": "80/10/10",
    "sources": {
        "DJI_person": {"total": sum(stats["DJI_person"].values()), "splits": stats["DJI_person"]},
        "DJI_bg":     {"total": sum(stats["DJI_bg"].values()),     "splits": stats["DJI_bg"]},
        "arc":        {"total": sum(stats["arc"].values()),        "splits": stats["arc"]},
        "new":        {"total": sum(stats["new"].values()),        "splits": stats["new"]},
        "inet":       {"total": sum(stats["inet"].values()),       "splits": stats["inet"]},
    },
    "per_split": summary,
}
stats_path = OUTPUT / "statistics.json"
with open(stats_path, "w", encoding="utf-8") as f:
    json.dump(statistics, f, indent=2, ensure_ascii=False)
print(f"  Written: {stats_path}")

print("\n" + "=" * 60)
print("  T2 DONE")
print("=" * 60)
print(f"  Output: {OUTPUT}")
print(f"  {total} images → {summary['train']['images']} train / {summary['val']['images']} val / {summary['test']['images']} test")
