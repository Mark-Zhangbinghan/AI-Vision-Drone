"""T1: Analyze stage2_cls_dataset source composition and quality."""
import os, sys
from pathlib import Path
from collections import defaultdict, Counter
import json
import random

# ─── 1. Analyze unified_dataset label source composition ───
base = Path(r"D:/AI_Drone_Vision/picture_process/unified_dataset/labels")
splits = ["train", "val", "test"]

results = defaultdict(lambda: defaultdict(int))

for split in splits:
    label_dir = base / split
    if not label_dir.exists():
        continue
    for tf in label_dir.glob("*.txt"):
        prefix = tf.stem.split("_")[0]
        lines = tf.read_text().strip().splitlines()
        for line in lines:
            if not line.strip():
                continue
            cls = int(float(line.strip().split()[0]))
            results[prefix][cls] += 1

print("=" * 65)
print("  Unified Dataset — Class Distribution by Source")
print("=" * 65)
class_names = {
    0: "person", 1: "boat", 2: "surfboard", 3: "wood",
    4: "life_buoy", 5: "drowning", 6: "background", 7: "swimming"
}

for src in ["arc", "DJI", "new", "inet"]:
    if src in results:
        data = results[src]
        total = sum(data.values())
        print(f"\n  [{src}] total instances: {total}")
        for cls_id in sorted(data.keys()):
            name = class_names.get(cls_id, f"class_{cls_id}")
            count = data[cls_id]
            pct = count / total * 100
            bar = "█" * int(pct / 2)
            print(f"    cls{cls_id} {name:<15} {count:>7} ({pct:5.1f}%) {bar}")

print("\n" + "-" * 65)
print("  Drowning(5) + Swimming(7) Summary")
print("-" * 65)
print(f"  {'Source':<12} {'drowning':>10} {'swimming':>10} {'ratio(d/s)':>10}")
total_d = sum(results[s].get(5, 0) for s in results)
total_s = sum(results[s].get(7, 0) for s in results)
for src in ["arc", "DJI", "new", "inet"]:
    d = results[src].get(5, 0)
    s = results[src].get(7, 0)
    if d > 0 or s > 0:
        print(f"  {src:<12} {d:>10} {s:>10} {d/s:>10.4f}")
print(f"  {'TOTAL':<12} {total_d:>10} {total_s:>10} {total_d/total_s:>10.4f}")

# ─── 2. Check if internet data has background mapped to drowning ───
print("\n" + "=" * 65)
print("  Internet Data Labeling Error Check")
print("=" * 65)
# Original internet labels have class 15=drowning, 16=background, 17=swimming
# After unify remapping, they become 5, 6, 7 in unified dataset
# We need to check the ORIGINAL labels
int_label_dir = Path(r"D:/AI_Drone_Vision/picture_process/internet/label")
if int_label_dir.exists():
    orig_cls = defaultdict(int)
    for tf in int_label_dir.glob("*.txt"):
        for line in tf.read_text().strip().splitlines():
            if not line.strip():
                continue
            cls = int(float(line.strip().split()[0]))
            orig_cls[cls] += 1
    print("  Original internet labels (class IDs):")
    for cls_id in sorted(orig_cls.keys()):
        print(f"    class {cls_id}: {orig_cls[cls_id]} instances")
    
    # Check: class 16 (originally background) → what did it become?
    # Read unify config to check mapping
    print("\n  Checking unify_config remapping...")

# ─── 3. Check stage2 crops for any non-arc_ files ───
print("\n" + "=" * 65)
print("  Stage2 Dataset — Source Prefix Check")
print("=" * 65)
s2_base = Path(r"D:/AI_Drone_Vision/picture_process/stage2_cls_dataset")
for split_name in ["train", "val"]:
    for cls_name in ["drowning", "swimming"]:
        s2_dir = s2_base / split_name / cls_name
        if not s2_dir.exists():
            continue
        # Sample prefixes
        prefixes = Counter()
        for f in s2_dir.iterdir():
            pre = f.stem.split("_")[0]
            prefixes[pre] += 1
        unique_prefs = len(prefixes)
        total = sum(prefixes.values())
        print(f"\n  {split_name}/{cls_name}: {total} files, {unique_prefs} unique prefixes")
        for pre, count in prefixes.most_common(5):
            print(f"    {pre}: {count} ({count/total*100:.1f}%)")
        if len(prefixes) > 5:
            print(f"    ... and {len(prefixes)-5} more prefixes")

# ─── 4. Image quality check (sample) ───
print("\n" + "=" * 65)
print("  Stage2 Image Quality (Sample)")
print("=" * 65)
import cv2
import numpy as np

def sample_quality(base_dir, class_name, n=100):
    """Sample images and compute quality metrics."""
    img_dir = base_dir / class_name
    files = list(img_dir.glob("*.jpg"))
    import random
    random.seed(42)
    sample = random.sample(files, min(n, len(files)))
    
    sizes = []
    blurs = []
    for f in sample:
        try:
            img = cv2.imread(str(f))
            if img is None:
                continue
            sizes.append(img.shape[:2])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blurs.append(laplacian_var)
        except:
            pass
    
    sizes = np.array(sizes)
    blurs = np.array(blurs)
    return {
        "count": len(sizes),
        "size_mean": sizes.mean(axis=0) if len(sizes) > 0 else None,
        "size_std": sizes.std(axis=0) if len(sizes) > 0 else None,
        "blur_mean": blurs.mean() if len(blurs) > 0 else None,
        "blur_std": blurs.std() if len(blurs) > 0 else None,
        "blur_min": blurs.min() if len(blurs) > 0 else None,
        "blur_poor": int((blurs < 100).sum()) if len(blurs) > 0 else 0,  # Laplacian var < 100 = blurry
    }

for cls_name in ["drowning", "swimming"]:
    for split_name in ["train", "val"]:
        s2_dir = s2_base / split_name
        q = sample_quality(s2_dir, cls_name, n=100)
        if q["size_mean"] is not None:
            print(f"  {split_name}/{cls_name}:")
            print(f"    Resolution: {q['size_mean'][1]:.0f}x{q['size_mean'][0]:.0f} (all 256x256 expected)")
            print(f"    Blur (Laplacian var): mean={q['blur_mean']:.0f}, poor={q['blur_poor']}/{q['count']}")

print("\n" + "=" * 65)
print("  T1 Analysis Complete")
print("=" * 65)
