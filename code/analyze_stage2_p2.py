"""T1 Part 2: Check internet labeling + stage2 quality."""
from pathlib import Path
from collections import Counter, defaultdict
import cv2, numpy as np, random

# ─── 1. Internet labels check ───
print("=" * 65)
print("  Internet Data Labeling Error Check")
print("=" * 65)
int_label_dir = Path(r"D:/AI_Drone_Vision/picture_process/internet/label")
int_classes_file = int_label_dir / "classes.txt"

if int_classes_file.exists():
    classes = int_classes_file.read_text(encoding="utf-8").strip().splitlines()
    print(f"  classes.txt: {len(classes)} classes")
    for i, name in enumerate(classes):
        has_data = False
        for tf in int_label_dir.glob("*.txt"):
            if tf.name == "classes.txt":
                continue
            for line in tf.read_text(encoding="utf-8").strip().splitlines():
                if not line.strip():
                    continue
                parts = line.strip().split()
                try:
                    cls = int(float(parts[0]))
                    if cls == i:
                        has_data = True
                        break
                except ValueError:
                    pass
            if has_data:
                break
        marker = " <<< HAS DATA" if has_data else ""
        print(f"    class {i}: {name}{marker}")

# ─── 2. Check unify remap config for internet data ───
print("\n  Unify remap config for internet source:")
unify_config = Path(r"D:/AI_Drone_Vision/picture_process/pipeline/unify_config.yaml")
if unify_config.exists():
    content = unify_config.read_text(encoding="utf-8")
    # Look for internet-related sections
    in_internet = False
    for line in content.splitlines():
        if "internet" in line.lower():
            in_internet = True
        elif in_internet and line.strip().startswith("-"):
            print(f"    {line.strip()}")
        elif in_internet and not line.strip().startswith(("-", " ", "\t")):
            in_internet = False

# ─── 3. Stage2 quality deep dive ───
print("\n" + "=" * 65)
print("  Stage2 Image Quality Analysis")
print("=" * 65)
s2_base = Path(r"D:/AI_Drone_Vision/picture_process/stage2_cls_dataset")
random.seed(42)

for cls_name in ["drowning", "swimming"]:
    all_imgs = list(s2_base.glob(f"train/{cls_name}/*.jpg")) + list(s2_base.glob(f"val/{cls_name}/*.jpg"))
    print(f"\n  {cls_name}: {len(all_imgs)} total")

    # Sample 200 images for quality check
    sample = random.sample(all_imgs, min(200, len(all_imgs)))
    
    laplacian_scores = []
    sizes = []
    for f in sample:
        try:
            img = cv2.imread(str(f))
            if img is None:
                continue
            sizes.append(img.shape[:2])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F).var()
            laplacian_scores.append(lap)
        except:
            pass

    laplacian_scores = np.array(laplacian_scores)
    sizes = np.array(sizes)

    print(f"    Resolution: min={sizes[:,0].min():.0f}x{sizes[:,1].min():.0f}, "
          f"mean={sizes[:,0].mean():.0f}x{sizes[:,1].mean():.0f}, "
          f"max={sizes[:,0].max():.0f}x{sizes[:,1].max():.0f}")
    
    hist_bins = [0, 50, 100, 200, 500, 1000, 99999]
    hist_labels = ["0-50", "50-100", "100-200", "200-500", "500-1K", "1K+"]
    print(f"    Laplacian variance distribution:")
    total_valid = len(laplacian_scores)
    for i in range(len(hist_bins)-1):
        count = int(((laplacian_scores >= hist_bins[i]) & (laplacian_scores < hist_bins[i+1])).sum())
        bar = "█" * (count * 50 // total_valid) if total_valid > 0 else ""
        print(f"      {hist_labels[i]:>8}: {count:>4} ({count/total_valid*100:5.1f}%) {bar}")
    
    poor = int((laplacian_scores < 100).sum())
    print(f"    BLURRY (<100): {poor}/{total_valid} ({poor/total_valid*100:.1f}%)")

# ─── 4. Summary for T1 ───
print("\n" + "=" * 65)
print("  T1 SUMMARY")
print("=" * 65)
print("""
  Key Findings:
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. Stage2 dataset: 100% arc_ prefix (archive/Kaggle)        │
  │    - NO inet_ or new_ or DJI_ data in Stage2                │
  │    - All ground-view perspective, no drone-view data         │
  │                                                             │
  │ 2. Class ratio (unified labels bucket):                     │
  │    drowning:swimming = 15,571:12,915 ≈ 1.21:1               │
  │                                                             │
  │ 3. inet_ data has drowning=2,340 + swimming=564             │
  │    Could add swimming crops from inet_ to Stage2             │
  │                                                             │
  │ 4. new_ data has swimming=3,346 but NO drowning             │
  │    Large potential source for swimming data                  │
  │                                                             │
  │ 5. DJI_ data has NO drowning/swimming labels                │
  │    Cannot use for Stage2 without manual labeling             │
  └─────────────────────────────────────────────────────────────┘
""")
