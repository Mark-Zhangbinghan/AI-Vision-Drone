"""
eval_stage1.py - Stage 1 粗粒度检测器评估脚本
=============================================
在 test split 上评估 Stage 1 模型:
  - mAP50, mAP50-95
  - 各类别 Precision, Recall, F1
  - 混淆矩阵
  - PR 曲线

Usage:
    python eval_stage1.py                          # 默认评估 (best.pt on test)
    python eval_stage1.py --split val              # 在 val split 上评估
    python eval_stage1.py --weights last.pt        # 评估 last.pt
    python eval_stage1.py --data path/to/data.yaml # 指定数据集配置
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Default paths
DEFAULT_WEIGHTS = str(PROJECT_ROOT / "runs" / "stage1_v1" / "best.pt")
DEFAULT_DATA = str(PROJECT_ROOT.parent / "picture_process" / "stage1_dataset" / "data_local.yaml")

STAGE1_CLASS_NAMES = ["person", "person_in_water", "boat", "floating_object", "life_buoy"]


def main():
    parser = argparse.ArgumentParser(description="Stage 1 检测器评估")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS,
                        help="模型权重路径 (best.pt 或 last.pt)")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA,
                        help="数据集 data.yaml 路径 (本地评估用 data_local.yaml)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"],
                        help="评估数据集 split (val 或 test)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="检测置信度阈值")
    parser.add_argument("--iou", type=float, default=0.6,
                        help="NMS IoU 阈值")
    parser.add_argument("--device", type=str, default="0",
                        help="评估设备 (0 / cpu)")
    parser.add_argument("--batch", type=int, default=16,
                        help="评估 batch size")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="结果保存目录 (默认: runs/stage1_eval)")
    args = parser.parse_args()

    # Validate paths
    weights_path = Path(args.weights)
    data_path = Path(args.data)

    if not weights_path.exists():
        print(f"[ERROR] 权重文件不存在: {weights_path}")
        print(f"  请确认 best.pt 位置，或用 --weights 指定正确路径")
        sys.exit(1)

    if not data_path.exists():
        print(f"[ERROR] 数据集配置不存在: {data_path}")
        print(f"  请确认 data.yaml/data_local.yaml 位置，或用 --data 指定正确路径")
        sys.exit(1)

    # Print evaluation config
    print("=" * 60)
    print("  Stage 1 检测器评估")
    print("=" * 60)
    print(f"  权重:   {weights_path}")
    print(f"  数据:   {data_path}")
    print(f"  Split:  {args.split}")
    print(f"  Conf:   {args.conf}")
    print(f"  IoU:    {args.iou}")
    print(f"  Device: {args.device}")
    print(f"  类别:   {STAGE1_CLASS_NAMES}")
    print("=" * 60)

    # Load model and run validation
    from ultralytics import YOLO

    model = YOLO(str(weights_path))

    save_dir = args.save_dir or str(PROJECT_ROOT / "runs" / "stage1_eval")

    results = model.val(
        data=str(data_path),
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        batch=args.batch,
        plots=True,
        save_dir=save_dir,
        verbose=True,
    )

    # Print detailed results
    print("\n" + "=" * 60)
    print("  评估结果")
    print("=" * 60)

    # Overall metrics
    if hasattr(results, 'box'):
        box_results = results.box
        print(f"\n  [总体指标]")
        print(f"    mAP50:      {box_results.map50:.4f}")
        print(f"    mAP50-95:   {box_results.map:.4f}")
        print(f"    Precision:  {box_results.mp:.4f}")
        print(f"    Recall:     {box_results.mr:.4f}")

        # Per-class metrics
        print(f"\n  [各类别指标]")
        print(f"    {'类别':<20s} {'P':>8s} {'R':>8s} {'F1':>8s} {'AP50':>8s} {'AP50-95':>8s}")
        print(f"    {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        for i, cls_name in enumerate(STAGE1_CLASS_NAMES):
            p = box_results.p[i] if i < len(box_results.p) else 0
            r = box_results.r[i] if i < len(box_results.r) else 0
            ap50 = box_results.ap50[i] if i < len(box_results.ap50) else 0
            ap = box_results.ap[i] if i < len(box_results.ap) else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            print(f"    {cls_name:<20s} {p:>8.4f} {r:>8.4f} {f1:>8.4f} {ap50:>8.4f} {ap:>8.4f}")

    # Save results to JSON
    eval_results = {
        "weights": str(weights_path),
        "data": str(data_path),
        "split": args.split,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "class_names": STAGE1_CLASS_NAMES,
    }

    if hasattr(results, 'box'):
        box_results = results.box
        eval_results["overall"] = {
            "mAP50": float(box_results.map50),
            "mAP50-95": float(box_results.map),
            "precision": float(box_results.mp),
            "recall": float(box_results.mr),
        }
        per_class = {}
        for i, cls_name in enumerate(STAGE1_CLASS_NAMES):
            p = float(box_results.p[i]) if i < len(box_results.p) else 0
            r = float(box_results.r[i]) if i < len(box_results.r) else 0
            ap50 = float(box_results.ap50[i]) if i < len(box_results.ap50) else 0
            ap = float(box_results.ap[i]) if i < len(box_results.ap) else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            per_class[cls_name] = {
                "precision": p, "recall": r, "f1": f1,
                "AP50": ap50, "AP50-95": ap,
            }
        eval_results["per_class"] = per_class

    results_json_path = Path(save_dir) / "eval_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    print(f"\n  结果保存到: {results_json_path}")
    print(f"  图表保存到: {save_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
