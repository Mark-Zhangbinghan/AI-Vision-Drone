"""
compare_optimizations.py - YOLO26 Ablation Study Runner
=============================================================
Serial execution of multiple training experiments with different
optimization strategies. Each experiment runs 50 epochs for quick
comparison. Generates comprehensive comparison charts and ranking.

Strategy A: Quick comparison (~1.5h per experiment, ~9h total)
  Exp 0: Baseline      - Default config, no optimizations
  Exp 1: +Freeze       - Progressive layer freezing
  Exp 2: +FocalLoss    - Stronger class weighting (cls_pw=1.0)
  Exp 3: +LabelSmooth  - Label smoothing regularization
  Exp 4: +OneCycleLR   - Cosine LR schedule
  Exp 5: +TTA          - Test-time augmentation (val only)
  Exp 6: ALL_COMBINED  - All optimizations together

Output:
  runs/compare/
    ├── 00_baseline/results.csv, best.pt, ...
    ├── 01_freeze/...
    ├── ...
    ├── compare_metrics_bar.png
    ├── compare_radar.png
    ├── ablation_heatmap.png
    ├── compare_training_curves.png
    ├── best_combo.yaml          ← Recommended optimal config
    └── comparison_report.txt    ← Ranking table

Usage:
    python compare_optimizations.py                   # Run all experiments
    python compare_optimizations.py --epochs 50       # Custom epoch count
    python compare_optimizations.py --skip 0 1 2     # Skip experiments
    python compare_optimizations.py --only-compare    # Only generate charts
"""

import sys
import json
import time
import shutil
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_drowning import get_compare_configs, DROWNING_CONFIG
from custom_plots import DrowningVisualizer


# ===========================================================================
#  Experiment runner
# ===========================================================================

def run_experiment(name, config, output_base):
    """
    Run a single training experiment.

    Args:
        name: Experiment display name (e.g., "Baseline", "Freeze")
        config: Full config dict
        output_base: Base directory for experiment outputs

    Returns:
        dict: Experiment results summary
    """
    from train_drowning import setup_model, build_train_args, ProgressiveFreezeCallback

    output_dir = Path(output_base) / config["name"]
    if output_dir.exists():
        print(f"[SKIP] {name}: output already exists at {output_dir}")
        # Check if results.csv exists (means training completed)
        if (output_dir / "results.csv").exists():
            return collect_experiment_results(output_dir, name)
        else:
            print(f"  Incomplete run detected, removing and restarting...")
            shutil.rmtree(output_dir)

    print("\n" + "=" * 70)
    print(f"  Experiment: {name}")
    print(f"  Output:     {output_dir}")
    print("=" * 70)

    # Update project/name to use compare directory
    config["project"] = str(output_base)
    config["name"] = config["name"]

    t_start = time.time()

    try:
        # Initialize model
        model, weights_source = setup_model(config)

        # Build training args
        train_args = build_train_args(config)

        # Register freeze callback if applicable
        freeze_stages = config.get("freeze_stages", [])
        if freeze_stages:
            freeze_cb = ProgressiveFreezeCallback(freeze_stages)
            model.add_callback("on_train_epoch_start", freeze_cb)
            print(f"[FREEZE] Progressive freezing: {len(freeze_stages)} stage(s)")

        # Train
        results = model.train(**train_args)

        elapsed = time.time() - t_start
        print(f"[DONE] {name}: completed in {elapsed:.0f}s ({elapsed/3600:.1f}h)")

    except Exception as e:
        print(f"[ERROR] {name} failed: {e}")
        import traceback
        traceback.print_exc()
        return {"name": name, "status": "failed", "error": str(e)}

    # Collect results
    return collect_experiment_results(output_dir, name)


def collect_experiment_results(exp_dir, exp_name):
    """Collect metrics from a completed experiment."""
    import csv

    results_csv = Path(exp_dir) / "results.csv"
    if not results_csv.exists():
        return {"name": exp_name, "status": "no_results"}

    data = {}
    with open(results_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {"name": exp_name, "status": "empty_results"}

    # Extract best metrics
    epochs = [int(r.get("epoch", i + 1)) for i, r in enumerate(rows)]

    def get_col(name):
        vals = [float(r[name]) for r in rows if name in r and r[name].strip()]
        return vals

    map50_vals = get_col("metrics/mAP50(B)")
    map50_95_vals = get_col("metrics/mAP50-95(B)")
    prec_vals = get_col("metrics/precision(B)")
    rec_vals = get_col("metrics/recall(B)")

    best_map50 = max(map50_vals) if map50_vals else 0.0
    best_map50_95 = max(map50_95_vals) if map50_95_vals else 0.0
    best_epoch = map50_vals.index(best_map50) + 1 if map50_vals else 0
    final_precision = prec_vals[-1] if prec_vals else 0.0
    final_recall = rec_vals[-1] if rec_vals else 0.0

    # Time info
    time_vals = get_col("time")
    total_time_h = sum(time_vals) / 3600.0 if time_vals else 0.0

    return {
        "name": exp_name,
        "status": "completed",
        "output_dir": str(exp_dir),
        "best_epoch": best_epoch,
        "best_map50": round(best_map50, 4),
        "best_map50_95": round(best_map50_95, 4),
        "final_precision": round(final_precision, 4),
        "final_recall": round(final_recall, 4),
        "total_time_h": round(total_time_h, 1),
        "epochs_trained": len(rows),
    }


# ===========================================================================
#  Compare runner
# ===========================================================================

def run_all_experiments(experiments_config, output_base, skip=None):
    """
    Run all experiments sequentially.

    Args:
        experiments_config: dict of {name: config} from get_compare_configs()
        output_base: Path to output directory
        skip: set of experiment indices to skip

    Returns:
        list of result dicts
    """
    skip = skip or set()
    results = []

    for idx, (name, config) in enumerate(experiments_config.items()):
        if idx in skip:
            print(f"[SKIP] Experiment {idx} ({name}): explicitly skipped")
            continue

        print(f"\n{'#' * 70}")
        print(f"#  Experiment {idx + 1}/{len(experiments_config)}: {name}")
        print(f"{'#' * 70}")

        result = run_experiment(name, config, output_base)
        results.append(result)

    return results


# ===========================================================================
#  Report generation
# ===========================================================================

def generate_report(results, output_base):
    """Generate ranking report and save best combo config."""
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    # Sort by mAP50-95
    ranked = sorted(
        [r for r in results if r.get("status") == "completed"],
        key=lambda x: x.get("best_map50_95", 0),
        reverse=True,
    )

    # Write text report
    report_path = output_base / "comparison_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  YOLO26 Drowning Detection - Optimization Comparison Report\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"{'Rank':<6} {'Experiment':<20} {'mAP50':<10} {'mAP50-95':<10} "
                f"{'Precision':<10} {'Recall':<10} {'BestEp':<8} {'Time(h)':<8}\n")
        f.write("-" * 70 + "\n")

        for rank, r in enumerate(ranked, 1):
            f.write(f"{rank:<6} {r['name']:<20} {r['best_map50']:<10.4f} "
                    f"{r['best_map50_95']:<10.4f} {r['final_precision']:<10.4f} "
                    f"{r['final_recall']:<10.4f} {r['best_epoch']:<8} "
                    f"{r.get('total_time_h', 0):<8.1f}\n")

        f.write("=" * 70 + "\n")

        if ranked:
            winner = ranked[0]
            f.write(f"\nBest combination: {winner['name']}\n")
            f.write(f"  mAP50:     {winner['best_map50']}\n")
            f.write(f"  mAP50-95:  {winner['best_map50_95']}\n")
            f.write(f"  Output:    {winner['output_dir']}\n")

        # Write comparison with baseline
        baseline = next((r for r in ranked if r["name"] == "Baseline"), None)
        if baseline and len(ranked) > 1:
            f.write("\n--- Improvement over Baseline ---\n")
            for r in ranked:
                if r["name"] == "Baseline":
                    continue
                delta = r["best_map50_95"] - baseline["best_map50_95"]
                f.write(f"  {r['name']:<20}: {delta:+.4f} mAP50-95\n")

    print(f"\n[REPORT] Saved: {report_path}")

    # Print to console
    with open(report_path, "r") as f:
        print(f.read())

    # Save best combo config as YAML
    if ranked:
        winner = ranked[0]
        best_config = {}
        # Find the experiment config that produced the winner
        exp_configs = get_compare_configs()
        for name, cfg in exp_configs.items():
            if cfg["name"] == winner.get("output_dir", "").split("/")[-1].split("\\")[-1]:
                best_config = {
                    "name": name,
                    "best_map50_95": winner["best_map50_95"],
                    "epochs": 300,  # Recommend full training
                    **{k: v for k, v in cfg.items() if not k.startswith("_")},
                }
                break

        if best_config:
            best_yaml = output_base / "best_combo.yaml"
            with open(best_yaml, "w", encoding="utf-8") as f:
                f.write(f"# Best optimization combination: {best_config.get('name', 'Unknown')}\n")
                f.write(f"# Achieved mAP50-95: {best_config.get('best_map50_95', 'N/A')}\n")
                f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                for k, v in sorted(best_config.items()):
                    if k in ("name", "best_map50_95"):
                        continue
                    f.write(f"{k}: {v}\n")
            print(f"[CONFIG] Best combo saved: {best_yaml}")

    return report_path


# ===========================================================================
#  Main
# ===========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="YOLO26 Optimization Ablation Study")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Epochs per experiment")
    parser.add_argument("--output", type=str, default=None,
                        help="Output base directory")
    parser.add_argument("--skip", type=int, nargs="+", default=[],
                        help="Experiment indices to skip (0=Baseline, 1=Freeze, ...)")
    parser.add_argument("--only-compare", action="store_true",
                        help="Only generate comparison charts (skip training)")
    parser.add_argument("--only", type=int, nargs="+", default=None,
                        help="Only run specific experiment indices")

    args = parser.parse_args()

    # Output directory
    if args.output:
        output_base = Path(args.output)
    else:
        output_base = PROJECT_ROOT / "runs" / "compare"
    output_base.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  YOLO26 Drowning Detection - Ablation Study (Strategy A)")
    print("=" * 70)
    print(f"  Epochs per experiment: {args.epochs}")
    print(f"  Output directory:      {output_base}")
    print(f"  Total experiments:     {len(get_compare_configs())}")
    print("=" * 70)

    # Get experiment configurations
    experiments = get_compare_configs()

    # Override epochs
    for name, config in experiments.items():
        config["epochs"] = args.epochs

    # Determine which experiments to run
    skip_set = set(args.skip)
    if args.only is not None:
        # Only run specified indices
        all_indices = set(range(len(experiments)))
        skip_set = all_indices - set(args.only)

    if not args.only_compare:
        results = run_all_experiments(experiments, output_base, skip=skip_set)
    else:
        # Collect existing results
        results = []
        for name, config in experiments.items():
            exp_dir = output_base / config["name"]
            if exp_dir.exists():
                result = collect_experiment_results(exp_dir, name)
                if result.get("status") == "completed":
                    results.append(result)

    if not results:
        print("No completed experiments found. Nothing to compare.")
        return

    # Generate report
    generate_report(results, output_base)

    # Generate comparison charts
    visualizer = DrowningVisualizer(save_dir=str(output_base))
    visualizer.create_comparison_plots(str(output_base))

    print("\n" + "=" * 70)
    print("  Ablation study complete!")
    print(f"  All outputs in: {output_base}")
    print("=" * 70)


if __name__ == "__main__":
    main()
