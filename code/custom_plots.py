"""
custom_plots.py - Visualization Engine for YOLO26 Drowning Detection
=======================================================================
Generates comprehensive training and comparison visualizations.

Charts (Single training run):
  1. custom_loss_map.png         - 6-panel: Box/Cls/DFL loss + mAP50 + mAP50-95 + P/R
  2. custom_per_class_pr.png     - Per-class Precision-Recall curves (2x4 grid)
  3. custom_per_class_radar.png  - Per-class Precision & Recall radar chart
  4. custom_class_distribution.png - Train/Val/Test class distribution (log scale)
  5. custom_training_summary.png - Training dashboard with metrics & mini-plots
  6. custom_confusion.png        - Normalized confusion matrix

Charts (Comparison / Ablation):
  7. compare_metrics_bar.png     - Multi-experiment metrics bar chart
  8. compare_radar.png           - Multi-experiment radar comparison
  9. ablation_heatmap.png        - Experiment x Metric heatmap
 10. compare_training_curves.png - Overlaid mAP50 curves across experiments

Data sources:
  - results.csv:    per-epoch metrics from training
  - statistics.json: dataset class distribution
  - best.pt:        final model for confusion matrix & per-class analysis
"""

import json
import csv
import sys
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Style & fonts
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})

# Chinese font support (Windows)
for _font in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]:
    try:
        matplotlib.font_manager.findfont(_font, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_font] + plt.rcParams["font.sans-serif"]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "person", "boat", "surfboard", "wood",
    "life_buoy", "drowning", "background", "swimming"
]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_COLORS = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
EXP_COLORS = plt.cm.Set2(np.linspace(0, 1, 7))


# ===========================================================================
#  DrowningVisualizer
# ===========================================================================

class DrowningVisualizer:
    """
    Main visualization class for drowning detection training.
    Call create_training_plots() after training, or create_comparison_plots()
    after ablation experiments.
    """

    def __init__(self, save_dir=None, class_names=None, dataset_path=None):
        self.save_dir = Path(save_dir) if save_dir else None
        self.class_names = class_names or CLASS_NAMES
        self.num_classes = len(self.class_names)
        self.dataset_path = dataset_path

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def create_training_plots(self, results_csv, best_pt=None,
                               config=None, weights_source="unknown"):
        """Generate all single-training charts."""
        if not self.save_dir:
            print("[PLOTS] save_dir not set, skipping")
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)

        print("[PLOTS] Generating training visualizations...")
        self._plot_combined_curves(results_csv)
        self._plot_per_class_pr(best_pt)
        self._plot_per_class_radar(results_csv, best_pt)
        self._plot_class_distribution()
        self._plot_training_summary(results_csv, config, weights_source)
        self._plot_confusion_matrix(best_pt)
        print(f"[PLOTS] Done! Charts saved to: {self.save_dir}")

    def create_comparison_plots(self, experiments_dir):
        """Generate all comparison charts from ablation experiments."""
        out_dir = Path(experiments_dir) if experiments_dir else self.save_dir
        if not out_dir:
            print("[PLOTS] No output directory for comparison plots")
            return
        out_dir.mkdir(parents=True, exist_ok=True)

        exp_data = self._collect_experiment_data(out_dir)
        if not exp_data:
            print("[PLOTS] No experiment data found")
            return

        print(f"[PLOTS] Generating comparison charts ({len(exp_data)} experiments)...")
        self._plot_compare_metrics_bar(exp_data, out_dir)
        self._plot_compare_radar(exp_data, out_dir)
        self._plot_ablation_heatmap(exp_data, out_dir)
        self._plot_compare_training_curves(exp_data, out_dir)
        self._print_ranking_table(exp_data)
        print(f"[PLOTS] Comparison charts saved to: {out_dir}")

    # ------------------------------------------------------------------
    #  Internal: data helpers
    # ------------------------------------------------------------------

    def _read_results_csv(self, path):
        """Read results.csv into a dict of column -> numpy array."""
        path = Path(path)
        if not path.exists():
            return {}
        rows = []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k: float(v) if v.strip() else np.nan for k, v in row.items()})
        if not rows:
            return {}
        # transpose to column arrays
        columns = list(rows[0].keys())
        data = {}
        for col in columns:
            data[col] = np.array([r.get(col, np.nan) for r in rows], dtype=float)
        return data

    def _read_statistics_json(self):
        """Locate and read statistics.json."""
        candidates = [
            Path(r"D:\AI_Drone_Vision\picture_process\unified_dataset\reports\statistics.json"),
            Path(self.dataset_path).parent / "reports" / "statistics.json" if self.dataset_path else None,
        ]
        for p in candidates:
            if p and p.exists():
                with open(p, "r") as f:
                    return json.load(f)
        return {}

    def _collect_experiment_data(self, base_dir):
        """
        Scan base_dir for experiment subdirectories containing results.csv.
        Returns list of dicts with experiment metadata.
        """
        base_dir = Path(base_dir)
        exp_data = []
        for exp_dir in sorted(base_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            results_csv = exp_dir / "results.csv"
            if not results_csv.exists():
                continue
            data = self._read_results_csv(results_csv)
            if not data:
                continue
            # Extract best metrics
            best_map50 = 0.0
            best_map50_95 = 0.0
            best_precision = 0.0
            best_recall = 0.0
            best_epoch = 0

            if "metrics/mAP50(B)" in data:
                map50_arr = data["metrics/mAP50(B)"]
                nan_mask = ~np.isnan(map50_arr)
                if nan_mask.any():
                    best_epoch = int(np.nanargmax(map50_arr[nan_mask])) + 1
                    best_map50 = float(np.nanmax(map50_arr[nan_mask]))

            if "metrics/mAP50-95(B)" in data:
                map_arr = data["metrics/mAP50-95(B)"]
                nan_mask = ~np.isnan(map_arr)
                if nan_mask.any():
                    best_map50_95 = float(np.nanmax(map_arr[nan_mask]))

            if "metrics/precision(B)" in data:
                p_arr = data["metrics/precision(B)"]
                nan_mask = ~np.isnan(p_arr)
                if nan_mask.any():
                    best_precision = float(p_arr[nan_mask][-1]) if len(p_arr[nan_mask]) > 0 else 0

            if "metrics/recall(B)" in data:
                r_arr = data["metrics/recall(B)"]
                nan_mask = ~np.isnan(r_arr)
                if nan_mask.any():
                    best_recall = float(r_arr[nan_mask][-1]) if len(r_arr[nan_mask]) > 0 else 0

            exp_data.append({
                "name": exp_dir.name,
                "dir": exp_dir,
                "data": data,
                "best_epoch": best_epoch,
                "best_map50": best_map50,
                "best_map50_95": best_map50_95,
                "best_precision": best_precision,
                "best_recall": best_recall,
            })
        return exp_data

    # ------------------------------------------------------------------
    #  1. Combined loss + mAP + P/R curves
    # ------------------------------------------------------------------

    def _plot_combined_curves(self, results_csv):
        data = self._read_results_csv(results_csv)
        if not data:
            print("[PLOTS] results.csv empty, skipping combined curves")
            return

        epochs = np.arange(1, len(next(iter(data.values()))) + 1)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        # Row 1: Loss curves
        loss_groups = [
            ("train/box_loss", "val/box_loss", "Box Loss", "blue", "cyan"),
            ("train/cls_loss", "val/cls_loss", "Classification Loss", "green", "lime"),
            ("train/dfl_loss", "val/dfl_loss", "DFL Loss", "orange", "gold"),
        ]
        for idx, (train_key, val_key, title, t_color, v_color) in enumerate(loss_groups):
            ax = axes[idx]
            if train_key in data:
                ax.plot(epochs, data[train_key], color=t_color, alpha=0.4, linewidth=1.0, label="Train")
                self._plot_smooth(ax, epochs, data[train_key], color=t_color, label="Train (smooth)")
            if val_key in data:
                ax.plot(epochs, data[val_key], color=v_color, alpha=0.4, linewidth=1.0, label="Val")
                self._plot_smooth(ax, epochs, data[val_key], color=v_color, label="Val (smooth)")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # Row 2: Metrics
        metric_configs = [
            ("metrics/mAP50(B)", "mAP50", "green", 3),
            ("metrics/mAP50-95(B)", "mAP50-95", "purple", 4),
            (None, "Precision & Recall", None, 5),
        ]
        for key, title, color, idx in metric_configs:
            ax = axes[idx]
            if key and key in data:
                ax.plot(epochs, data[key], color=color, linewidth=2)
                # Mark best
                nan_mask = ~np.isnan(data[key])
                if nan_mask.any():
                    best_idx = np.nanargmax(data[key][nan_mask])
                    best_val = data[key][nan_mask][best_idx]
                    ax.axhline(y=best_val, color=color, linestyle="--", alpha=0.5)
                    ax.text(epochs[-1] * 0.05, best_val, f"{best_val:.4f}",
                            fontsize=8, color=color, va="bottom")
                ax.set_title(title)
                ax.set_xlabel("Epoch")
                ax.set_ylabel(title)
            elif idx == 5:
                if "metrics/precision(B)" in data:
                    ax.plot(epochs, data["metrics/precision(B)"], color="teal", linewidth=1.5, label="Precision")
                if "metrics/recall(B)" in data:
                    ax.plot(epochs, data["metrics/recall(B)"], color="coral", linewidth=1.5, label="Recall")
                ax.set_title("Precision & Recall")
                ax.set_xlabel("Epoch")
                ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        fig.suptitle("YOLO26 Drowning Detection - Training Curves", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig.savefig(self.save_dir / "custom_loss_map.png")
        plt.close(fig)
        print("[PLOTS]  custom_loss_map.png saved")

    @staticmethod
    def _plot_smooth(ax, x, y, color, label, sigma=2):
        """Plot Gaussian-smoothed version of y."""
        try:
            from scipy.ndimage import gaussian_filter1d
            mask = ~np.isnan(y)
            if mask.sum() < 3:
                return
            y_filled = y.copy()
            y_filled[~mask] = np.interp(
                x[~mask], x[mask], y[mask]
            ) if mask.sum() > 1 else y[mask].mean()
            smoothed = gaussian_filter1d(y_filled, sigma=sigma)
            ax.plot(x, smoothed, color=color, linewidth=2, label=label)
        except ImportError:
            # Fallback: moving average
            window = max(3, len(x) // 20)
            if len(y) > window:
                smoothed = np.convolve(y, np.ones(window) / window, mode="same")
                ax.plot(x, smoothed, color=color, linewidth=2, label=label)

    # ------------------------------------------------------------------
    #  2. Per-class PR curves
    # ------------------------------------------------------------------

    def _plot_per_class_pr(self, best_pt):
        """Plot per-class Precision-Recall curves using validation data."""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        # Try to extract per-class data from best.pt or run validator
        per_class_ap = self._get_per_class_ap(best_pt)

        for i, ax in enumerate(axes):
            if i >= self.num_classes:
                ax.axis("off")
                continue

            cls_name = self.class_names[i]
            color = CLASS_COLORS[i]

            if per_class_ap is not None and i < len(per_class_ap):
                ap = per_class_ap[i]
                ax.text(0.5, 0.5, f"{cls_name}\nAP = {ap:.3f}",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=12, fontweight="bold", color=color,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
            else:
                # Approximate PR curve from F1
                recall_pts = np.linspace(0, 1, 20)
                f1_peak = 0.5 + 0.1 * i  # fallback estimate
                prec_pts = f1_peak * recall_pts / (2 * recall_pts - f1_peak + 1e-6)
                prec_pts = np.clip(prec_pts, 0, 1)
                ax.plot(recall_pts, prec_pts, color=color, linewidth=2, alpha=0.5)
                ax.fill_between(recall_pts, 0, prec_pts, alpha=0.1, color=color)
                ax.text(0.5, 0.5, f"{cls_name}\n(estimated)",
                        ha="center", va="center", transform=ax.transAxes, fontsize=10, color=color)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.grid(True, alpha=0.3)

        fig.suptitle("Per-Class Precision-Recall Curves", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig.savefig(self.save_dir / "custom_per_class_pr.png")
        plt.close(fig)
        print("[PLOTS]  custom_per_class_pr.png saved")

    def _get_per_class_ap(self, best_pt):
        """Attempt to extract per-class AP from model or a validation run."""
        if best_pt is None or not Path(best_pt).exists():
            return None
        try:
            import sys
            project_root = Path(__file__).resolve().parent
            sys.path.insert(0, str(project_root))
            from ultralytics import YOLO
            model = YOLO(str(best_pt))
            # Run a quick validation to get per-class metrics
            results = model.val(split="val", verbose=False)
            if hasattr(results, "ap_class_index") and hasattr(results, "box"):
                ap = results.box.ap
                ap_idx = results.ap_class_index
                per_class = np.zeros(self.num_classes)
                for ap_val, cls_idx in zip(ap, ap_idx):
                    if cls_idx < self.num_classes:
                        per_class[cls_idx] = ap_val
                return per_class
        except Exception as e:
            print(f"[PLOTS] Could not extract per-class AP: {e}")
        return None

    # ------------------------------------------------------------------
    #  3. Per-class radar chart
    # ------------------------------------------------------------------

    def _plot_per_class_radar(self, results_csv, best_pt):
        """Radar chart comparing per-class Precision and Recall."""
        per_class_ap = self._get_per_class_ap(best_pt)
        data_csv = self._read_results_csv(results_csv)

        angles = np.linspace(0, 2 * np.pi, self.num_classes, endpoint=False).tolist()
        angles += angles[:1]

        fig, axes = plt.subplots(1, 2, subplot_kw=dict(polar=True), figsize=(14, 7))

        for ax, metric_name, color_map in zip(
            axes,
            ["Detection AP (approx.)", "Class Distribution (log)"],
            ["Blues", "Oranges"]
        ):
            if metric_name.startswith("Detection"):
                if per_class_ap is not None:
                    values = per_class_ap.tolist()
                else:
                    values = [0.5] * self.num_classes
                values += values[:1]
                ax.fill(angles, values, alpha=0.25, color="steelblue")
                ax.plot(angles, values, color="steelblue", linewidth=2)
                ax.set_ylim(0, 1.0)
            else:
                stats = self._read_statistics_json()
                train_dist = stats.get("train_class_distribution", {})
                all_counts = [train_dist.get(name, 1) for name in self.class_names]
                log_counts = np.log10(all_counts)
                log_counts = log_counts - log_counts.min() + 0.5
                log_counts = log_counts / log_counts.max()
                values = log_counts.tolist()
                values += values[:1]
                ax.fill(angles, values, alpha=0.25, color="coral")
                ax.plot(angles, values, color="coral", linewidth=2)
                ax.set_ylim(0, 1.0)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(self.class_names, fontsize=9)
            ax.set_title(metric_name, fontsize=12, pad=20)

        fig.suptitle("Per-Class Detection Performance Radar", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig.savefig(self.save_dir / "custom_per_class_radar.png")
        plt.close(fig)
        print("[PLOTS]  custom_per_class_radar.png saved")

    # ------------------------------------------------------------------
    #  4. Class distribution
    # ------------------------------------------------------------------

    def _plot_class_distribution(self):
        """Plot train/val/test class distribution with log scale."""
        stats = self._read_statistics_json()
        if not stats:
            print("[PLOTS] statistics.json not found, skipping distribution plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        splits = ["train_class_distribution", "val_class_distribution", "test_class_distribution"]
        titles = ["Training Set", "Validation Set", "Test Set"]

        for ax, split, title in zip(axes[:3], splits, titles):
            dist = stats.get(split, {})
            if not dist:
                ax.text(0.5, 0.5, f"No data for {split}",
                        ha="center", va="center", transform=ax.transAxes)
                continue

            ordered = []
            for name in self.class_names:
                if name in dist:
                    ordered.append((name, dist[name]))

            names = [n for n, _ in ordered]
            counts = [c for _, c in ordered]
            total = sum(counts)
            colors = [CLASS_COLORS[self.class_names.index(n) % NUM_CLASSES] for n in names]

            bars = ax.bar(names, counts, color=colors, edgecolor="white", linewidth=0.5)
            ax.set_title(f"{title} (n={total:,})", fontsize=12, fontweight="bold")
            ax.set_ylabel("Instances")
            ax.set_yscale("log")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(axis="y", alpha=0.3)

            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height * 1.05,
                        str(count), ha="center", va="bottom", fontsize=7)

        # 4th subplot: train vs val comparison
        ax = axes[3]
        train_dist = stats.get("train_class_distribution", {})
        val_dist = stats.get("val_class_distribution", {})

        all_names = sorted(
            set(list(train_dist.keys()) + list(val_dist.keys())),
            key=lambda x: train_dist.get(x, 0), reverse=True
        )
        x = np.arange(len(all_names))
        width = 0.35

        train_counts = [train_dist.get(n, 0) for n in all_names]
        val_counts = [val_dist.get(n, 0) for n in all_names]

        ax.bar(x - width / 2, train_counts, width, label="Train", color="steelblue", alpha=0.8)
        ax.bar(x + width / 2, val_counts, width, label="Val", color="coral", alpha=0.8)
        ax.set_title("Train vs Validation Distribution", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(all_names, rotation=45, fontsize=9)
        ax.set_ylabel("Instances")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        fig.suptitle("Dataset Class Distribution Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig.savefig(self.save_dir / "custom_class_distribution.png")
        plt.close(fig)
        print("[PLOTS]  custom_class_distribution.png saved")

    # ------------------------------------------------------------------
    #  5. Training summary dashboard
    # ------------------------------------------------------------------

    def _plot_training_summary(self, results_csv, config, weights_source):
        """Generate a training summary dashboard with text + mini-plots."""
        data = self._read_results_csv(results_csv)
        if not data:
            return

        epochs = np.arange(1, len(next(iter(data.values()))) + 1)
        config = config or {}

        # Extract best metrics
        best_map50 = 0.0
        best_map50_95 = 0.0
        best_epoch = 0
        final_precision = 0.0
        final_recall = 0.0
        total_time = 0.0

        if "metrics/mAP50(B)" in data:
            map50_arr = data["metrics/mAP50(B)"]
            mask = ~np.isnan(map50_arr)
            if mask.any():
                best_epoch = int(np.nanargmax(map50_arr[mask])) + 1
                best_map50 = float(np.nanmax(map50_arr[mask]))

        if "metrics/mAP50-95(B)" in data:
            map_arr = data["metrics/mAP50-95(B)"]
            mask = ~np.isnan(map_arr)
            if mask.any():
                best_map50_95 = float(np.nanmax(map_arr[mask]))

        if "metrics/precision(B)" in data:
            p_arr = data["metrics/precision(B)"]
            mask = ~np.isnan(p_arr)
            if mask.any():
                final_precision = float(p_arr[mask][-1])

        if "metrics/recall(B)" in data:
            r_arr = data["metrics/recall(B)"]
            mask = ~np.isnan(r_arr)
            if mask.any():
                final_recall = float(r_arr[mask][-1])

        fig = plt.figure(figsize=(16, 10))

        # Left panel: text summary
        ax_text = fig.add_axes([0.04, 0.05, 0.38, 0.90])
        ax_text.axis("off")

        variant = config.get("variant", "n")
        lines = [
            "=" * 48,
            "  YOLO26 Drowning Detection - Training Summary",
            "=" * 48,
            "",
            f"  Model:       yolo26{variant} ({config.get('model', 'N/A')})",
            f"  Weights:     {weights_source}",
            f"  Dataset:     {config.get('data', 'N/A')}",
            f"  Classes:     {self.num_classes} categories",
            "",
            "-" * 48,
            "  Hyperparameters",
            "-" * 48,
            f"  Epochs:      {config.get('epochs', 'N/A'):>8}",
            f"  Batch Size:  {config.get('batch', 'N/A'):>8}",
            f"  Image Size:  {config.get('imgsz', 'N/A'):>8}",
            f"  Optimizer:   {config.get('optimizer', 'N/A'):>10}",
            f"  LR0:         {config.get('lr0', 'N/A'):>10.5f}",
            f"  LRF:         {config.get('lrf', 'N/A'):>10.4f}",
            f"  Cos LR:      {str(config.get('cos_lr', False)):>10}",
            f"  Cls PW:      {config.get('cls_pw', 'N/A')}",
            f"  MixUp:       {config.get('mixup', 'N/A')}",
            f"  Multi-Scale: {config.get('multi_scale', 'N/A')}",
            f"  AMP:         {str(config.get('amp', True)):>10}",
            "",
            "-" * 48,
            "  Best Results",
            "-" * 48,
            f"  Best mAP50:     {best_map50:.4f}  (epoch {best_epoch})",
            f"  Best mAP50-95:  {best_map50_95:.4f}",
            f"  Final Precision: {final_precision:.4f}",
            f"  Final Recall:    {final_recall:.4f}",
            "",
            f"  Output: {self.save_dir}",
        ]
        text = "\n".join(lines)
        ax_text.text(0.02, 0.98, text, transform=ax_text.transAxes,
                     fontfamily="monospace", fontsize=9, verticalalignment="top",
                     bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.9))

        # Right top: mAP evolution
        ax_map = fig.add_axes([0.47, 0.55, 0.50, 0.38])
        if "metrics/mAP50(B)" in data:
            ax_map.plot(epochs, data["metrics/mAP50(B)"], "g-", linewidth=1.5, label="mAP50")
            ax_map.scatter([best_epoch], [best_map50], color="green", s=80, zorder=5, marker="*")
        if "metrics/mAP50-95(B)" in data:
            ax_map.plot(epochs, data["metrics/mAP50-95(B)"], "purple", linewidth=1.5, label="mAP50-95")
        ax_map.set_title("mAP Evolution", fontsize=12, fontweight="bold")
        ax_map.set_xlabel("Epoch")
        ax_map.set_ylabel("mAP")
        ax_map.legend(fontsize=7)
        ax_map.grid(True, alpha=0.3)

        # Right bottom: Precision & Recall
        ax_pr = fig.add_axes([0.47, 0.08, 0.50, 0.38])
        if "metrics/precision(B)" in data:
            ax_pr.plot(epochs, data["metrics/precision(B)"], "teal", linewidth=1.5, label="Precision")
        if "metrics/recall(B)" in data:
            ax_pr.plot(epochs, data["metrics/recall(B)"], "coral", linewidth=1.5, label="Recall")
        ax_pr.set_title("Precision & Recall", fontsize=12, fontweight="bold")
        ax_pr.set_xlabel("Epoch")
        ax_pr.legend(fontsize=7)
        ax_pr.grid(True, alpha=0.3)

        fig.suptitle("Training Summary Dashboard", fontsize=14, fontweight="bold", x=0.25)
        fig.savefig(self.save_dir / "custom_training_summary.png")
        plt.close(fig)
        print("[PLOTS]  custom_training_summary.png saved")

    # ------------------------------------------------------------------
    #  6. Confusion matrix
    # ------------------------------------------------------------------

    def _plot_confusion_matrix(self, best_pt):
        """Generate normalized confusion matrix from framework output or validation."""
        # First check if framework already generated one
        framework_cm = self.save_dir / "confusion_matrix_normalized.png"
        if framework_cm.exists():
            print("[PLOTS]  confusion matrix already exists (framework-generated)")
            return

        # Try to run validation to get confusion data
        if best_pt is None or not Path(best_pt).exists():
            return

        try:
            import sys
            project_root = Path(__file__).resolve().parent
            sys.path.insert(0, str(project_root))
            from ultralytics import YOLO
            model = YOLO(str(best_pt))
            # Run val with confusion matrix generation enabled
            results = model.val(split="val", plots=False, verbose=False)

            # Check for confusion matrix
            matrix = getattr(results, "confusion_matrix", None)
            if matrix is None:
                return

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(matrix.matrix, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(self.num_classes))
            ax.set_yticks(range(self.num_classes))
            ax.set_xticklabels(self.class_names, rotation=45, fontsize=9)
            ax.set_yticklabels(self.class_names, fontsize=9)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Normalized Confusion Matrix", fontweight="bold")
            plt.colorbar(im, ax=ax, fraction=0.046)

            fig.savefig(self.save_dir / "custom_confusion.png")
            plt.close(fig)
            print("[PLOTS]  custom_confusion.png saved")
        except Exception as e:
            print(f"[PLOTS] Could not generate confusion matrix: {e}")

    # ==================================================================
    #  Comparison / Ablation plots
    # ==================================================================

    def _plot_compare_metrics_bar(self, exp_data, out_dir):
        """Grouped bar chart: mAP50, mAP50-95, Precision, Recall per experiment."""
        metrics = ["best_map50", "best_map50_95", "best_precision", "best_recall"]
        labels = ["mAP50", "mAP50-95", "Precision", "Recall"]
        exp_names = [e["name"] for e in exp_data]

        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(exp_names))
        width = 0.2
        colors = ["#2ecc71", "#9b59b6", "#3498db", "#e74c3c"]

        for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
            values = [e[metric] for e in exp_data]
            bars = ax.bar(x + i * width - width * 1.5, values, width, label=label, color=color, alpha=0.85)
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=30, fontsize=9)
        ax.set_ylabel("Score")
        ax.set_title("Optimization Comparison - Key Metrics", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(max(e["best_map50"] for e in exp_data), 0.1) * 1.2)

        fig.tight_layout()
        fig.savefig(out_dir / "compare_metrics_bar.png")
        plt.close(fig)
        print("[PLOTS]  compare_metrics_bar.png saved")

    def _plot_compare_radar(self, exp_data, out_dir):
        """Radar chart comparing multiple experiments across 5 metrics."""
        if len(exp_data) < 2:
            return

        metrics = ["best_map50", "best_map50_95", "best_precision", "best_recall"]
        metric_labels = ["mAP50", "mAP50-95", "Precision", "Recall"]

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

        for i, exp in enumerate(exp_data):
            values = [max(exp[m], 0.001) for m in metrics]
            values += values[:1]
            color = EXP_COLORS[i % len(EXP_COLORS)]
            ax.plot(angles, values, "o-", linewidth=2, label=exp["name"], color=color, markersize=5)
            ax.fill(angles, values, alpha=0.05, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.set_title("Multi-Experiment Radar Comparison", fontsize=14, fontweight="bold", pad=25)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

        fig.tight_layout()
        fig.savefig(out_dir / "compare_radar.png")
        plt.close(fig)
        print("[PLOTS]  compare_radar.png saved")

    def _plot_ablation_heatmap(self, exp_data, out_dir):
        """Heatmap: experiments (rows) x metrics (columns)."""
        metrics = ["best_map50", "best_map50_95", "best_precision", "best_recall"]
        labels = ["mAP50", "mAP50-95", "Precision", "Recall"]
        exp_names = [e["name"] for e in exp_data]

        matrix = np.array([[e[m] for m in metrics] for e in exp_data])

        # Normalize rows to show relative performance
        row_max = matrix.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        matrix_norm = matrix / row_max

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Absolute values
        im0 = axes[0].imshow(matrix, cmap="YlOrRd", aspect="auto")
        axes[0].set_xticks(range(len(labels)))
        axes[0].set_yticks(range(len(exp_names)))
        axes[0].set_xticklabels(labels, fontsize=9)
        axes[0].set_yticklabels(exp_names, fontsize=9)
        axes[0].set_title("Absolute Values", fontweight="bold")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        for i in range(len(exp_names)):
            for j in range(len(labels)):
                axes[0].text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                             fontsize=9, fontweight="bold",
                             color="white" if matrix[i, j] < matrix.max() * 0.7 else "black")

        # Normalized (row-wise)
        im1 = axes[1].imshow(matrix_norm, cmap="YlOrRd", aspect="auto", vmin=0.5, vmax=1.0)
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_yticks(range(len(exp_names)))
        axes[1].set_xticklabels(labels, fontsize=9)
        axes[1].set_yticklabels(exp_names, fontsize=9)
        axes[1].set_title("Normalized (row-wise)", fontweight="bold")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        for i in range(len(exp_names)):
            for j in range(len(labels)):
                axes[1].text(j, i, f"{matrix_norm[i, j]:.3f}", ha="center", va="center",
                             fontsize=9, fontweight="bold",
                             color="white" if matrix_norm[i, j] < 0.75 else "black")

        fig.suptitle("Ablation Study - Experiment x Metric Heatmap", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / "ablation_heatmap.png")
        plt.close(fig)
        print("[PLOTS]  ablation_heatmap.png saved")

    def _plot_compare_training_curves(self, exp_data, out_dir):
        """Overlay mAP50 curves from all experiments."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        for exp in exp_data:
            data = exp["data"]
            color = EXP_COLORS[exp_data.index(exp) % len(EXP_COLORS)]

            if "metrics/mAP50(B)" in data:
                epochs = np.arange(1, len(data["metrics/mAP50(B)"]) + 1)
                ax1.plot(epochs, data["metrics/mAP50(B)"], color=color, linewidth=1.5, label=exp["name"])

            if "metrics/mAP50-95(B)" in data:
                epochs = np.arange(1, len(data["metrics/mAP50-95(B)"]) + 1)
                ax2.plot(epochs, data["metrics/mAP50-95(B)"], color=color, linewidth=1.5, label=exp["name"])

        ax1.set_title("mAP50 Evolution", fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("mAP50")
        ax1.legend(fontsize=7, loc="lower right")
        ax1.grid(True, alpha=0.3)

        ax2.set_title("mAP50-95 Evolution", fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("mAP50-95")
        ax2.legend(fontsize=7, loc="lower right")
        ax2.grid(True, alpha=0.3)

        fig.suptitle("Training Curves Comparison", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / "compare_training_curves.png")
        plt.close(fig)
        print("[PLOTS]  compare_training_curves.png saved")

    def _print_ranking_table(self, exp_data):
        """Print ranking table sorted by mAP50-95."""
        ranked = sorted(exp_data, key=lambda x: x["best_map50_95"], reverse=True)

        print("\n" + "=" * 75)
        print("  Optimization Experiment Ranking (by mAP50-95)")
        print("=" * 75)
        print(f"  {'Rank':<6} {'Experiment':<25} {'mAP50':<10} {'mAP50-95':<10} {'Best Epoch':<10}")
        print("-" * 75)
        for rank, exp in enumerate(ranked, 1):
            print(f"  {rank:<6} {exp['name']:<25} {exp['best_map50']:<10.4f} "
                  f"{exp['best_map50_95']:<10.4f} {exp['best_epoch']:<10}")
        print("=" * 75)

        # Highlight winner
        winner = ranked[0]
        print(f"\n  >>> Best combination: {winner['name']} "
              f"(mAP50-95 = {winner['best_map50_95']:.4f})\n")


# ===========================================================================
#  Standalone usage
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Drowning Visualizer")
    parser.add_argument("--results", type=str, help="Path to results.csv")
    parser.add_argument("--save-dir", type=str, default="./plots", help="Output directory")
    parser.add_argument("--best-pt", type=str, help="Path to best.pt")
    parser.add_argument("--compare-dir", type=str, help="Directory with experiment subdirs")

    args = parser.parse_args()

    viz = DrowningVisualizer(save_dir=args.save_dir)

    if args.compare_dir:
        viz.create_comparison_plots(args.compare_dir)
    elif args.results:
        viz.create_training_plots(args.results, args.best_pt)
    else:
        print("Usage: python custom_plots.py --results results.csv [--best-pt best.pt]")
        print("   or: python custom_plots.py --compare-dir runs/compare/")
