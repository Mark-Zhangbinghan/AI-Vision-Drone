"""
strip_ignore_labels.py - 从 Stage1 标签文件中删除 ignore 类 (cls_id=2)
========================================================
用于 nc=3→2 的数据集清理。

路径自动检测: 环境变量 SURVEIL_DATA_ROOT → 默认 /root/autodl-tmp/unified_surveillance_v1

Usage:
    python strip_ignore_labels.py
    python strip_ignore_labels.py --data /root/autodl-tmp/unified_surveillance_v1
"""
import os
import sys
from pathlib import Path


def find_labels_root():
    """三级路径检测: CLI -> 环境变量 -> 默认。"""
    # 尝试从命令行参数获取
    for i, arg in enumerate(sys.argv):
        if arg == "--data" and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1]) / "stage1_detect" / "labels"
    # 环境变量
    env = os.environ.get("SURVEIL_DATA_ROOT")
    if env:
        return Path(env) / "stage1_detect" / "labels"
    # 默认服务器路径
    return Path("/root/autodl-tmp/unified_surveillance_v1/stage1_detect/labels")


LABELS_ROOT = find_labels_root()

if not LABELS_ROOT.exists():
    print(f"[FATAL] 标签目录不存在: {LABELS_ROOT}")
    print(f"  请用 --data 指定数据集根目录或设置环境变量 SURVEIL_DATA_ROOT")
    sys.exit(1)

print(f"[INFO] 标签目录: {LABELS_ROOT}")

for split in ["train", "val", "test"]:
    d = LABELS_ROOT / split
    if not d.exists():
        print(f"[SKIP] {d} does not exist")
        continue
    cleaned = 0
    total_lines = 0
    for f in list(d.iterdir()):
        if not f.suffix == ".txt":
            continue
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except Exception as e:
            print(f"  [ERR] {f.name}: {e}")
            continue
        new = [l for l in lines if not l.startswith("2 ") and not l.startswith("2\t")]
        if len(new) != len(lines):
            f.write_text("\n".join(new) + ("\n" if new else ""), encoding="utf-8")
            cleaned += 1
            total_lines += (len(lines) - len(new))
    print(f"{split}: 清理 {cleaned} 文件, 删 {total_lines} 行 ignore 标签")

print("Done!")
