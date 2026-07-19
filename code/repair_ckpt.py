"""
repair_ckpt.py — 从截断的 PyTorch checkpoint (last.pt) 中恢复模型权重
====================================================================
当训练中断导致 last.pt 的 ZIP 中央目录缺失时，
通过解析 ZIP local file headers 提取 data.pkl 的数据流，
再尝试用 torch.load 加载恢复后的数据。
"""

import io
import os
import struct
import sys
import zipfile
from pathlib import Path


def find_local_file_entries(data: bytes):
    """Parse raw bytes for ZIP local file header signatures (PK\x03\x04)."""
    entries = []
    offset = 0
    sig = b"PK\x03\x04"
    while True:
        pos = data.find(sig, offset)
        if pos == -1:
            break
        # Parse local file header (30 bytes fixed)
        try:
            # Fields at offsets: sig(4), version(2), flags(2), compression(2),
            #   mod_time(2), mod_date(2), crc32(4), comp_size(4), uncomp_size(4),
            #   name_len(2), extra_len(2)
            hdr = struct.unpack_from("<HHHHHIIIHH", data, pos + 4)
            compression = hdr[2]
            comp_size = hdr[6]      # compressed size
            uncomp_size = hdr[7]    # uncompressed size
            name_len = hdr[8]
            extra_len = hdr[9]

            name = data[pos + 30 : pos + 30 + name_len].decode("utf-8", errors="replace")
            data_start = pos + 30 + name_len + extra_len
            data_end = data_start + comp_size

            entries.append({
                "offset": pos,
                "filename": name,
                "compression": compression,
                "comp_size": comp_size,
                "uncomp_size": uncomp_size,
                "data_start": data_start,
                "data_end": data_end,
            })
        except Exception:
            pass
        offset = pos + 1
    return entries


def repair_last_pt(src_path: str, dst_path: str):
    """Attempt to recover model weights from a truncated last.pt."""
    print(f"[REPAIR] Reading: {src_path}")
    with open(src_path, "rb") as f:
        data = f.read()
    print(f"  File size: {len(data):,} bytes")

    # 1. Find all local file entries
    entries = find_local_file_entries(data)
    print(f"  Found {len(entries)} ZIP local entries:")
    for e in entries:
        size_ok = e["data_end"] <= len(data)
        flag = "+" if size_ok else "!TRUNCATED"
        print(f"    [{flag}] {e['filename']} (compressed={e['comp_size']}, "
              f"uncompressed={e['uncomp_size']}, comp_method={e['compression']})")

    # 2. Locate archive/data.pkl
    pkl_entry = None
    for e in entries:
        if "data.pkl" in e["filename"]:
            pkl_entry = e
            break
    if not pkl_entry:
        print("\n[FAIL] archive/data.pkl not found in ZIP entries")
        return False

    if pkl_entry["data_end"] > len(data):
        print(f"\n[FAIL] archive/data.pkl is truncated "
              f"({pkl_entry['comp_size']} bytes needed, "
              f"{len(data) - pkl_entry['data_start']} available)")
        return False

    # 3. Extract the raw compressed data
    raw_data = data[pkl_entry["data_start"] : pkl_entry["data_end"]]
    print(f"\n  Extracted {len(raw_data):,} bytes from archive/data.pkl")

    # 4. Decompress if needed (method 0 = stored, method 8 = deflated)
    if pkl_entry["compression"] == 0:
        decompressed = raw_data
    elif pkl_entry["compression"] == 8:
        try:
            decompressed = zlib_decompress(raw_data)
        except Exception as e:
            print(f"[FAIL] Decompression error: {e}")
            return False
    else:
        print(f"[FAIL] Unknown compression method: {pkl_entry['compression']}")
        return False

    print(f"  Decompressed: {len(decompressed):,} bytes")

    # 5. Try to load with torch
    try:
        import torch
        buffer = io.BytesIO(decompressed)
        ckpt = torch.load(buffer, map_location="cpu", weights_only=False)
        keys = list(ckpt.keys())
        print(f"\n[OK] Successfully loaded checkpoint!")
        print(f"  Keys: {keys}")

        # Save repaired checkpoint
        torch.save(ckpt, dst_path)
        print(f"\n[OK] Repaired checkpoint saved to: {dst_path}")
        return True

    except Exception as e:
        print(f"\n[FAIL] torch.load error: {e}")

        # 5b. Fallback: create a minimal .pt with just the pickle data wrapped in ZIP
        print("\n  Trying fallback: write raw pickle as ZIP...")
        try:
            with zipfile.ZipFile(dst_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("archive/data.pkl", decompressed)
            print(f"\n[OK] Wrapped pickle into new ZIP: {dst_path}")
            print("  Try loading this file with detect_gui.py")
            return True
        except Exception as e2:
            print(f"[FAIL] Fallback ZIP error: {e2}")
            return False


def zlib_decompress(data: bytes) -> bytes:
    """Decompress raw DEFLATE data (no zlib header)."""
    import zlib
    # Raw deflate: try with -15 window bits
    try:
        return zlib.decompress(data, -15)
    except zlib.error:
        pass
    # Try with standard zlib header
    try:
        return zlib.decompress(data)
    except zlib.error:
        pass
    # Try with inflate
    d = zlib.decompressobj(-15)
    result = d.decompress(data)
    result += d.flush()
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python repair_ckpt.py <path/to/last.pt> [output_path]")
        sys.exit(1)

    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src.replace(".pt", "_repaired.pt")

    if not os.path.exists(src):
        print(f"[ERROR] File not found: {src}")
        sys.exit(1)

    ok = repair_last_pt(src, dst)
    if ok:
        print(f"\nDone! You can now use: {dst}")
    else:
        print("\n[FAIL] Could not repair the checkpoint.")
        print("The file's data is too damaged to recover.")
        print("Use best.pt instead for detection — it contains the best epoch's weights.")
        sys.exit(1)
