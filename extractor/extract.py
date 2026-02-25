"""
Extractor for Improved Diffusion .npz samples.

- Loads .npz produced by scripts/image_sample.py
- Exports images as PNG
- If label indices exist and labels.json is provided (or found), filenames include class names
- Writes manifest.csv: idx,filename,label_index,label_name

Usage:
  python extract_sample.py --npz_path /path/to/samples_*.npz \
                           --out_dir ./extracted \
                           --labels_json /path/to/labels.json \
                           --use_subdirs True
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def load_npz(npz_path: str):
    """
    Detect image and (optional) label arrays defensively.
    Expect images as uint8 [N, H, W, C] with C in {1, 3, 4}.
    """
    data = np.load(npz_path)
    keys = list(data.keys())
    if not keys:
        raise ValueError(f"No arrays found in: {npz_path}")

    img_key, lbl_key = None, None
    for k in keys:
        arr = data[k]
        if arr.dtype == np.uint8 and arr.ndim == 4 and arr.shape[-1] in (1, 3, 4):
            img_key = k
        elif arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer):
            lbl_key = k

    if img_key is None:
        img_key = keys[0]  # fall back to first array

    images = data[img_key]
    labels = data[lbl_key] if lbl_key is not None else None
    return images, labels


def load_label_map(labels_json_path: Optional[str]):
    """
    Load class index→name map.
    Accepts:
      - {"index_to_label": {"0": "cat", ...}}
      - {"label_to_index": {"cat": 0, ...}}  (inverted here)
    """
    if not labels_json_path:
        return None

    with open(labels_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    idx2name = meta.get("index_to_label")
    if idx2name is None:
        name2idx = meta.get("label_to_index")
        if name2idx is None:
            return None
        idx2name = {str(v): k for k, v in name2idx.items()}

    out = {}
    for k, v in idx2name.items():
        try:
            out[int(k)] = str(v)
        except Exception:
            pass
    return out


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    p = argparse.ArgumentParser(description="Export Improved Diffusion .npz samples to PNG + manifest.csv")
    p.add_argument("--npz_path", required=True, help="Path to samples_*.npz produced by image_sample.py")
    p.add_argument("--out_dir", default="extracted_samples", help="Output directory")
    p.add_argument("--labels_json", default="", help="Path to labels.json (optional). If omitted, tries next to the .npz")
    p.add_argument("--use_subdirs", type=lambda s: s.lower() in {"1", "true", "yes"}, default=False,
                   help="If true, create subfolders per class name")
    p.add_argument("--limit", type=int, default=0, help="Max images to export (0 = no limit)")
    args = p.parse_args()

    npz_path = Path(args.npz_path)
    out_dir = Path(args.out_dir)

    # Auto-discover labels.json beside the .npz if not provided
    labels_json_path = args.labels_json
    if not labels_json_path:
        candidate = npz_path.parent / "labels.json"
        if candidate.exists():
            labels_json_path = str(candidate)

    idx2name = load_label_map(labels_json_path) if labels_json_path else None

    images, labels = load_npz(str(npz_path))
    if labels is None:
        print("[info] No labels array found in .npz (unconditional or unlabeled).")
    else:
        print(f"[info] Labels shape: {labels.shape}")

    ensure_dir(out_dir)
    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        mf.write("idx,filename,label_index,label_name\n")

        N = images.shape[0]
        if args.limit and args.limit > 0:
            N = min(N, args.limit)

        for i in range(N):
            img = images[i]
            if img.ndim != 3 or img.shape[-1] not in (1, 3, 4):
                raise ValueError(f"Unexpected image shape at {i}: {img.shape} (expected HxWxC)")

            # PIL: squeeze single-channel to (H, W) for clean 'L' mode
            if img.shape[-1] == 1:
                img_to_save = img.squeeze(-1)
            else:
                img_to_save = img

            lbl_idx = int(labels[i]) if labels is not None else None
            lbl_name = idx2name.get(lbl_idx) if (lbl_idx is not None and idx2name is not None) else None

            if lbl_name is not None:
                safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in lbl_name)
                fname = f"{safe}_{i:06d}.png"
                save_dir = out_dir / safe if args.use_subdirs else out_dir
            elif lbl_idx is not None:
                fname = f"class{lbl_idx}_{i:06d}.png"
                save_dir = out_dir
            else:
                fname = f"sample_{i:06d}.png"
                save_dir = out_dir

            ensure_dir(save_dir)
            fp = save_dir / fname

            pil_img = Image.fromarray(img_to_save)
            pil_img.save(fp)

            mf.write(f"{i},{fp.relative_to(out_dir)},{'' if lbl_idx is None else lbl_idx},{'' if lbl_name is None else lbl_name}\n")

    print(f"[done] Images: {out_dir}")
    print(f"[done] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
