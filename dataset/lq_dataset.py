# time-discrete/improved-diffusion/datasets/lq_dataset.py
import argparse, json, csv, re, shutil, uuid
from pathlib import Path

# Supported image extensions
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def slugify_label(name: str) -> str:
    """
    Normalize a folder name into a consistent label:
    - Keep underscores
    - Lowercase everything
    - Replace spaces with '-'
    - Strip non-alphanumeric/underscore/hyphen chars
    - Collapse consecutive separators
    """
    name = name.strip().lower()
    name = name.replace(" ", "-")
    name = re.sub(r"[^a-z0-9_\-]+", "", name)
    name = re.sub(r"[-_]{2,}", "_", name)
    return name or "unknown"

def iter_images(root: Path):
    """
    Yield (image_path, class_name) for all valid images found under root.
    Class is determined by the immediate parent folder of the image.
    """
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            yield p, p.parent.name

def build_dataset(
    data_dir: Path,
    out_dir: Path,
    link: bool = False,
    keep_stem: bool = True,
    with_index_prefix: bool = False,
):
    """
    Build a flat dataset for improved-diffusion:
    - Discover all classes
    - Save label mappings (labels.json)
    - Copy or symlink images into a single folder with standardized names
    - Write a manifest.csv for traceability
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    labels_json = out_dir / "labels.json"

    # Discover classes (sorted for deterministic index assignment)
    classes = sorted({slugify_label(lbl) for _, lbl in iter_images(data_dir)})
    label_to_index = {lbl: i for i, lbl in enumerate(classes)}
    index_to_label = {i: lbl for lbl, i in label_to_index.items()}

    # Save labels.json
    with labels_json.open("w", encoding="utf-8") as f:
        json.dump(
            {"index_to_label": index_to_label, "label_to_index": label_to_index},
            f,
            indent=2,
            ensure_ascii=False,
        )

    used_names = set()
    with manifest_path.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["src_path", "dst_filename", "label", "label_index"])

        for src, raw_label in iter_images(data_dir):
            label = slugify_label(raw_label)
            lbl_idx = label_to_index[label]

            stem = src.stem if keep_stem else ""
            uid = uuid.uuid4().hex[:8]
            base = f"{label}_{stem}__{uid}" if stem else f"{label}__{uid}"

            if with_index_prefix:
                base = f"{lbl_idx:03d}_{base}"

            dst_name = f"{base}{src.suffix.lower()}"

            # Ensure unique filenames
            while dst_name in used_names or (out_dir / dst_name).exists():
                uid = uuid.uuid4().hex[:8]
                dst_name = f"{base}__{uid}{src.suffix.lower()}"
            used_names.add(dst_name)

            dst = out_dir / dst_name
            dst.parent.mkdir(parents=True, exist_ok=True)

            # Use symlinks if requested and supported, otherwise copy
            if link:
                try:
                    dst.symlink_to(src.resolve())  # Unix-friendly
                except Exception:
                    shutil.copy2(src, dst)
            else:
                shutil.copy2(src, dst)

            writer.writerow([str(src), dst_name, label, lbl_idx])

    return {
        "out_dir": str(out_dir),
        "count": len(used_names),
        "labels_json": str(labels_json),
        "manifest": str(manifest_path),
        "num_classes": len(classes),
    }

def main():
    ap = argparse.ArgumentParser(description="Prepare dataset for improved-diffusion")
    ap.add_argument("--root", type=str, default=".", help="Project root folder")
    ap.add_argument("--data", type=str, default="data", help="Input folder with class subfolders")
    ap.add_argument("--out", type=str, default="built_data", help="Output folder for the flat dataset")
    ap.add_argument("--link", action="store_true", help="Use symlinks instead of copying (Unix-like systems)")
    ap.add_argument("--no-stem", action="store_true", help="Do not include original file stem in output names")
    ap.add_argument("--with-index-prefix", action="store_true", help="Prefix class index to filename (ImageFolder-style)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    data_dir = (root / args.data).resolve()
    out_dir = (root / args.out).resolve()

    if not data_dir.exists():
        raise SystemExit(f"[ERROR] data_dir not found: {data_dir}")

    info = build_dataset(
        data_dir=data_dir,
        out_dir=out_dir,
        link=args.link,
        keep_stem=not args.no_stem,
        with_index_prefix=args.with_index_prefix,
    )

    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    main()
