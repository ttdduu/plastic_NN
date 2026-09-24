#!/usr/bin/env python3
"""
Apply the Retinal Sampling Layer (RSL) transform to a folder of already-upscaled gratings.

This is analogous to `RSL_subset_creation.py`, but assumes inputs are already at
~2048x2048 so no upscaling is performed here.

Expected usage (default paths):
  - source: /home/tomasdu/repos/datasets/gratings-2048/val
  - inputs listed in manifest.csv as relative paths (e.g. images/xxx.png)

Output:
  - writes transformed (RSL output) images (default 256x256) to target dir,
    preserving relative paths from the manifest when present.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# Add the repo root to path so we can import `src.*` when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from src.data.transforms.RSL import RetinalCompression


def _read_manifest(manifest_path: Path) -> List[Path]:
    """Return list of relative image paths from a daCosta-style manifest.csv (filename column)."""
    rels: List[Path] = []
    with manifest_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if "filename" not in (reader.fieldnames or []):
            raise ValueError(f"manifest missing 'filename' column: {manifest_path}")
        for row in reader:
            rels.append(Path(row["filename"]))
    return rels


def _discover_images(source_dir: Path, exts: Tuple[str, ...]) -> List[Path]:
    """Fallback: recursively discover images when no manifest is used."""
    out: List[Path] = []
    for ext in exts:
        out.extend(source_dir.glob(f"**/*{ext}"))
    # return paths relative to source_dir
    return [p.relative_to(source_dir) for p in sorted(set(out))]


def _load_rgb_uint8(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    return rgb


def _apply_rsl(
    img_rgb_uint8: np.ndarray,
    compressor: RetinalCompression,
    *,
    fov: float,
    out_size: int,
    rsl_type: int,
    inv: int,
    apply_mask: bool,
) -> np.ndarray:
    transformed, mask = compressor.distort_image(
        image=img_rgb_uint8,
        fov=float(fov),
        out_size=int(out_size),
        inv=int(inv),
        type=int(rsl_type),
        series=1,
    )

    if apply_mask and mask is not None and len(mask) > 0:
        mask_reshaped = np.reshape(mask, [transformed.shape[0], transformed.shape[1]])
        mask_3d = np.repeat(mask_reshaped[:, :, np.newaxis], 3, axis=2)
        transformed = np.multiply(transformed, mask_3d)

    transformed = transformed.astype(np.uint8, copy=False)
    return transformed


def _process_one(args) -> bool:
    rel_path, source_dir, target_dir, compressor, fov, out_size, rsl_type, inv, apply_mask, expect_size, strict_size = args
    src = source_dir / rel_path
    dst = target_dir / rel_path
    try:
        img = _load_rgb_uint8(src)
        if expect_size is not None:
            h, w = img.shape[:2]
            if strict_size and (h != expect_size or w != expect_size):
                raise ValueError(f"Expected {expect_size}x{expect_size}, got {h}x{w} for {src}")
        out = _apply_rsl(
            img,
            compressor,
            fov=fov,
            out_size=out_size,
            rsl_type=rsl_type,
            inv=inv,
            apply_mask=apply_mask,
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        return True
    except Exception as e:
        print(f"[RSL_gratings] failed: {src} -> {dst}: {e}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=str, default="/home/tomasdu/repos/datasets/gratings-2048/val")
    ap.add_argument("--target-dir", type=str, default="/home/tomasdu/repos/datasets/gratings-2048-rsl/val")
    ap.add_argument("--manifest", type=str, default="manifest.csv",
                    help="Manifest filename inside source-dir. If missing, falls back to recursive glob.")
    ap.add_argument("--exts", type=str, default=".png,.jpg,.jpeg,.JPEG",
                    help="Comma-separated extensions for fallback discovery.")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--limit", type=int, default=0, help="If >0, process only first N images (for testing).")

    # RSL parameters
    ap.add_argument("--fov", type=float, default=20.0)
    ap.add_argument("--out-size", type=int, default=256)
    ap.add_argument("--type", type=int, default=1, help="0=cones, 1=ganglion (matches paper usage).")
    ap.add_argument("--inv", type=int, default=0, help="0=compress (normal->RSL), 1=decompress/normalize")
    ap.add_argument("--apply-mask", action="store_true", default=True)
    ap.add_argument("--no-apply-mask", dest="apply_mask", action="store_false")

    ap.add_argument("--expect-size", type=int, default=2048,
                    help="Expected input H=W. Set to 0 to disable checking.")
    ap.add_argument("--strict-size", action="store_true",
                    help="If set, error when input size != expect-size.")

    args = ap.parse_args()

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    manifest_path = source_dir / args.manifest
    exts = tuple(e.strip() for e in args.exts.split(",") if e.strip())

    if manifest_path.exists():
        rel_paths = _read_manifest(manifest_path)
    else:
        rel_paths = _discover_images(source_dir, exts=exts)
        if not rel_paths:
            raise FileNotFoundError(
                f"No images found in {source_dir} (manifest missing and glob found none)."
            )

    if args.limit and args.limit > 0:
        rel_paths = rel_paths[: int(args.limit)]

    expect_size: Optional[int] = int(args.expect_size) if int(args.expect_size) > 0 else None

    compressors = [RetinalCompression() for _ in range(max(1, int(args.workers)))]

    jobs = [
        (
            rel,
            source_dir,
            target_dir,
            compressors[i % len(compressors)],
            args.fov,
            args.out_size,
            args.type,
            args.inv,
            args.apply_mask,
            expect_size,
            bool(args.strict_size),
        )
        for i, rel in enumerate(rel_paths)
    ]

    ok = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = [ex.submit(_process_one, j) for j in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="RSL gratings"):
            if fut.result():
                ok += 1
            else:
                fail += 1

    print(f"Done. ok={ok}, fail={fail}, target={target_dir}")


if __name__ == "__main__":
    main()

