#!/usr/bin/env python3
"""
Generate daCosta-style sinring stimuli with arbitrary spatial-frequency ranges.

Generates two folders per eccentricity under ``<out-root>/``:

    undistorted/<ecc>_Ecc/{idx}_{label}_{ecc}.jpg   (CNN condition, 256×256)
    distorted/<ecc>_Ecc/{idx}_{label}_{ecc}.jpg      (RSL condition, 256×256)

Sinring formula (verified against OSF sinrings.h5):
    pixel_value = bg + contrast · mask · sin(2·n·e·θ)

where:
    n      — frequency index (paper uses n ∈ {2,5,8,11,14,17,20,23})
    e      — eccentricity in degrees
    label  — 2·n  (the filename label)
    θ      — polar angle at each pixel (radians, from arctan2)
    mask   — Mask_Norm/<ecc>.jpg from the OSF dataset (ring envelope)
    bg     — 128 (gray background in uint8)
    contrast — 127

The factor e ensures equal spatial frequency in cycles/degree at all
eccentricities: the angular SF (cycles/radian of θ) is n·e, so cycles/degree
of arc = (n·e) / e = n (constant across eccentricities).

Distorted images are produced by passing the undistorted sinring (rendered
directly at out_size×out_size) through RetinalCompression.distort_image(), then
optionally applying a per-pixel affine correction (corrected = a*distorted + b)
learned from sinrings.h5 so the output matches the paper's OSF stimuli.

Use --source hybrid to get paper's 8 (1_4..8_46) from h5 (exact match) plus
extended SFs via RSL. Use --source h5 for paper's 8 only.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.transforms.RSL import RetinalCompression

# Paper's 8 spatial frequencies (n values) in sinrings.h5
PAPER_N = [2, 5, 8, 11, 14, 17, 20, 23]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StimSpec:
    e_deg: float
    n: int

    @property
    def label(self) -> int:
        """Filename label used by the OSF dataset (= 2·n)."""
        return 2 * self.n

    @property
    def cpr(self) -> float:
        """Cycles per degree of arc at the ring = n (angular SF n·e / eccentricity e)."""
        return float(self.n)

    @property
    def cycles_around_ring(self) -> float:
        """Total cycles in 2π radians = 2·n·e."""
        return 2.0 * self.n * self.e_deg


def fmt_ecc(e_deg: float) -> str:
    """'1' not '1.0', but '2.8' stays '2.8'."""
    return f"{float(e_deg):g}"


def _n_range(n_min: int, n_max: int, n_step: int) -> list:
    if n_step <= 0:
        raise ValueError("step must be > 0")
    return list(range(n_min, n_max + 1, n_step))


def _to_unit_float(u8: np.ndarray) -> np.ndarray:
    """Map uint8 [0..255] (bg ~128) to float ~[-1..1]."""
    return (u8.astype(np.float32) - 128.0) / 127.0


def _from_unit_float(x: np.ndarray) -> np.ndarray:
    """Map float ~[-1..1] back to uint8 [0..255]."""
    return np.clip(x * 127.0 + 128.0, 0, 255).astype(np.uint8)


def _fit_affine_per_pixel(D: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-pixel affine C = a*D + b via least squares. D, C shape (N, H, W)."""
    Dm = D.mean(axis=0)
    Cm = C.mean(axis=0)
    num = np.sum((D - Dm) * (C - Cm), axis=0)
    den = np.sum((D - Dm) * (D - Dm), axis=0) + 1e-9
    a = num / den
    b = Cm - a * Dm
    return a.astype(np.float32), b.astype(np.float32)


def load_affine_rsl_correction_from_h5(
    h5_path: str,
    granularity: str = "per_ecc",
) -> dict:
    """
    Derive correction maps from OSF sinrings.h5.

    h5 contains distorted (40,256,256) and distorted_corrected (40,256,256) in [-1,1].
    Ordering is FREQUENCY-FIRST: idx = freq_idx * 5 + ecc_idx.

    Granularity options:
      global:    Single correction for all 40 stimuli (1 map). Fit over all
                 distorted/corrected pairs. Assumes one correction for everything.
      per_ecc:   Fit per eccentricity over all 8 frequencies (5 maps).
      per_freq:  Fit per frequency over all 5 eccentricities (8 maps).
      per_freq_ecc: One map per (freq, ecc) — 40 maps. Scaling only: a = C/D.
    """
    import h5py

    eccs = ["1", "2.8", "4.7", "6.6", "8.5"]
    ecc_idx = {e: i for i, e in enumerate(eccs)}

    with h5py.File(h5_path, "r") as f:
        dist = f["distorted"][:].astype(np.float32)
        corr = f["distorted_corrected"][:].astype(np.float32)

    if dist.shape != corr.shape or dist.shape[0] != 40:
        raise ValueError(f"Unexpected sinrings.h5 shape: distorted={dist.shape} corrected={corr.shape}")

    correction: dict = {}

    if granularity == "global":
        a, b = _fit_affine_per_pixel(dist, corr)
        correction["__global__"] = (a, b)

    elif granularity == "per_ecc":
        for ecc_s, bi in ecc_idx.items():
            indices = [bi + fi * 5 for fi in range(8)]
            D, C = dist[indices], corr[indices]
            a, b = _fit_affine_per_pixel(D, C)
            correction[ecc_s] = (a, b)

    elif granularity == "per_freq":
        for fi, n in enumerate(PAPER_N):
            indices = [ecc_idx[e] + fi * 5 for e in eccs]
            D, C = dist[indices], corr[indices]
            a, b = _fit_affine_per_pixel(D, C)
            correction[n] = (a, b)

    elif granularity == "per_freq_ecc":
        for fi, n in enumerate(PAPER_N):
            for ecc_s, bi in ecc_idx.items():
                idx = fi * 5 + bi
                D = dist[idx : idx + 1]
                C = corr[idx : idx + 1]
                # 1 sample: scaling only, a = C/D
                eps = 1e-6
                a = np.where(np.abs(D[0]) > eps, C[0] / D[0], np.float32(1.0))
                b = np.zeros_like(a, dtype=np.float32)
                correction[(n, ecc_s)] = (a, b)

    else:
        raise ValueError(f"Unknown granularity: {granularity}")

    return correction


def load_affine_rsl_correction_from_rsl_output(
    h5_path: str,
    rc: "RetinalCompression",
    granularity: str = "per_ecc",
    hires: int = 2048,
    fov: float = 20.0,
    out_size: int = 256,
    mask_norm_dir: Path | None = None,
    mask_calc_dir: Path | None = None,
) -> dict:
    """
    Fit per-pixel affine C = a·D + b where D is OUR generated sinring and C is
    h5 distorted_corrected.

    When mask_calc_dir is provided (preferred), sinrings are generated directly
    in the RSL output space using Mask_Calc masks — the same method as the paper.
    Otherwise falls back to the hires-sinring + RSL approach.

    Granularity options: same as load_affine_rsl_correction_from_h5.
    """
    import h5py

    eccs = ["1", "2.8", "4.7", "6.6", "8.5"]
    ecc_vals = [1.0, 2.8, 4.7, 6.6, 8.5]
    ecc_idx = {e: i for i, e in enumerate(eccs)}

    with h5py.File(h5_path, "r") as f:
        corr = f["distorted_corrected"][:].astype(np.float32)

    if corr.shape[0] != 40:
        raise ValueError(f"Unexpected h5 shape: distorted_corrected={corr.shape}")

    our_dist = np.zeros((40, out_size, out_size), dtype=np.float32)
    use_mask_calc = mask_calc_dir is not None and mask_calc_dir.exists()
    method = "Mask_Calc (direct output-space)" if use_mask_calc else f"hires={hires} + RSL"
    print(f"[correction] Rendering {len(PAPER_N)*len(eccs)} sinrings via {method}...", flush=True)

    for ei, ecc_s in enumerate(eccs):
        e_deg = ecc_vals[ei]
        bi = ecc_idx[ecc_s]
        if use_mask_calc:
            mask = load_mask_calc(mask_calc_dir, rc, ecc_s, out_size, fov)
        elif mask_norm_dir is not None and mask_norm_dir.exists():
            mask = load_mask_norm(mask_norm_dir, ecc_s, hires)
        else:
            mask = make_gaussian_ring_mask(hires, e_deg=e_deg, fov=fov)
        for fi, n in enumerate(PAPER_N):
            h5_idx = fi * 5 + bi
            if use_mask_calc:
                img = make_sinring(out_size, e_deg, n, mask)
                our_dist[h5_idx] = _to_unit_float(img)
            else:
                img_h = make_sinring(hires, e_deg, n, mask)
                out_img, _ = rc.distort_image(
                    image=img_h, fov=fov, out_size=out_size, inv=0, type=1, series=1,
                )
                if out_img.ndim == 3:
                    out_img = out_img[..., 0]
                our_dist[h5_idx] = _to_unit_float(out_img)
        print(f"  ecc={ecc_s} done", flush=True)

    correction: dict = {}

    if granularity == "global":
        a, b = _fit_affine_per_pixel(our_dist, corr)
        correction["__global__"] = (a, b)

    elif granularity == "per_ecc":
        for ecc_s, bi in ecc_idx.items():
            indices = [bi + fi * 5 for fi in range(8)]
            D, C = our_dist[indices], corr[indices]
            a, b = _fit_affine_per_pixel(D, C)
            correction[ecc_s] = (a, b)

    elif granularity == "per_freq":
        for fi, n in enumerate(PAPER_N):
            indices = [ecc_idx[e] + fi * 5 for e in eccs]
            D, C = our_dist[indices], corr[indices]
            a, b = _fit_affine_per_pixel(D, C)
            correction[n] = (a, b)

    elif granularity == "per_freq_ecc":
        for fi, n in enumerate(PAPER_N):
            for ecc_s, bi in ecc_idx.items():
                idx = fi * 5 + bi
                D = our_dist[idx : idx + 1]
                C = corr[idx : idx + 1]
                eps = 1e-6
                a = np.where(np.abs(D[0]) > eps, C[0] / D[0], np.float32(1.0))
                b = np.zeros_like(a, dtype=np.float32)
                correction[(n, ecc_s)] = (a, b)

    else:
        raise ValueError(f"Unknown granularity: {granularity}")

    return correction


def get_correction_for_stimulus(
    correction_maps: dict,
    n: int,
    ecc_s: str,
    granularity: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (a, b) for applying corrected = a*distorted + b to a stimulus (n, ecc_s).
    For extended n (not in PAPER_N), interpolates between nearest paper frequencies.
    """
    if granularity == "global":
        return correction_maps["__global__"]

    if granularity == "per_ecc":
        return correction_maps[ecc_s]

    if granularity == "per_freq":
        if n in PAPER_N:
            return correction_maps[n]
        # Interpolate between nearest paper n
        lo = max((p for p in PAPER_N if p <= n), default=PAPER_N[0])
        hi = min((p for p in PAPER_N if p >= n), default=PAPER_N[-1])
        if lo == hi:
            return correction_maps[lo]
        t = (n - lo) / (hi - lo) if hi != lo else 0.0
        a_lo, b_lo = correction_maps[lo]
        a_hi, b_hi = correction_maps[hi]
        return ((1 - t) * a_lo + t * a_hi, (1 - t) * b_lo + t * b_hi)

    if granularity == "per_freq_ecc":
        if n in PAPER_N and (n, ecc_s) in correction_maps:
            return correction_maps[(n, ecc_s)]
        # Interpolate a,b from nearest paper frequencies
        lo = max((p for p in PAPER_N if p <= n), default=PAPER_N[0])
        hi = min((p for p in PAPER_N if p >= n), default=PAPER_N[-1])
        if lo == hi:
            return correction_maps[(lo, ecc_s)]
        t = (n - lo) / (hi - lo) if hi != lo else 0.0
        a_lo, b_lo = correction_maps[(lo, ecc_s)]
        a_hi, b_hi = correction_maps[(hi, ecc_s)]
        return ((1 - t) * a_lo + t * a_hi, (1 - t) * b_lo + t * b_hi)

    raise ValueError(f"Unknown granularity: {granularity}")


def _imwrite_jpeg(path: Path, img: np.ndarray, quality: int = 95) -> bool:
    """Write image as JPEG with specified quality."""
    return cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])


def extract_paper_stimuli_from_h5(
    h5_path: str,
    out_root: Path,
    ecc_list: list[float],
    overwrite: bool = False,
    do_manifest: bool = False,
    jpeg_quality: int = 95,
) -> int:
    """
    Extract the paper's 8 SFs × 5 ecc directly from sinrings.h5.
    Uses paper naming: 1_4_8.5.jpg, 2_10_8.5.jpg, ..., 8_46_8.5.jpg.
    Returns number of files written.
    """
    import h5py

    eccs_h5 = ["1", "2.8", "4.7", "6.6", "8.5"]
    ecc_idx_map = {float(e): i for i, e in enumerate(eccs_h5)}
    if set(ecc_list) != set(float(e) for e in eccs_h5):
        raise ValueError(
            f"For --source h5, ecc must be exactly {eccs_h5}. Got {ecc_list}"
        )

    with h5py.File(h5_path, "r") as f:
        und = f["undistorted"][:].astype(np.float64)
        dist = f["distorted_corrected"][:].astype(np.float64)

    if und.shape != (40, 256, 256) or dist.shape != (40, 256, 256):
        raise ValueError(f"Unexpected h5 shape: {und.shape} {dist.shape}")

    n_written = 0
    for ecc_deg in ecc_list:
        ecc_s = fmt_ecc(ecc_deg)
        bi = ecc_idx_map[ecc_deg]
        undist_dir = out_root / "undistorted" / f"{ecc_s}_Ecc"
        dist_dir = out_root / "distorted" / f"{ecc_s}_Ecc"
        ensure_dir(undist_dir)
        ensure_dir(dist_dir)

        for fi, n in enumerate(PAPER_N):
            label = 2 * n
            paper_idx = fi + 1
            fname = f"{paper_idx}_{label}_{ecc_s}.jpg"
            h5_idx = fi * 5 + bi

            und_path = undist_dir / fname
            dist_path = dist_dir / fname
            if und_path.exists() and dist_path.exists() and not overwrite:
                n_written += 2
                continue

            und_u8 = _from_unit_float(und[h5_idx])
            dist_u8 = _from_unit_float(dist[h5_idx])
            _imwrite_jpeg(und_path, cv2.cvtColor(und_u8, cv2.COLOR_GRAY2BGR), jpeg_quality)
            _imwrite_jpeg(dist_path, cv2.cvtColor(dist_u8, cv2.COLOR_GRAY2BGR), jpeg_quality)
            n_written += 2
            print(f"[h5] ecc={ecc_s} n={n} label={label} → {fname}")

        if do_manifest:
            manifest_rows = [
                {
                    "filename": f"{fi + 1}_{2 * n}_{ecc_s}.jpg",
                    "e_deg": ecc_deg,
                    "n": n,
                    "label_2n": 2 * n,
                    "cpr": round((n / math.pi) * ecc_deg, 4),
                    "cycles_around_ring": round(2.0 * n * ecc_deg, 2),
                    "source": "h5",
                }
                for fi, n in enumerate(PAPER_N)
            ]
            write_manifest(undist_dir / "manifest.csv", manifest_rows)

    return n_written


# ---------------------------------------------------------------------------
# Mask loading
# ---------------------------------------------------------------------------

def load_mask_norm(mask_dir: Path, ecc_str: str, size: int) -> np.ndarray:
    """Load Mask_Norm/<ecc>.jpg and resize to (size, size).  Returns float32 in [0, 1]."""
    candidates = [mask_dir / f"{ecc_str}.jpg"]
    try:
        e = float(ecc_str)
        if e == int(e):
            candidates.append(mask_dir / f"{int(e)}.jpg")
    except ValueError:
        pass

    img = None
    for p in candidates:
        if p.exists():
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            break
    if img is None:
        raise FileNotFoundError(f"No Mask_Norm found for ecc={ecc_str} in {mask_dir}")

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    m = img.astype(np.float32) / 255.0
    peak = m.max()
    if peak > 0:
        m /= peak
    return m


def load_mask_calc(
    mask_dir: Path,
    rc: "RetinalCompression",
    ecc_str: str,
    size: int,
    fov: float = 20.0,
) -> np.ndarray:
    """Load Mask_Calc/<rout>.jpg — ring mask defined in RSL output space.

    The paper generates distorted sinrings directly in the RSL output coordinate
    space (no geometric warp of a hires sinring).  Mask_Calc files are named by
    the ring's RSL output radius scaled to a 2048-px canvas:
        filename = round(fi(e) / fi(fov/2) × 1024)
    They are 2048×2048 images with a Gaussian ring of FWHM ≈ 71 px (≈ 8.9 px
    at 256×256), which matches the ring width in h5 distorted_corrected.

    Returns float32 in [0, 1].
    """
    fi_e = rc.fi(float(ecc_str))
    fi_max = rc.fi(fov / 2)
    raw = fi_e / fi_max * 1024
    # Try nearby integers to handle floating-point rounding inconsistencies
    candidates = [int(raw), round(int(raw + 0.5)), int(raw) + 1, int(raw) - 1]
    seen = set()
    p = None
    for stem in candidates:
        if stem in seen:
            continue
        seen.add(stem)
        candidate = mask_dir / f"{stem}.jpg"
        if candidate.exists():
            p = candidate
            break
    if p is None:
        raise FileNotFoundError(
            f"No Mask_Calc found for ecc={ecc_str} (tried stems near {raw:.2f}) in {mask_dir}"
        )
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    m = img.astype(np.float32) / 255.0
    peak = m.max()
    if peak > 0:
        m /= peak
    return m


def make_gaussian_ring_mask(
    size: int, *, e_deg: float, fov: float, ring_fwhm_deg: float = 1.0
) -> np.ndarray:
    """Fallback Gaussian ring mask when Mask_Norm images are unavailable."""
    H = W = size
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r0 = (e_deg / (fov / 2.0)) * (size / 2.0)
    sigma_px = max((ring_fwhm_deg / 2.355) / (fov / 2.0) * (size / 2.0), 1e-6)
    m = np.exp(-0.5 * ((r - r0) / sigma_px) ** 2).astype(np.float32)
    peak = m.max()
    if peak > 0:
        m /= peak
    return m


# ---------------------------------------------------------------------------
# Sinring generation
# ---------------------------------------------------------------------------

def make_sinring(
    size: int,
    e_deg: float,
    n: int,
    mask: np.ndarray,
    phase: float = 0.0,
    bg: float = 128.0,
    contrast: float = 127.0,
) -> np.ndarray:
    """
    Render an achromatic sinring stimulus at `size`×`size` pixels.

    Formula: sin(2·n·e·θ)  where θ is the polar angle (arctan2).
    This gives angular SF = n·e cycles/radian, which equals n cycles/degree of
    arc at eccentricity e — constant across eccentricities for fixed n.

    Rendered directly at the target resolution (no hi-res intermediate) to avoid
    axis-aligned box-filter artifacts from downsampling.

    Returns uint8 (H, W) image.
    """
    H = W = size
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    theta = np.arctan2(yy - cy, xx - cx)
    ang = 2.0 * n * e_deg * theta + phase
    stim = bg + contrast * mask * np.sin(ang)
    return np.clip(stim, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_manifest(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate daCosta-style sinring stimuli with configurable SF range."
    )
    ap.add_argument(
        "--sinrings-root", type=str,
        default="/home/tomasdu/repos/datasets/daCosta2024/stimuli/sinrings",
        help="OSF download root (for Mask_Norm images).",
    )
    ap.add_argument(
        "--out-root", type=str,
        default="/home/tomasdu/repos/datasets/daCosta2024-remake_generate7/stimuli/sinrings",
        help="Output root (undistorted/ and distorted/ created here).",
    )
    ap.add_argument(
        "--ecc", type=float, nargs="*",
        default=[1.0, 2.8, 4.7, 6.6, 8.5],
    )

    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--n", type=int, nargs="*", default=None,
                      help="Explicit list of n values.")
    grp.add_argument("--n-min", type=int, default=0)
    ap.add_argument("--n-max", type=int, default=0)
    ap.add_argument("--n-step", type=int, default=1)

    ap.add_argument("--fov", type=float, default=20.0)
    ap.add_argument("--hires", type=int, default=2048,
                     help="Resolution for hi-res undistorted (input to RSL).")
    ap.add_argument("--out-size", type=int, default=256,
                     help="Output resolution for saved images.")
    ap.add_argument("--ring-fwhm-deg", type=float, default=1.0,
                     help="Gaussian ring FWHM (only when Mask_Norm is unavailable).")
    ap.add_argument("--use-mask-norm", action="store_true",
                     help="Use Mask_Norm/<ecc>.jpg from OSF (recommended).")
    ap.add_argument("--phase", type=float, default=0.0)
    ap.add_argument("--jpeg-quality", type=int, default=95,
                     help="JPEG quality 1-100 (lower = smaller files).")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument(
        "--rsl-correction",
        type=str,
        default="affine_from_h5",
        choices=["none", "affine_from_h5", "affine_from_own_rsl"],
        help=(
            "Post-correct RSL-distorted images:\n"
            "  none: no correction;\n"
            "  affine_from_h5: affine fit from h5 distorted → h5 distorted_corrected "
            "(assumes our RSL ≈ paper's RSL);\n"
            "  affine_from_own_rsl: affine fit from OUR RSL output → h5 distorted_corrected "
            "(recommended when hires or mask differs from paper's pipeline)."
        ),
    )
    ap.add_argument(
        "--correction-granularity",
        type=str,
        default="per_ecc",
        choices=["global", "per_ecc", "per_freq", "per_freq_ecc"],
        help="How to fit correction: global (1 map, all 40); per_ecc (5 maps); "
        "per_freq (8 maps); per_freq_ecc (40 maps, scaling only).",
    )
    ap.add_argument(
        "--sinrings-h5",
        type=str,
        default="/home/tomasdu/repos/datasets/daCosta2024/stimuli/sinrings/sinrings.h5",
        help="Path to sinrings.h5 for correction maps (when --rsl-correction=affine_from_h5).",
    )
    ap.add_argument(
        "--source",
        type=str,
        default="generate",
        choices=["generate", "h5", "hybrid"],
        help="'generate': all via RSL. 'h5': paper's 8 only from sinrings.h5 (exact match, paper naming). "
        "'hybrid': paper's 8 from h5 (1_4..8_46) + extended SFs via RSL (9+, 10+, ...).",
    )
    ap.add_argument(
        "--paper-plus-extended",
        action="store_true",
        help="[hybrid only] In paper range [2,23], include ONLY paper's 8 (from h5). "
        "Extended SFs only outside that range (n<2 or n>23). Avoids RSL-generated stimuli in paper range.",
    )

    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    sinrings_root = Path(args.sinrings_root)
    out_root = Path(args.out_root)
    mask_norm_dir = sinrings_root / "Mask_Norm"
    mask_calc_dir = sinrings_root / "Mask_Calc"

    ecc_list = [float(e) for e in args.ecc]

    if args.n is not None:
        n_list = [int(x) for x in args.n]
    elif args.n_min > 0 and args.n_max > 0:
        n_list = _n_range(args.n_min, args.n_max, args.n_step)
    else:
        n_list = [2, 5, 8, 11, 14, 17, 20, 23]

    # Hybrid + paper-plus-extended: in paper range [2,23] only paper's 8; extended only outside
    if args.source == "hybrid" and args.paper_plus_extended:
        paper_min, paper_max = min(PAPER_N), max(PAPER_N)
        extended_outside = [n for n in n_list if n not in PAPER_N and (n < paper_min or n > paper_max)]
        n_list = sorted(set(PAPER_N) | set(extended_outside))
        if extended_outside:
            print(f"[paper-plus-extended] Paper's 8 in [2,23]; extended only: n<2 or n>{paper_max}")

    if args.source == "h5":
        if n_list != PAPER_N:
            raise SystemExit(
                f"--source h5 requires exactly the paper's 8 frequencies (n={PAPER_N}). "
                f"Use --n 2 5 8 11 14 17 20 23 or omit --n/--n-min/--n-max."
            )
        n_written = extract_paper_stimuli_from_h5(
            args.sinrings_h5, out_root, ecc_list,
            overwrite=args.overwrite, do_manifest=args.write_manifest,
            jpeg_quality=args.jpeg_quality,
        )
        print(f"\nDone. Extracted {n_written} images from h5 to {out_root}")
        return

    # For hybrid: load h5 so we can extract paper's 8 with paper naming
    h5_data = None
    ecc_idx_map = None
    if args.source == "hybrid" and set(PAPER_N).issubset(set(n_list)):
        import h5py
        with h5py.File(args.sinrings_h5, "r") as f:
            h5_data = {
                "undistorted": f["undistorted"][:].astype(np.float64),
                "distorted_corrected": f["distorted_corrected"][:].astype(np.float64),
            }
        ecc_idx_map = {float(e): i for i, e in enumerate(["1", "2.8", "4.7", "6.6", "8.5"])}
        extended = sorted(set(n_list) - set(PAPER_N))
        print(f"[hybrid] Paper's 8 (n={PAPER_N}) from h5 → 1_4..8_46; extended (n={extended}) via RSL → 9+")

    rc = RetinalCompression()
    correction_maps = None
    correction_granularity = "-"
    if args.rsl_correction == "affine_from_h5":
        correction_granularity = args.correction_granularity
        correction_maps = load_affine_rsl_correction_from_h5(
            args.sinrings_h5, granularity=correction_granularity
        )
    elif args.rsl_correction == "affine_from_own_rsl":
        correction_granularity = args.correction_granularity
        correction_maps = load_affine_rsl_correction_from_rsl_output(
            args.sinrings_h5,
            rc=rc,
            granularity=correction_granularity,
            hires=args.hires,
            fov=args.fov,
            out_size=args.out_size,
            mask_norm_dir=mask_norm_dir if args.use_mask_norm else None,
            mask_calc_dir=mask_calc_dir if mask_calc_dir.exists() else None,
        )

    total = len(ecc_list) * len(n_list)

    done = 0

    for e_deg in ecc_list:
        ecc_s = fmt_ecc(e_deg)
        undist_dir = out_root / "undistorted" / f"{ecc_s}_Ecc"
        dist_dir = out_root / "distorted" / f"{ecc_s}_Ecc"
        ensure_dir(undist_dir)
        ensure_dir(dist_dir)

        # mask_out: used for undistorted images (rendered directly at out_size).
        # mask_hires: fallback for distorted images when Mask_Calc is unavailable
        # (hires sinring → RSL warp; gives very thin rings, not recommended).
        if args.use_mask_norm and mask_norm_dir.exists():
            mask_out = load_mask_norm(mask_norm_dir, ecc_s, args.out_size)
            mask_hires = load_mask_norm(mask_norm_dir, ecc_s, args.hires)
        else:
            mask_out = make_gaussian_ring_mask(
                args.out_size, e_deg=e_deg, fov=args.fov, ring_fwhm_deg=args.ring_fwhm_deg
            )
            mask_hires = make_gaussian_ring_mask(
                args.hires, e_deg=e_deg, fov=args.fov, ring_fwhm_deg=args.ring_fwhm_deg
            )

        # Load the ring mask for the distorted case.
        # Mask_Calc contains rings defined directly in RSL output space (at the
        # cortical radius fi(e)/fi(max)×128 px) with FWHM ≈ 8.9 px at 256×256,
        # matching the paper's h5 distorted_corrected.  Using this mask we
        # generate the distorted sinring directly at out_size — no RSL warp needed.
        mask_calc = None
        if mask_calc_dir.exists():
            try:
                mask_calc = load_mask_calc(
                    mask_calc_dir, rc, ecc_s, args.out_size, args.fov
                )
            except FileNotFoundError:
                mask_calc = None

        manifest_rows: list = []
        extended_list = sorted(set(n_list) - set(PAPER_N)) if args.source == "hybrid" else []

        for n in n_list:
            spec = StimSpec(e_deg=e_deg, n=n)
            label = spec.label
            # Paper naming (1_4..8_46) for paper's 8 in hybrid; else sequential idx
            if args.source == "hybrid" and n in PAPER_N:
                idx_in_ecc = PAPER_N.index(n) + 1
            elif args.source == "hybrid" and extended_list:
                idx_in_ecc = 9 + extended_list.index(n)
            else:
                idx_in_ecc = n_list.index(n) + 1
            fname = f"{idx_in_ecc}_{label}_{ecc_s}.jpg"
            undist_path = undist_dir / fname
            dist_path = dist_dir / fname

            if undist_path.exists() and dist_path.exists() and not args.overwrite:
                done += 1
                continue

            # Hybrid: paper's 8 from h5 (exact match, paper naming)
            if h5_data is not None and n in PAPER_N and ecc_idx_map is not None and e_deg in ecc_idx_map:
                freq_idx = PAPER_N.index(n)
                ecc_idx = ecc_idx_map[e_deg]
                h5_idx = freq_idx * 5 + ecc_idx
                undist_out = _from_unit_float(h5_data["undistorted"][h5_idx])
                distorted_out = _from_unit_float(h5_data["distorted_corrected"][h5_idx])
                if undist_out.ndim == 2:
                    undist_out = cv2.cvtColor(undist_out, cv2.COLOR_GRAY2BGR)
                    distorted_out = cv2.cvtColor(distorted_out, cv2.COLOR_GRAY2BGR)
                _imwrite_jpeg(undist_path, undist_out, args.jpeg_quality)
                _imwrite_jpeg(dist_path, distorted_out, args.jpeg_quality)
                done += 1
                print(f"[{done}/{total}] ecc={ecc_s}  n={n}  label={label}  [h5] → {fname}")
                manifest_rows.append({
                    "filename": fname,
                    "e_deg": e_deg,
                    "n": n,
                    "label_2n": label,
                    "cpr": round(spec.cpr, 4),
                    "cycles_around_ring": round(spec.cycles_around_ring, 2),
                    "rsl_correction": "h5",
                    "source": "h5",
                })
                continue

            # Undistorted: render directly at out_size. The former hi-res→INTER_AREA
            # pipeline applied an axis-aligned box filter that attenuated diagonal
            # ring sections differently from cardinal ones, producing the "twist"
            # artifact. Direct rendering gives uniform per-pixel sampling everywhere.
            undist_out = make_sinring(
                args.out_size, e_deg, n, mask_out, phase=args.phase,
            )

            # Distorted: generate sinring directly in RSL output space.
            # Mask_Calc places the ring at the cortical radius (fi(e)/fi(max)×128)
            # so the sinusoidal content lands at the correct distorted location and
            # spatial frequency without any geometric warp of the image.
            # Fallback: hires + RSL (gives very thin rings due to extreme local
            # compression, not matching the paper's stimuli).
            if mask_calc is not None:
                distorted_raw = make_sinring(
                    args.out_size, e_deg, n, mask_calc, phase=args.phase,
                )
            else:
                undist_hires = make_sinring(
                    args.hires, e_deg, n, mask_hires, phase=args.phase,
                )
                distorted_raw, _ = rc.distort_image(
                    image=undist_hires,
                    fov=float(args.fov),
                    out_size=args.out_size,
                    inv=0, type=1, series=1,
                )
                if distorted_raw.ndim == 3:
                    distorted_raw = distorted_raw[..., 0]

            if correction_maps is not None:
                a, b = get_correction_for_stimulus(
                    correction_maps, n, ecc_s, correction_granularity
                )
                d_unit = _to_unit_float(distorted_raw)
                d_corr = np.clip(a * d_unit + b, -1.0, 1.0)
                distorted_out = _from_unit_float(d_corr)
            else:
                distorted_out = distorted_raw

            _imwrite_jpeg(undist_path, undist_out, args.jpeg_quality)
            _imwrite_jpeg(dist_path, distorted_out, args.jpeg_quality)

            done += 1
            print(f"[{done}/{total}] ecc={ecc_s}  n={n}  label={label}  → {fname}")

            manifest_rows.append({
                "filename": fname,
                "e_deg": e_deg,
                "n": n,
                "label_2n": label,
                "cpr": round(spec.cpr, 4),
                "cycles_around_ring": round(spec.cycles_around_ring, 2),
                "rsl_correction": args.rsl_correction,
                "correction_granularity": correction_granularity,
                "source": "rsl",
            })

        if args.write_manifest and manifest_rows:
            manifest_rows.sort(key=lambda r: (r["e_deg"], r["label_2n"]))
            write_manifest(undist_dir / "manifest.csv", manifest_rows)

    print(f"\nDone. {done} stimuli written to {out_root}")


if __name__ == "__main__":
    main()
