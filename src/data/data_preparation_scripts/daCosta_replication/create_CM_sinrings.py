#!/usr/bin/env python3
"""
Generate daCosta-style sinring stimuli with a configurable cortical magnification (CM) transform.

Supported transforms (--transform):

  rsl      — Watson (2014) ganglion cell density RSL (RetinalCompression).
             The distorted sinring is generated directly in RSL output space
             using Mask_Calc ring masks — no geometric image warp is performed.
             An affine pixel-wise correction (corrected = a·D + b) is learned
             from the paper's sinrings.h5 to match the OSF reference stimuli.

  fisheye  — Piecewise fisheye cortical magnification (FisheyeTransform).
             The undistorted sinring (rendered at out_size using Mask_Norm) is
             warped through FisheyeTransform with crop_to_valid=True so the
             saved distorted image matches the image size the network sees during
             training (scotoma_dataset.py applies the same crop).
             For C=1, K=-7, rfov=30 on a 256×256 input the cropped output is
             156×156.  No pre-computed Mask_Calc files are needed; ring position
             and width in the output follow directly from the warp.

Sinring formula (verified against OSF sinrings.h5):

    pixel_value = bg + contrast · mask · sin(2·n·e·θ)

where:
    n        — frequency index (paper uses n ∈ {2,5,8,11,14,17,20,23})
    e        — eccentricity in degrees
    label    — 2·n  (used in filenames)
    θ        — polar angle at each pixel (arctan2)
    mask     — Mask_Norm/<ecc>.jpg from the OSF dataset (ring envelope)
    bg       — 128  (gray background, uint8)
    contrast — 127

The factor e ensures equal spatial frequency in cycles/degree at all
eccentricities: angular SF = n·e cycles/radian, so cycles/degree of arc = n.

Output directory layout:

    <out-root>/
      undistorted/<ecc>_Ecc/{idx}_{label}_{ecc}.jpg   (always out_size × out_size)
      distorted/<ecc>_Ecc/{idx}_{label}_{ecc}.jpg      (RSL: out_size; fisheye: crop size)

For --source hybrid (RSL only) the paper's 8 SFs are taken verbatim from
sinrings.h5 (filenames 1_4..8_46) and extended SFs are generated via RSL.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Paper's 8 spatial frequencies (n values) in sinrings.h5
PAPER_N = [2, 5, 8, 11, 14, 17, 20, 23]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StimSpec:
    e_deg: float
    n: int

    @property
    def label(self) -> int:
        """Filename label = 2·n."""
        return 2 * self.n

    @property
    def cpr(self) -> float:
        """Cycles per degree of arc = n (constant across eccentricities)."""
        return float(self.n)

    @property
    def cycles_around_ring(self) -> float:
        """Total sinusoidal cycles in 2π radians = 2·n·e."""
        return 2.0 * self.n * self.e_deg


def fmt_ecc(e_deg: float) -> str:
    """'1' not '1.0', '2.8' stays '2.8'."""
    return f"{float(e_deg):g}"


def _n_range(n_min: int, n_max: int, n_step: int) -> list:
    if n_step <= 0:
        raise ValueError("step must be > 0")
    return list(range(n_min, n_max + 1, n_step))


def _to_unit_float(u8: np.ndarray) -> np.ndarray:
    """Map uint8 [0..255] (bg=128) to float32 [-1..1] (bg=0)."""
    return (u8.astype(np.float32) - 128.0) / 127.0


def _from_unit_float(x: np.ndarray) -> np.ndarray:
    """Map float32 [-1..1] (bg=0) back to uint8 [0..255] (bg=128)."""
    return np.clip(x * 127.0 + 128.0, 0, 255).astype(np.uint8)


def _fit_affine_per_pixel(D: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-pixel affine C = a·D + b via least squares. D, C shape (N, H, W)."""
    Dm = D.mean(axis=0)
    Cm = C.mean(axis=0)
    num = np.sum((D - Dm) * (C - Cm), axis=0)
    den = np.sum((D - Dm) ** 2, axis=0) + 1e-9
    a = (num / den).astype(np.float32)
    b = (Cm - a * Dm).astype(np.float32)
    return a, b


def _imwrite_jpeg(path: Path, img: np.ndarray, quality: int = 95) -> bool:
    return cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])


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
# Sinring generation (shared)
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
    Render an achromatic sinring stimulus at ``size``×``size`` pixels.

    Formula: pixel = bg + contrast · mask · sin(2·n·e·θ)
    Angular SF = n·e cycles/radian ≡ n cycles/degree of arc at eccentricity e.

    Rendered directly at the target resolution to avoid axis-aligned box-filter
    artifacts from downsampling a hi-res intermediate.

    Returns uint8 (H, W).
    """
    cx, cy = (size - 1) / 2.0, (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    theta = np.arctan2(yy - cy, xx - cx)
    ang = 2.0 * n * e_deg * theta + phase
    stim = bg + contrast * mask * np.sin(ang)
    return np.clip(stim, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Mask loading (shared — used for undistorted images)
# ---------------------------------------------------------------------------

def load_mask_norm(mask_dir: Path, ecc_str: str, size: int) -> np.ndarray:
    """Load Mask_Norm/<ecc>.jpg and resize to (size, size). Returns float32 in [0, 1].

    Mask_Norm files are defined in undistorted (input) coordinate space, with the
    ring centered at radius r = (e / (fov/2)) × (size/2) pixels.  Used for
    undistorted sinrings and as the input mask for the fisheye warp.
    """
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


def make_gaussian_ring_mask(
    size: int, *, e_deg: float, fov: float, ring_fwhm_deg: float = 1.0
) -> np.ndarray:
    """Fallback Gaussian ring mask when Mask_Norm images are unavailable."""
    cx, cy = (size - 1) / 2.0, (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r0 = (e_deg / (fov / 2.0)) * (size / 2.0)
    sigma_px = max((ring_fwhm_deg / 2.355) / (fov / 2.0) * (size / 2.0), 1e-6)
    m = np.exp(-0.5 * ((r - r0) / sigma_px) ** 2).astype(np.float32)
    peak = m.max()
    if peak > 0:
        m /= peak
    return m


# ---------------------------------------------------------------------------
# RSL-specific: Mask_Calc loading
# ---------------------------------------------------------------------------

def load_mask_calc(
    mask_dir: Path,
    rc: "RetinalCompression",
    ecc_str: str,
    size: int,
    fov: float = 20.0,
) -> np.ndarray:
    """Load Mask_Calc/<rout>.jpg — ring mask defined in RSL output space.

    The paper generates distorted sinrings directly in RSL output coordinates
    (no geometric warp of a hi-res sinring).  Mask_Calc files are named by the
    ring's RSL output radius scaled to a 2048-px canvas:
        filename ≈ int(fi(e) / fi(fov/2) × 1024)

    All Mask_Calc files are 2048×2048 with a Gaussian ring of FWHM ≈ 71 px
    (≈ 8.9 px at 256×256), matching the ring width in h5 distorted_corrected.

    Returns float32 in [0, 1].
    """
    fi_e = rc.fi(float(ecc_str))
    fi_max = rc.fi(fov / 2)
    raw = fi_e / fi_max * 1024
    # Try nearby integers to handle floating-point rounding inconsistencies
    seen: set = set()
    p = None
    for stem in [int(raw), round(int(raw + 0.5)), int(raw) + 1, int(raw) - 1]:
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


# ---------------------------------------------------------------------------
# RSL-specific: affine correction
# ---------------------------------------------------------------------------

def load_affine_rsl_correction_from_h5(
    h5_path: str,
    granularity: str = "per_ecc",
) -> dict:
    """Fit per-pixel affine C = a·D + b from h5 distorted → distorted_corrected.

    Granularity: global (1 map) | per_ecc (5) | per_freq (8) | per_freq_ecc (40).
    """
    import h5py

    eccs = ["1", "2.8", "4.7", "6.6", "8.5"]
    ecc_idx = {e: i for i, e in enumerate(eccs)}

    with h5py.File(h5_path, "r") as f:
        dist = f["distorted"][:].astype(np.float32)
        corr = f["distorted_corrected"][:].astype(np.float32)

    if dist.shape != corr.shape or dist.shape[0] != 40:
        raise ValueError(f"Unexpected sinrings.h5 shape: {dist.shape}")

    correction: dict = {}
    if granularity == "global":
        a, b = _fit_affine_per_pixel(dist, corr)
        correction["__global__"] = (a, b)
    elif granularity == "per_ecc":
        for ecc_s, bi in ecc_idx.items():
            indices = [bi + fi * 5 for fi in range(8)]
            a, b = _fit_affine_per_pixel(dist[indices], corr[indices])
            correction[ecc_s] = (a, b)
    elif granularity == "per_freq":
        for fi, n in enumerate(PAPER_N):
            indices = [ecc_idx[e] + fi * 5 for e in eccs]
            a, b = _fit_affine_per_pixel(dist[indices], corr[indices])
            correction[n] = (a, b)
    elif granularity == "per_freq_ecc":
        for fi, n in enumerate(PAPER_N):
            for ecc_s, bi in ecc_idx.items():
                idx = fi * 5 + bi
                D, C_map = dist[idx:idx + 1], corr[idx:idx + 1]
                eps = 1e-6
                a = np.where(np.abs(D[0]) > eps, C_map[0] / D[0], np.float32(1.0))
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
    out_size: int = 224,
    mask_norm_dir: Path | None = None,
    mask_calc_dir: Path | None = None,
) -> dict:
    """Fit per-pixel affine C = a·D + b where D is our generated sinring and C is
    h5 distorted_corrected.  Uses Mask_Calc (preferred) or falls back to hires+RSL.
    """
    import h5py

    eccs = ["1", "2.8", "4.7", "6.6", "8.5"]
    ecc_vals = [1.0, 2.8, 4.7, 6.6, 8.5]
    ecc_idx = {e: i for i, e in enumerate(eccs)}

    with h5py.File(h5_path, "r") as f:
        corr = f["distorted_corrected"][:].astype(np.float32)

    our_dist = np.zeros((40, out_size, out_size), dtype=np.float32)
    use_mask_calc = mask_calc_dir is not None and mask_calc_dir.exists()
    method = "Mask_Calc (direct output-space)" if use_mask_calc else f"hires={hires} + RSL"
    print(f"[correction] Rendering {len(PAPER_N) * len(eccs)} sinrings via {method}...", flush=True)

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
            a, b = _fit_affine_per_pixel(our_dist[indices], corr[indices])
            correction[ecc_s] = (a, b)
    elif granularity == "per_freq":
        for fi, n in enumerate(PAPER_N):
            indices = [ecc_idx[e] + fi * 5 for e in eccs]
            a, b = _fit_affine_per_pixel(our_dist[indices], corr[indices])
            correction[n] = (a, b)
    elif granularity == "per_freq_ecc":
        for fi, n in enumerate(PAPER_N):
            for ecc_s, bi in ecc_idx.items():
                idx = fi * 5 + bi
                D, C_map = our_dist[idx:idx + 1], corr[idx:idx + 1]
                eps = 1e-6
                a = np.where(np.abs(D[0]) > eps, C_map[0] / D[0], np.float32(1.0))
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
    """Return (a, b) for corrected = a·distorted + b.
    For extended n not in PAPER_N, linearly interpolates between nearest paper SFs.
    """
    if granularity == "global":
        return correction_maps["__global__"]

    if granularity == "per_ecc":
        return correction_maps[ecc_s]

    if granularity == "per_freq":
        if n in PAPER_N:
            return correction_maps[n]
        lo = max((p for p in PAPER_N if p <= n), default=PAPER_N[0])
        hi = min((p for p in PAPER_N if p >= n), default=PAPER_N[-1])
        if lo == hi:
            return correction_maps[lo]
        t = (n - lo) / (hi - lo)
        a_lo, b_lo = correction_maps[lo]
        a_hi, b_hi = correction_maps[hi]
        return ((1 - t) * a_lo + t * a_hi, (1 - t) * b_lo + t * b_hi)

    if granularity == "per_freq_ecc":
        if n in PAPER_N and (n, ecc_s) in correction_maps:
            return correction_maps[(n, ecc_s)]
        lo = max((p for p in PAPER_N if p <= n), default=PAPER_N[0])
        hi = min((p for p in PAPER_N if p >= n), default=PAPER_N[-1])
        if lo == hi:
            return correction_maps[(lo, ecc_s)]
        t = (n - lo) / (hi - lo)
        a_lo, b_lo = correction_maps[(lo, ecc_s)]
        a_hi, b_hi = correction_maps[(hi, ecc_s)]
        return ((1 - t) * a_lo + t * a_hi, (1 - t) * b_lo + t * b_hi)

    raise ValueError(f"Unknown granularity: {granularity}")


# ---------------------------------------------------------------------------
# RSL-specific: extract paper stimuli from h5
# ---------------------------------------------------------------------------

def extract_paper_stimuli_from_h5(
    h5_path: str,
    out_root: Path,
    ecc_list: list[float],
    overwrite: bool = False,
    do_manifest: bool = False,
    jpeg_quality: int = 95,
) -> int:
    """Extract the paper's 8 SFs × 5 eccentricities directly from sinrings.h5."""
    import h5py

    eccs_h5 = ["1", "2.8", "4.7", "6.6", "8.5"]
    ecc_idx_map = {float(e): i for i, e in enumerate(eccs_h5)}
    if set(ecc_list) != set(float(e) for e in eccs_h5):
        raise ValueError(f"--source h5 requires ecc={eccs_h5}. Got {ecc_list}")

    with h5py.File(h5_path, "r") as f:
        und = f["undistorted"][:].astype(np.float64)
        dist = f["distorted_corrected"][:].astype(np.float64)

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
            fname = f"{fi + 1}_{label}_{ecc_s}.jpg"
            h5_idx = fi * 5 + bi
            und_path, dist_path = undist_dir / fname, dist_dir / fname

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
            rows = [
                {"filename": f"{fi + 1}_{2 * n}_{ecc_s}.jpg", "e_deg": ecc_deg,
                 "n": n, "label_2n": 2 * n, "source": "h5"}
                for fi, n in enumerate(PAPER_N)
            ]
            write_manifest(undist_dir / "manifest.csv", rows)

    return n_written


# ---------------------------------------------------------------------------
# Fisheye-specific: apply warp to a sinring image
# ---------------------------------------------------------------------------

def apply_fisheye_to_sinring(img_u8: np.ndarray, fisheye_transform) -> np.ndarray:
    """Apply FisheyeTransform to a uint8 grayscale sinring image.

    The sinring is converted to float32 centered at 0 (bg=128 → 0.0) before
    warping so that out-of-bounds pixels (zero-padded by grid_sample) map to
    the gray background (128) after converting back.

    With crop_to_valid=True the returned array will have the cropped size
    (e.g. 156×156 for C=1, K=-7, rfov=30 applied to a 256×256 input).

    Returns uint8 (H', W').
    """
    import torch

    f = _to_unit_float(img_u8)          # [H, W] float32, bg=0
    t = torch.from_numpy(f)             # pass 2D — FisheyeTransform adds/removes batch+channel
    with torch.no_grad():
        out = fisheye_transform(t)      # [H', W'] after crop_to_valid squeeze
    return _from_unit_float(out.cpu().numpy())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate daCosta-style sinring stimuli with a configurable CM transform.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Transform selection ------------------------------------------------
    ap.add_argument(
        "--transform", type=str, default="rsl",
        choices=["rsl", "fisheye"],
        help=(
            "Cortical magnification transform for distorted images.\n"
            "  rsl     : Watson (2014) RetinalCompression (uses Mask_Calc + affine correction)\n"
            "  fisheye : Piecewise fisheye warp with crop_to_valid=True"
        ),
    )

    # --- Shared arguments ---------------------------------------------------
    ap.add_argument(
        "--sinrings-root", type=str,
        default="/home/tomasdu/repos/datasets/daCosta2024/stimuli/sinrings",
        help="OSF download root containing Mask_Norm/, Mask_Calc/, sinrings.h5.",
    )
    ap.add_argument( "--out-root", type=str,
        default="/home/tomasdu/repos/datasets/fisheye_sinrings_rfov30_more_granular_224/stimuli/sinrings",
        help="Output root (undistorted/ and distorted/ created here).",
    )
    ap.add_argument("--ecc", type=float, nargs="*", default=[1.0, 2.8, 4.7, 6.6, 8.5])
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--n", type=int, nargs="*", default=None, help="Explicit list of n values.")
    grp.add_argument("--n-min", type=int, default=0)
    ap.add_argument("--n-max", type=int, default=0)
    ap.add_argument("--n-step", type=int, default=1)
    ap.add_argument("--fov", type=float, default=20.0)
    ap.add_argument(
        "--out-size", type=int, default=224,
        help=(
            "Resolution of the undistorted images and the input to the transform. "
            "RSL distorted images are also out_size×out_size. "
            "Fisheye distorted images are cropped to the valid region (e.g. 156×156 "
            "for the default parameters on a 256×256 input)."
        ),
    )
    ap.add_argument(
        "--use-mask-norm", action="store_true",
        help="Use Mask_Norm/<ecc>.jpg from the OSF dataset for the ring envelope.",
    )
    ap.add_argument("--ring-fwhm-deg", type=float, default=1.0,
                    help="Gaussian ring FWHM in degrees (fallback when Mask_Norm unavailable).")
    ap.add_argument("--phase", type=float, default=0.0)
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--write-manifest", action="store_true")

    # --- Fisheye-specific ---------------------------------------------------
    fe = ap.add_argument_group("fisheye transform options (--transform fisheye)")
    fe.add_argument(
        "--fisheye-C", type=float, default=1.0,
        help="Foveal magnification factor. C=1 → center is unmagnified.",
    )
    fe.add_argument(
        "--fisheye-K", type=float, default=-7.0,
        help="Transition smoothness. Negative values compress the periphery.",
    )
    fe.add_argument(
        "--fisheye-rfov", type=float, default=30.0,
        help="Foveal radius in pixels: inner circle (r < rfov) is linear (e = r/C).",
    )

    # --- RSL-specific -------------------------------------------------------
    rsl = ap.add_argument_group("RSL transform options (--transform rsl)")
    rsl.add_argument("--hires", type=int, default=2048,
                     help="Hi-res canvas used only in the RSL hires+warp fallback.")
    rsl.add_argument(
        "--rsl-correction", type=str, default="affine_from_own_rsl",
        choices=["none", "affine_from_h5", "affine_from_own_rsl"],
        help=(
            "Post-correction for RSL distorted images:\n"
            "  none               : no correction\n"
            "  affine_from_h5     : fit from h5 distorted → h5 distorted_corrected\n"
            "  affine_from_own_rsl: fit from our Mask_Calc output → h5 distorted_corrected "
            "(recommended)"
        ),
    )
    rsl.add_argument(
        "--correction-granularity", type=str, default="per_ecc",
        choices=["global", "per_ecc", "per_freq", "per_freq_ecc"],
        help="Granularity of the affine correction: global (1 map) | per_ecc (5) | "
             "per_freq (8) | per_freq_ecc (40, scaling only).",
    )
    rsl.add_argument(
        "--sinrings-h5", type=str,
        default="/home/tomasdu/repos/datasets/daCosta2024/stimuli/sinrings/sinrings.h5",
    )
    rsl.add_argument(
        "--source", type=str, default="generate",
        choices=["generate", "h5", "hybrid"],
        help=(
            "generate : all stimuli via RSL pipeline\n"
            "h5       : paper's 8 SFs extracted verbatim from sinrings.h5\n"
            "hybrid   : paper's 8 (1_4..8_46) from h5 + extended SFs via RSL"
        ),
    )
    rsl.add_argument(
        "--paper-plus-extended", action="store_true",
        help="[hybrid only] In the paper range [2,23] only use paper's 8; "
             "extended SFs only outside that range.",
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
    mask_calc_dir = sinrings_root / "Mask_Calc"   # RSL only

    ecc_list = [float(e) for e in args.ecc]

    if args.n is not None:
        n_list = [int(x) for x in args.n]
    elif args.n_min > 0 and args.n_max > 0:
        n_list = _n_range(args.n_min, args.n_max, args.n_step)
    else:
        n_list = list(PAPER_N)

    # ------------------------------------------------------------------
    # Transform initialisation
    # ------------------------------------------------------------------
    rc = None
    fisheye_transform = None

    if args.transform == "rsl":
        from src.data.transforms.RSL import RetinalCompression
        rc = RetinalCompression()

    elif args.transform == "fisheye":
        from src.data.transforms.fisheye import FisheyeTransform
        fisheye_transform = FisheyeTransform(
            C=args.fisheye_C,
            K=args.fisheye_K,
            rfov=args.fisheye_rfov,
            crop_to_valid=True,
        )
        if args.source != "generate":
            raise SystemExit(
                f"--source {args.source!r} is only supported with --transform rsl. "
                "Use --source generate for fisheye."
            )

    # ------------------------------------------------------------------
    # RSL-only: hybrid / h5 source mode
    # ------------------------------------------------------------------
    if args.transform == "rsl":
        if args.source == "hybrid" and args.paper_plus_extended:
            paper_min, paper_max = min(PAPER_N), max(PAPER_N)
            extended_outside = [
                n for n in n_list if n not in PAPER_N and (n < paper_min or n > paper_max)
            ]
            n_list = sorted(set(PAPER_N) | set(extended_outside))

        if args.source == "h5":
            if n_list != PAPER_N:
                raise SystemExit(f"--source h5 requires n={PAPER_N}.")
            n_written = extract_paper_stimuli_from_h5(
                args.sinrings_h5, out_root, ecc_list,
                overwrite=args.overwrite, do_manifest=args.write_manifest,
                jpeg_quality=args.jpeg_quality,
            )
            print(f"\nDone. Extracted {n_written} images from h5 to {out_root}")
            return

    # ------------------------------------------------------------------
    # RSL-only: affine correction maps
    # ------------------------------------------------------------------
    correction_maps = None
    correction_granularity = "-"

    if args.transform == "rsl":
        if args.rsl_correction == "affine_from_h5":
            correction_granularity = args.correction_granularity
            correction_maps = load_affine_rsl_correction_from_h5(
                args.sinrings_h5, granularity=correction_granularity,
            )
        elif args.rsl_correction == "affine_from_own_rsl":
            correction_granularity = args.correction_granularity
            correction_maps = load_affine_rsl_correction_from_rsl_output(
                args.sinrings_h5, rc=rc,
                granularity=correction_granularity,
                hires=args.hires,
                fov=args.fov,
                out_size=args.out_size,
                mask_norm_dir=mask_norm_dir if args.use_mask_norm else None,
                mask_calc_dir=mask_calc_dir if mask_calc_dir.exists() else None,
            )

    # ------------------------------------------------------------------
    # RSL-only: pre-load h5 data for hybrid mode
    # ------------------------------------------------------------------
    h5_data = None
    ecc_idx_map_h5 = None

    if args.transform == "rsl" and args.source == "hybrid" and set(PAPER_N).issubset(set(n_list)):
        import h5py
        with h5py.File(args.sinrings_h5, "r") as f:
            h5_data = {
                "undistorted": f["undistorted"][:].astype(np.float64),
                "distorted_corrected": f["distorted_corrected"][:].astype(np.float64),
            }
        ecc_idx_map_h5 = {float(e): i for i, e in enumerate(["1", "2.8", "4.7", "6.6", "8.5"])}
        extended = sorted(set(n_list) - set(PAPER_N))
        print(f"[hybrid] Paper's 8 from h5; extended (n={extended}) via RSL")

    # ------------------------------------------------------------------
    # Main generation loop
    # ------------------------------------------------------------------
    total = len(ecc_list) * len(n_list)
    done = 0

    for e_deg in ecc_list:
        ecc_s = fmt_ecc(e_deg)
        undist_dir = out_root / "undistorted" / f"{ecc_s}_Ecc"
        dist_dir = out_root / "distorted" / f"{ecc_s}_Ecc"
        ensure_dir(undist_dir)
        ensure_dir(dist_dir)

        # ---- Ring mask for undistorted images (shared) ------------------
        # Mask_Norm is in undistorted (input) coordinate space.
        # For fisheye, this mask is also used as the input mask for the warp.
        if args.use_mask_norm and mask_norm_dir.exists():
            mask_out = load_mask_norm(mask_norm_dir, ecc_s, args.out_size)
            mask_hires = (
                load_mask_norm(mask_norm_dir, ecc_s, args.hires)
                if args.transform == "rsl" else None
            )
        else:
            mask_out = make_gaussian_ring_mask(
                args.out_size, e_deg=e_deg, fov=args.fov, ring_fwhm_deg=args.ring_fwhm_deg,
            )
            mask_hires = (
                make_gaussian_ring_mask(
                    args.hires, e_deg=e_deg, fov=args.fov, ring_fwhm_deg=args.ring_fwhm_deg,
                )
                if args.transform == "rsl" else None
            )

        # ---- RSL: Mask_Calc for distorted images ------------------------
        # Mask_Calc rings are defined in RSL output coordinate space; using them
        # lets us generate the distorted sinring directly at out_size without any
        # geometric warp.  Falls back to hires+RSL if Mask_Calc is unavailable.
        mask_calc = None
        if args.transform == "rsl" and mask_calc_dir.exists():
            try:
                mask_calc = load_mask_calc(mask_calc_dir, rc, ecc_s, args.out_size, args.fov)
            except FileNotFoundError:
                mask_calc = None

        extended_list = sorted(set(n_list) - set(PAPER_N)) if args.source == "hybrid" else []
        manifest_rows: list = []

        for n in n_list:
            spec = StimSpec(e_deg=e_deg, n=n)
            label = spec.label

            # File index: paper naming (1..8) for h5 SFs in hybrid, else sequential
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

            # ---- RSL hybrid: paper's 8 from h5 --------------------------
            if h5_data is not None and n in PAPER_N and e_deg in ecc_idx_map_h5:
                freq_idx = PAPER_N.index(n)
                h5_idx = freq_idx * 5 + ecc_idx_map_h5[e_deg]
                und_u8 = _from_unit_float(h5_data["undistorted"][h5_idx])
                dist_u8 = _from_unit_float(h5_data["distorted_corrected"][h5_idx])
                if und_u8.ndim == 2:
                    und_u8 = cv2.cvtColor(und_u8, cv2.COLOR_GRAY2BGR)
                    dist_u8 = cv2.cvtColor(dist_u8, cv2.COLOR_GRAY2BGR)
                _imwrite_jpeg(undist_path, und_u8, args.jpeg_quality)
                _imwrite_jpeg(dist_path, dist_u8, args.jpeg_quality)
                done += 1
                print(f"[{done}/{total}] ecc={ecc_s}  n={n}  label={label}  [h5] → {fname}")
                manifest_rows.append({
                    "filename": fname, "e_deg": e_deg, "n": n, "label_2n": label,
                    "cpr": round(spec.cpr, 4), "cycles_around_ring": round(spec.cycles_around_ring, 2),
                    "transform": "h5", "source": "h5",
                })
                continue

            # ---- Undistorted (shared) ------------------------------------
            undist_out = make_sinring(args.out_size, e_deg, n, mask_out, phase=args.phase)

            # ---- Distorted ----------------------------------------------
            if args.transform == "fisheye":
                # Warp the undistorted sinring through the fisheye transform.
                # Background maps to 0 in float space, so out-of-bounds pixels
                # (zero-padded by grid_sample) become gray (128) after conversion.
                # crop_to_valid=True trims to the valid region, matching what the
                # training network sees via scotoma_dataset.py.
                distorted_out = apply_fisheye_to_sinring(undist_out, fisheye_transform)

            else:  # rsl
                if mask_calc is not None:
                    # Preferred: generate sinring directly in RSL output space.
                    # Mask_Calc places the ring at the cortical radius (fi(e)/fi(max)×128),
                    # so no geometric warp of the image is needed.
                    distorted_raw = make_sinring(
                        args.out_size, e_deg, n, mask_calc, phase=args.phase,
                    )
                else:
                    # Fallback: hi-res sinring → RSL warp.  Produces very thin rings
                    # due to extreme local compression (~0.016× at e=8.5°); not
                    # recommended.
                    undist_hires = make_sinring(args.hires, e_deg, n, mask_hires, phase=args.phase)
                    distorted_raw, _ = rc.distort_image(
                        image=undist_hires, fov=float(args.fov),
                        out_size=args.out_size, inv=0, type=1, series=1,
                    )
                    if distorted_raw.ndim == 3:
                        distorted_raw = distorted_raw[..., 0]

                # RSL affine correction
                if correction_maps is not None:
                    a, b = get_correction_for_stimulus(
                        correction_maps, n, ecc_s, correction_granularity,
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
                "filename": fname, "e_deg": e_deg, "n": n, "label_2n": label,
                "cpr": round(spec.cpr, 4), "cycles_around_ring": round(spec.cycles_around_ring, 2),
                "transform": args.transform,
                "rsl_correction": getattr(args, "rsl_correction", "-") if args.transform == "rsl" else "-",
                "source": "generated",
            })

        if args.write_manifest and manifest_rows:
            manifest_rows.sort(key=lambda r: (r["e_deg"], r["label_2n"]))
            write_manifest(undist_dir / "manifest.csv", manifest_rows)

    print(f"\nDone. {done} stimuli written to {out_root}")


if __name__ == "__main__":
    main()
