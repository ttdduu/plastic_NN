"""
Dataset generator for sinusoidal gratings across orientations and spatial frequencies.

This script creates 224x224 grayscale images containing full-field sinusoidal gratings.
It sweeps orientations and spatial frequencies to form a grid, and writes a CSV manifest.

Spatial frequency is defined in cycles per pixel (cpp). Given wavelength λ in pixels,
f = 1 / λ [cpp]. For network RFs whose preferred wavelengths span λ ∈ [2, 20] px,
a good test set spans frequencies f ∈ [1/20, < 1/2] cpp. The 1/2 cpp is the
Nyquist limit for pixel grids; to avoid aliasing, we cap the maximum at 0.45 cpp by default.

Orientation: By default, we treat orientation (π-periodic) and sample evenly in [0, π).
If you need direction (2π-periodic), set --angle-range 2pi to sample in [0, 2π).

Outputs:
- Images as PNGs
- CSV manifest with columns: filename, orientation_deg, orientation_rad, frequency_cpp,
  wavelength_px, phase_rad, image_size
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image


AngleRange = Tuple[float, float]


@dataclass
class GenerationConfig:
    output_dir: str
    image_size: int = 256
    num_orientations: int = 10
    num_frequencies: int = 20  # More frequencies to cover extended range
    lambda_min_px: float = 2.0  # Finest grating: 2px wavelength → clipped by max_frequency_cpp
    lambda_max_px: float = 300.0  # Coarsest grating: 300px wavelength → ~0.0033 cpp
    max_frequency_cpp: float = 0.48  # Close to Nyquist (0.5), high SF end
    angle_range_mode: str = "pi"  # "pi" for [0, π), "2pi" for [0, 2π)
    num_phases: int = 4  # Phases evenly sampled in [0, 2π); use 1 for phase_rad=0 only
    contrast: float = 1.0  # Peak-to-peak Michelson contrast in [0, 1]
    mean_luminance: float = 0.5  # In [0, 1]
    apply_hann_window: bool = False


def generate_orientations(num_orientations: int, mode: str) -> np.ndarray:
    end = math.pi if mode == "pi" else 2.0 * math.pi
    # endpoint=False to avoid duplicating the wrap-around angle
    return np.linspace(0.0, end, num_orientations, endpoint=False, dtype=np.float64)


def generate_frequencies_from_wavelengths(
    num_frequencies: int,
    lambda_min_px: float,
    lambda_max_px: float,
    max_frequency_cpp: float,
) -> np.ndarray:
    # Log-space sampling in wavelength produces perceptually even SF steps
    wavelengths = np.logspace(
        math.log10(lambda_min_px), math.log10(lambda_max_px), num=num_frequencies, dtype=np.float64
    )
    frequencies = 1.0 / wavelengths
    if max_frequency_cpp is not None:
        # Avoid aliasing near Nyquist (0.5 cpp)
        frequencies = np.clip(frequencies, a_min=None, a_max=max_frequency_cpp)
        # Ensure strict monotonicity after clipping by deduplicating while preserving order
        unique_freqs: List[float] = []
        for f in frequencies.tolist():
            if len(unique_freqs) == 0 or abs(f - unique_freqs[-1]) > 1e-9:
                unique_freqs.append(f)
        frequencies = np.array(unique_freqs, dtype=np.float64)
    return frequencies


def generate_sinusoidal_grating(
    image_size: int,
    frequency_cpp: float,
    orientation_rad: float,
    phase_rad: float,
    contrast: float,
    mean_luminance: float,
    apply_hann_window: bool,
) -> np.ndarray:
    height = image_size
    width = image_size
    # Coordinate grid centered at image center, measured in pixels
    y = np.arange(height, dtype=np.float64) - (height - 1) / 2.0
    x = np.arange(width, dtype=np.float64) - (width - 1) / 2.0
    xx, yy = np.meshgrid(x, y)

    # Project coordinates onto grating axis
    kx = math.cos(orientation_rad)
    ky = math.sin(orientation_rad)
    # Spatial argument in cycles: f * (x*kx + y*ky)
    cycles = frequency_cpp * (xx * kx + yy * ky)
    radians = 2.0 * math.pi * cycles + phase_rad

    # Pure sinusoid in [-1, 1]
    grating = np.sin(radians, dtype=np.float64)

    # Michelson contrast scaling: I = I_mean * (1 + C * s), clip to [0, 1]
    image = mean_luminance * (1.0 + contrast * grating)
    image = np.clip(image, 0.0, 1.0)
    return image


def save_image(image01: np.ndarray, path: str) -> None:
    # Convert to 8-bit grayscale
    array_uint8 = np.round(image01 * 255.0).astype(np.uint8)
    img = Image.fromarray(array_uint8, mode="L")
    img.save(path)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_dataset(config: GenerationConfig) -> None:
    images_dir = os.path.join(config.output_dir, "images")
    ensure_dir(images_dir)

    orientations = generate_orientations(config.num_orientations, config.angle_range_mode)
    frequencies = generate_frequencies_from_wavelengths(
        num_frequencies=config.num_frequencies,
        lambda_min_px=config.lambda_min_px,
        lambda_max_px=config.lambda_max_px,
        max_frequency_cpp=config.max_frequency_cpp,
    )
    phases = np.linspace(0.0, 2.0 * math.pi, config.num_phases, endpoint=False)

    manifest_path = os.path.join(config.output_dir, "manifest.csv")
    with open(manifest_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "filename",
                "orientation_deg",
                "orientation_rad",
                "frequency_cpp",
                "wavelength_px",
                "phase_rad",
                "image_size",
            ]
        )

        # Loop order: ori → phase → freq
        # Downstream reshape: (NUM_ORI, NUM_PHASES, NUM_FREQ, C, X, Y)
        # → max over phase → (NUM_ORI, NUM_FREQ, C, X, Y)
        for oi, theta in enumerate(orientations):
            theta_deg = (theta * 180.0 / math.pi) % 360.0
            for pi, phase in enumerate(phases):
                for fi, freq in enumerate(frequencies):
                    wavelength = 1.0 / freq if freq > 0.0 else float("inf")
                    filename = f"grating_o{oi:02d}_p{pi:02d}_f{fi:02d}.png"
                    path = os.path.join(images_dir, filename)

                    image = generate_sinusoidal_grating(
                        image_size=config.image_size,
                        frequency_cpp=freq,
                        orientation_rad=theta,
                        phase_rad=phase,
                        contrast=config.contrast,
                        mean_luminance=config.mean_luminance,
                        apply_hann_window=config.apply_hann_window,
                    )
                    save_image(image, path)

                    writer.writerow(
                        [
                            os.path.join("images", filename),
                            f"{theta_deg:.6f}",
                            f"{theta:.9f}",
                            f"{freq:.9f}",
                            f"{wavelength:.6f}",
                            f"{phase:.9f}",
                            config.image_size,
                        ]
                    )

    n_images = config.num_orientations * config.num_phases * len(frequencies)
    print(
        f"Wrote {n_images} images to {images_dir} and manifest to {manifest_path}"
    )


def parse_args() -> GenerationConfig:
    parser = argparse.ArgumentParser(description="Generate a grid of sinusoidal gratings.")
    parser.add_argument("--output-dir", type=str, default="/home/tomasdu/repos/datasets/gratings-8phases-256-20SFs/val",help="Directory to write images and CSV")
    parser.add_argument("--image-size", type=int, default=256, help="Square image size (pixels)")
    parser.add_argument("--num-orientations", type=int, default=10, help="Number of orientations")
    parser.add_argument("--num-frequencies", type=int, default=20, help="Number of spatial frequencies")
    parser.add_argument("--lambda-min", type=float, default=1.0, help="Minimum wavelength in pixels")
    parser.add_argument("--lambda-max", type=float, default=200.0, help="Maximum wavelength in pixels")
    parser.add_argument(
        "--max-frequency-cpp",
        type=float,
        default=0.48,
        help="Maximum spatial frequency in cycles/pixel to avoid aliasing (<= 0.5)",
    )
    parser.add_argument(
        "--angle-range",
        choices=["pi", "2pi"],
        default="2pi",
        help="Orientation range: 'pi' for [0, π), '2pi' for [0, 2π)",
    )
    parser.add_argument("--num-phases", type=int, default=4,
                        help="Number of phases evenly sampled in [0, 2π) per orientation×SF")
    parser.add_argument(
        "--contrast",
        type=float,
        default=1.0,
        help="Michelson contrast in [0, 1]; 1.0 yields full dynamic range",
    )
    parser.add_argument("--mean", type=float, default=0.5, help="Mean luminance in [0, 1]")
    parser.add_argument(
        "--hann",
        action="store_true",
        help="Apply a 2D Hann window to reduce edge artifacts (default: off)",
    )

    args = parser.parse_args()
    return GenerationConfig(
        output_dir=args.output_dir,
        image_size=args.image_size,
        num_orientations=args.num_orientations,
        num_frequencies=args.num_frequencies,
        lambda_min_px=args.lambda_min,
        lambda_max_px=args.lambda_max,
        max_frequency_cpp=args.max_frequency_cpp,
        angle_range_mode=args.angle_range,
        num_phases=args.num_phases,
        contrast=args.contrast,
        mean_luminance=args.mean,
        apply_hann_window=args.hann,
    )


def main() -> None:
    config = parse_args()
    build_dataset(config)


if __name__ == "__main__":
    main()


