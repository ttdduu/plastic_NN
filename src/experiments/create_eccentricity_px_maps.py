import json
import os
from typing import Dict, List

import numpy as np


# Output directory for precomputed eccentricity maps
OUTPUT_DIR = "/home/tomasdu/repos/trained_models/02yo20f3/grid_search/radius_maps"

# Feature-map sizes per stage (from your description)
STAGE_SIZES: List[int] = [221, 110, 55, 27]

# Base size that represents the input-pixel scale (used to express radii
# in the same units across stages). This should match your input grid size.
BASE_INPUT_SIZE = 221


def compute_radius_map(size: int) -> np.ndarray:
    """Return radius map in stage pixels for a square map of given size.

    Center is at (size-1)/2 to match earlier stimulus convention.
    """
    center = (size - 1) / 2.0
    y = np.arange(size, dtype=np.float64) - center
    x = np.arange(size, dtype=np.float64) - center
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx * xx + yy * yy, dtype=np.float64)
    return r


def save_radius_maps() -> Dict[str, Dict[str, str]]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest: Dict[str, Dict[str, str]] = {}

    for size in STAGE_SIZES:
        # Radius in stage pixels
        r_stage = compute_radius_map(size)

        # Radius expressed in input-pixel units so all stages are comparable.
        # If the stage feature map is downsampled relative to the input, each
        # stage pixel spans BASE_INPUT_SIZE / size input pixels.
        scale = BASE_INPUT_SIZE / float(size)
        r_input = r_stage * scale

        base = f"radius_size{size}"
        stage_path = os.path.join(OUTPUT_DIR, f"{base}_stage_pixels.npy")
        input_path = os.path.join(OUTPUT_DIR, f"{base}_input_pixels.npy")
        np.save(stage_path, r_stage.astype(np.float32))
        np.save(input_path, r_input.astype(np.float32))

        manifest[str(size)] = {
            "size": str(size),
            "center": str((size - 1) / 2.0),
            "scale_input_per_stage_pixel": str(scale),
            "radius_stage_pixels": stage_path,
            "radius_input_pixels": input_path,
        }

    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote radius maps and manifest to {OUTPUT_DIR}")
    return manifest


if __name__ == "__main__":
    save_radius_maps()


