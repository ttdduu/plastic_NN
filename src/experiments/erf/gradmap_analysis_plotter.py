"""
gradmap_analysis_plotter.py

Plot orientation and spatial-frequency preference maps derived from Gabor fits
on gradient-based receptive fields (gradmaps).

Inputs (from gradmap_analysis.py _gabor_fits.npz):
    theta   (C, feat_H, feat_W)  preferred orientation [rad, 0..π)
    lambda_ (C, feat_H, feat_W)  preferred spatial wavelength [pixels, retinal space]
    r       (C, feat_H, feat_W)  Pearson r (fit quality, used as selectivity/weight)

Orientation plots (analogous to netStats_viz.py):
    - Population orientation map: selectivity-weighted circular mean across channels
    - Unfolded orientation map:   all channels tiled into spatial sheet

SF plots (analogous to daCosta2024/Analyze_EccSF.py):
    - Spatial map of geometric-mean preferred wavelength (over channels per position)
    - Preferred wavelength vs eccentricity: positions binned into concentric rings
      (geometric mean ± geometric std across all channels × positions in each ring)

Style matches netStats_viz.py (HSV for ori, viridis for SF).
"""

import math
import os

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from src.Lu2025.orientation_map_lc import (
    _channel_dims, channels_to_sheet, make_hsv_ori_map, _ori_colorbar,
)
from src.experiments.erf.netStats_viz import compute_eccentricity_map


# ── Core computation ──────────────────────────────────────────────────────────

def compute_population_ori_map(theta: np.ndarray, r: np.ndarray):
    """Selectivity-weighted circular mean of Gabor-fit orientations across channels.

    Args:
        theta: (C, H, W) preferred orientation [rad, 0..π) from Gabor fits.
        r:     (C, H, W) Pearson r used as per-neuron selectivity weight.
               Negative values (poor / inverted fits) are clamped to zero.

    Returns:
        pop_pref: (H, W) in [0, 2π) double-angle space — pass to make_hsv_ori_map.
        pop_sel:  (H, W) population coherence in [0, 1] — use as saturation.
    """
    r = np.clip(r, 0, None)
    # theta_code uses image-y-down convention: theta_visual = π − theta_code.
    # In double-angle space: 2*(π−θ) = 2π−2θ → cos unchanged, sin negated.
    double = 2.0 * theta                                    # [0, 2π)
    cos_sum = np.sum(np.cos(double) * r, axis=0)            # (H, W)
    sin_sum = np.sum(np.sin(double) * r, axis=0)
    pop_pref = np.arctan2(-sin_sum, cos_sum) % (2 * math.pi)
    resultant = np.sqrt(cos_sum ** 2 + sin_sum ** 2)
    pop_sel = resultant / (np.sum(r, axis=0) + 1e-8)
    return pop_pref, pop_sel


# ── Plotting functions ────────────────────────────────────────────────────────

def plot_population_ori_map(
    theta: np.ndarray,
    r: np.ndarray,
    title: str,
    output_path: str,
):
    """Save population orientation preference heatmap.

    Args:
        theta:       (C, H, W) preferred orientation [rad, 0..π)
        r:           (C, H, W) Pearson r (selectivity weight)
        title:       figure title
        output_path: where to save the PNG
    """
    pop_pref, pop_sel = compute_population_ori_map(theta, r)

    fig, (ax_map, ax_cb) = plt.subplots(
        2, 1, figsize=(6, 6.5),
        gridspec_kw={"height_ratios": [20, 1]},
    )
    rgb = make_hsv_ori_map(
        np.nan_to_num(pop_pref, nan=0.0),
        np.clip(np.nan_to_num(pop_sel, nan=0.0), 0, 1),
    )
    ax_map.imshow(rgb, interpolation="nearest", origin="upper")
    ax_map.set_title(title, fontsize=9)
    ax_map.axis("off")
    _ori_colorbar(ax_cb)
    fig.tight_layout(h_pad=1.5)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_unfolded_ori_map(
    theta: np.ndarray,
    r: np.ndarray,
    title: str,
    output_path: str,
    max_unfolded_px: int = 2048,
    sat_mode: str = "selectivity",
):
    """Unfold all channels into a spatial sheet and save as HSV orientation image.

    Each spatial position is expanded into a c0×c1 tile, one cell per channel.

    Args:
        theta:           (C, H, W) preferred orientation [rad, 0..π)
        r:               (C, H, W) Pearson r (fit quality / selectivity)
        title:           figure title
        output_path:     where to save the PNG
        max_unfolded_px: downsample if sheet height would exceed this
        sat_mode:        "selectivity" – saturation encodes Pearson r (recommended)
                         "uniform"     – all tiles fully saturated
    """
    double = (2.0 * (math.pi - theta)) % (2 * math.pi)   # visual conv: θ_vis = π−θ_code
    r_clipped = np.clip(r, 0, None)

    # (C, H, W) → (H, W, C) for channels_to_sheet
    pref_hwc = double.transpose(1, 2, 0)
    sel_hwc  = r_clipped.transpose(1, 2, 0)

    H, W, C = pref_hwc.shape
    c0, c1  = _channel_dims(C)
    scale   = max(1, H * c0 // max_unfolded_px)
    p_ds    = pref_hwc[::scale, ::scale, :]
    s_ds    = sel_hwc[::scale, ::scale, :]

    sheet_pref = channels_to_sheet(p_ds)
    sheet_sel  = channels_to_sheet(s_ds)

    if sat_mode == "uniform":
        sat = np.ones_like(sheet_sel)
    elif sat_mode == "selectivity":
        sat = np.clip(sheet_sel, 0, 1)
    else:
        raise ValueError(f"Unknown sat_mode: '{sat_mode}'. Choose 'selectivity' or 'uniform'.")

    rgb_sheet = make_hsv_ori_map(sheet_pref, sat)

    Hd, Wd = p_ds.shape[:2]
    fig, (ax_map, ax_cb) = plt.subplots(
        2, 1, figsize=(10, 10.5),
        gridspec_kw={"height_ratios": [20, 1]},
    )
    ax_map.imshow(rgb_sheet, interpolation="nearest")
    ax_map.set_title(
        f"{title}\n"
        f"{Hd}×{Wd} spatial × {C} channels → {Hd*c0}×{Wd*c1} sheet "
        f"(tile={c0}×{c1}, scale=1/{scale})",
        fontsize=9,
    )
    ax_map.axis("off")
    _ori_colorbar(ax_cb)
    fig.tight_layout(h_pad=1.5)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Top-level entry ───────────────────────────────────────────────────────────

def plot_gradmap_ori_maps(
    fits_npz_path: str,
    out_dir: str,
    model_name: str,
    layer_name: str,
    r_threshold: float = 0.3,
    sat_mode: str = "selectivity",
):
    """Load Gabor fits and save population + unfolded orientation preference maps.

    Args:
        fits_npz_path: path to _gabor_fits.npz from gradmap_analysis.py
        out_dir:       directory to save plots
        model_name:    used in filenames and titles
        layer_name:    used in filenames and titles
        r_threshold:   neurons with Pearson r below this contribute zero weight
                       in the population map (treated as unfit / non-oriented)
        sat_mode:      saturation encoding for the unfolded map
    """
    os.makedirs(out_dir, exist_ok=True)

    d = np.load(fits_npz_path)
    theta = d["theta"]   # (C, H, W) in [0, π)
    r     = d["r"]       # (C, H, W) Pearson r

    # For the population map, down-weight neurons whose Gabor fit is poor
    r_weighted = np.where(r >= r_threshold, r, 0.0)

    layer_clean = layer_name.replace(".", "_")

    # ── Population map ─────────────────────────────────────────────────────────
    pop_path = os.path.join(
        out_dir, f"ori_map_population_{model_name}_{layer_clean}.png"
    )
    plot_population_ori_map(
        theta, r_weighted,
        title=(
            f"Population orientation map\n"
            f"{model_name} | {layer_name} | r ≥ {r_threshold}"
        ),
        output_path=pop_path,
    )
    print(f"Saved population map → {pop_path}")

    # ── Unfolded map ───────────────────────────────────────────────────────────
    unfolded_path = os.path.join(
        out_dir, f"ori_map_unfolded_{model_name}_{layer_clean}.png"
    )
    plot_unfolded_ori_map(
        theta, r,
        title=f"Unfolded orientation map | {model_name} | {layer_name}",
        output_path=unfolded_path,
        sat_mode=sat_mode,
    )
    print(f"Saved unfolded map   → {unfolded_path}")


# ── SF spatial map ────────────────────────────────────────────────────────────

def plot_lambda_map(
    lambda_: np.ndarray,
    r: np.ndarray,
    title: str,
    output_path: str,
    r_threshold: float = 0.0,
):
    """2D heatmap of geometric-mean preferred wavelength per spatial position.

    Collapses channels by taking the geometric mean over all channels at each
    (H, W) position. Geometric mean = exp(mean(log(λ))) is appropriate because
    SF preferences scale in octaves (log space), so the natural average is
    multiplicative, not additive.

    Args:
        lambda_:     (C, H, W) preferred wavelength [pixels, retinal space]
        r:           (C, H, W) Pearson r
        title:       figure title
        output_path: where to save the PNG
        r_threshold: neurons with r below this are excluded
    """
    masked = np.where(r >= r_threshold, lambda_, np.nan)
    sfmap = 1.0 / np.exp(np.nanmean(np.log(masked), axis=0))   # (H, W) — geomean SF over channels

    valid = sfmap[np.isfinite(sfmap)]
    vmin, vmax = np.percentile(valid, 2), np.percentile(valid, 98)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sfmap, interpolation="nearest", origin="upper", cmap="viridis",
                   vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, label="Preferred SF [cycles/px]", shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Gabor fit quality map ─────────────────────────────────────────────────────

def plot_r_map(
    r: np.ndarray,
    title: str,
    output_path: str,
):
    """2D heatmap of mean Pearson r (Gabor fit quality) per spatial position.

    Collapses channels by arithmetic mean — r is a linear quantity in [-1, 1],
    not multiplicative, so the arithmetic mean is appropriate here.

    Args:
        r:           (C, H, W) Pearson r from Gabor fits
        title:       figure title
        output_path: where to save the PNG
    """
    rmap = np.nanmean(np.where(r > 0, r, np.nan), axis=0)   # fitted neurons only

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(rmap, interpolation="nearest", origin="upper",
                   cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, label="Mean Pearson r (Gabor fit quality)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── SF vs eccentricity (da Costa style) ──────────────────────────────────────

def compute_lambda_vs_eccentricity(
    lambda_: np.ndarray,
    r: np.ndarray,
    num_ecc_bins: int = 5,
    r_threshold: float = 0.0,
):
    """Preferred wavelength per eccentricity ring (da Costa / Analyze_EccSF pipeline).

    Eccentricity bins are concentric rings in feature-map pixel space, evenly
    spaced from 0 to max_ecc (= half-diagonal of the (H, W) grid).

    For each bin, lambda_ is collected from ALL channels at ALL positions in
    that ring with r >= r_threshold. The central tendency is the geometric mean
    (= exp(mean(log(λ)))), with spread expressed as a geometric ±1 std band
    (= geomean × exp(±std(log(λ)))). This is appropriate because SF preferences
    scale in octaves — the meaningful distance between 4px and 8px wavelength
    is the same as between 8px and 16px (a doubling), so log space is the
    natural domain for averaging and dispersion.

    Args:
        lambda_:      (C, H, W) preferred wavelength [pixels, retinal space]
        r:            (C, H, W) Pearson r fit quality
        num_ecc_bins: number of concentric rings
        r_threshold:  only include neurons with r >= this

    Returns:
        bin_centers: (num_ecc_bins,) eccentricity of each ring [feature-map px]
        geomeans:    (num_ecc_bins,) geometric mean preferred lambda_ per ring
        lo, hi:      (num_ecc_bins,) geomean / exp(std(log)) and geomean * exp(std(log))
        n_neurons:   (num_ecc_bins,) number of valid neurons per ring
    """
    C, H, W = lambda_.shape
    ecc_map = compute_eccentricity_map(H, W)     # (H, W), feature-map px

    max_ecc = ecc_map.max()
    bin_edges = np.linspace(0, max_ecc, num_ecc_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    geomeans, lo_list, hi_list, n_list = [], [], [], []

    for i in range(num_ecc_bins):
        mask_hw = (ecc_map >= bin_edges[i]) & (ecc_map < bin_edges[i + 1])  # (H, W)
        valid = (r >= r_threshold) & mask_hw[np.newaxis, :, :]              # (C, H, W)
        vals = lambda_[valid]
        if len(vals) > 0:
            log_vals = np.log(vals)
            gm = float(np.exp(log_vals.mean()))
            gs = float(log_vals.std())
            geomeans.append(gm)
            lo_list.append(gm / np.exp(gs))    # one geometric std below
            hi_list.append(gm * np.exp(gs))    # one geometric std above
        else:
            geomeans.append(np.nan)
            lo_list.append(np.nan)
            hi_list.append(np.nan)
        n_list.append(int(valid.sum()))

    return (
        bin_centers,
        np.array(geomeans),
        np.array(lo_list),
        np.array(hi_list),
        np.array(n_list),
    )


def plot_lambda_vs_eccentricity(
    lambda_: np.ndarray,
    r: np.ndarray,
    title: str,
    output_path: str,
    num_ecc_bins: int = 5,
    r_threshold: float = 0.0,
):
    """Plot geometric-mean preferred wavelength vs eccentricity (da Costa style).

    Y-axis is log-scaled so the geometric ±1 std band appears symmetric.

    Args:
        lambda_:      (C, H, W) preferred wavelength [pixels, retinal space]
        r:            (C, H, W) Pearson r
        title:        figure title
        output_path:  where to save
        num_ecc_bins: number of concentric eccentricity bins
        r_threshold:  only include neurons with r >= this
    """
    bin_centers, geomeans, lo, hi, n_neurons = compute_lambda_vs_eccentricity(
        lambda_, r, num_ecc_bins=num_ecc_bins, r_threshold=r_threshold,
    )

    # Invert to spatial frequency (cycles/px); lo/hi swap because λ↑ → SF↓
    # Drop the last bin (outermost eccentricity, often undersampled)
    bin_centers = bin_centers[:-1]
    sf_means = (1.0 / geomeans)[:-1]
    sf_lo    = (1.0 / hi)[:-1]
    sf_hi    = (1.0 / lo)[:-1]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(bin_centers, sf_means, marker="o", linewidth=2, color="steelblue")
    ax.fill_between(bin_centers, sf_lo, sf_hi, alpha=0.3, color="steelblue",
                    label="geom. ±1 std")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs="all", numticks=20))
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.set_xlabel("Eccentricity [feature-map px]", fontsize=9)
    ax.set_ylabel("Preferred SF [cycles/px]", fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Top-level SF entry ────────────────────────────────────────────────────────

def plot_gradmap_sf_maps(
    fits_npz_path: str,
    out_dir: str,
    model_name: str,
    layer_name: str,
    r_threshold: float = 0.3,
    num_ecc_bins: int = 5,
):
    """Load Gabor fits and save SF spatial map + preferred wavelength vs eccentricity.

    Args:
        fits_npz_path: path to _gabor_fits.npz from gradmap_analysis.py
        out_dir:       directory to save plots
        model_name:    used in filenames and titles
        layer_name:    used in filenames and titles
        r_threshold:   only include neurons with Pearson r >= this
        num_ecc_bins:  number of concentric eccentricity rings
    """
    os.makedirs(out_dir, exist_ok=True)

    d = np.load(fits_npz_path)
    lambda_ = d["lambda_"]   # (C, H, W) preferred wavelength [px]
    r       = d["r"]         # (C, H, W) Pearson r

    layer_clean = layer_name.replace(".", "_")

    # ── Spatial wavelength map ──────────────────────────────────────────────
    lmap_path = os.path.join(
        out_dir, f"sf_map_spatial_{model_name}_{layer_clean}.png"
    )
    plot_lambda_map(
        lambda_, r,
        title=(
            f"Preferred SF map (geometric mean over channels)\n"
            f"{model_name} | {layer_name} | r ≥ {r_threshold}"
        ),
        output_path=lmap_path,
        r_threshold=r_threshold,
    )
    print(f"Saved SF spatial map  → {lmap_path}")

    # ── Preferred wavelength vs eccentricity ────────────────────────────────
    ecc_path = os.path.join(
        out_dir, f"sf_vs_ecc_{model_name}_{layer_clean}.png"
    )
    plot_lambda_vs_eccentricity(
        lambda_, r,
        title=(
            f"Preferred wavelength vs eccentricity\n"
            f"{model_name} | {layer_name} | r ≥ {r_threshold}"
        ),
        output_path=ecc_path,
        num_ecc_bins=num_ecc_bins,
        r_threshold=r_threshold,
    )
    print(f"Saved SF vs ecc plot  → {ecc_path}")


def plot_gradmap_fit_quality(
    fits_npz_path: str,
    out_dir: str,
    model_name: str,
    layer_name: str,
):
    """Load Gabor fits and save a single fit-quality (mean Pearson r) spatial map."""
    os.makedirs(out_dir, exist_ok=True)

    r = np.load(fits_npz_path)["r"]   # (C, H, W)
    layer_clean = layer_name.replace(".", "_")

    r_path = os.path.join(out_dir, f"gabor_fit_r_{model_name}_{layer_clean}.png")
    plot_r_map(
        r,
        title=f"Gabor fit quality (mean r over channels)\n{model_name} | {layer_name}",
        output_path=r_path,
    )
    print(f"Saved fit quality map → {r_path}")


if __name__ == "__main__":
    # model_name = "4i2ygs9p"
    model_name = "wpmy4wt0"

    layer_name = "stages.0.0.act"
    fits_npz = (
        f"/home/tomasdu/repos/trained_models/gradmaps/{model_name}/{layer_name}"
        f"/gradmaps_{model_name}_{layer_name}_gabor_fits.npz"
    )
    out_dir = (
        f"/home/tomasdu/repos/trained_models/gradmaps/{model_name}/{layer_name}/gradmap_plots_{model_name}_{layer_name.replace('.', '_')}-SF_instead_of_lambda"
    )
    plot_gradmap_ori_maps(
        fits_npz_path=fits_npz,
        out_dir=out_dir,
        model_name=model_name,
        layer_name=layer_name,
        r_threshold=0.5,
        sat_mode="selectivity",
    )
    plot_gradmap_sf_maps(
        fits_npz_path=fits_npz,
        out_dir=out_dir,
        model_name=model_name,
        layer_name=layer_name,
        r_threshold=0.3,
        num_ecc_bins=5,
    )
    plot_gradmap_fit_quality(
        fits_npz_path=fits_npz,
        out_dir=out_dir,
        model_name=model_name,
        layer_name=layer_name,
    )
