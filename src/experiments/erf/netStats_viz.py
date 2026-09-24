import math
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from src.Lu2025.orientation_map_lc import _channel_dims, channels_to_sheet, make_hsv_ori_map, _ori_colorbar
from src.experiments.erf.netStats import correct_orientation_for_fisheye


# First layer for special/other channel comparison


NUM_ORI = 10    # Updated from manifest at runtime
NUM_PHASES = 8  # Updated from manifest at runtime
ORI_RADIANS = None  # Sorted unique orientation values in radians; set from manifest at runtime
SPECIAL_CHANNELS = None

def compute_orientation_per_channel(acts, channel_indices, num_freq, sf_idx, sel_mode="vector_strength"):
    """Circular mean per channel (no channel median).

    Args:
        acts:            (NUM_ORI * num_freq, C, X, Y) — phase-max'd activations
        sf_idx:          which spatial frequency to use
        sel_mode:        how to compute selectivity:
                           "vector_strength" – resultant / sum(acts)  [0,1], matches Lu et al. 2025
                           "resultant"       – raw √(x²+y²), unnormalized

    Returns:
        preferred:   (C, X, Y) in [0, 2π) double-angle space
        selectivity: (C, X, Y)
    """
    acts = acts[:, channel_indices, :, :]
    acts = acts.reshape(NUM_ORI, num_freq, *acts.shape[1:])  # (O, F, C, X, Y)
    acts = acts[:, sf_idx, :, :]                             # (O, C, X, Y)
    acts = np.clip(acts, 0, None)                            # ignore suppression (match paper)
    # Double-angle trick: map orientation θ → 2θ so that opposite orientations
    # (θ and θ+π) reinforce rather than cancel in the circular mean.
    # Works for both π-periodic ([0,π)) and 2π-periodic ([0,2π)) datasets.
    oris = (2 * ORI_RADIANS)[:, np.newaxis, np.newaxis, np.newaxis]  # (O, 1, 1, 1)
    x = np.sum(np.cos(oris) * acts, axis=0)                  # (C, X, Y)
    y = np.sum(np.sin(oris) * acts, axis=0)
    # The grating dataset uses kx=cos(θ), ky=sin(θ) with y increasing downward
    # (image convention). This maps θ → physical bar orientation = 90° − θ (mod 180°),
    # NOT θ + 90°. In double-angle space this is a reflection: physical_double = π − 2θ,
    # which corresponds to negating x before arctan2 (not adding π).
    preferred = np.arctan2(y, -x) % (2 * math.pi)
    resultant = np.sqrt(x**2 + y**2)
    if sel_mode == "vector_strength":
        selectivity = resultant / (np.sum(acts, axis=0) + 1e-8)
    elif sel_mode == "resultant":
        selectivity = resultant
    else:
        raise ValueError(f"Unknown sel_mode: '{sel_mode}'. Choose 'vector_strength' or 'resultant'.")
    return preferred, selectivity


def compute_orientation_preference_map(acts, channel_indices, num_freq, sf_idx, sel_mode="vector_strength"):
    """Population orientation preference map — Lu et al. 2025 two-step approach.

    Step 1: per-channel circular mean → (C, X, Y) preferred + selectivity.
    Step 2: selectivity-weighted circular mean across channels → (X, Y).

    Args:
        acts:       (NUM_ORI * num_freq, C, X, Y) — phase-max'd activations
        sf_idx:     which spatial frequency to use
        sel_mode:   passed to compute_orientation_per_channel (see its docstring)

    Returns:
        pop_pref: (X, Y) in [0, 2π) double-angle space (pass directly to make_hsv_ori_map)
        pop_sel:  (X, Y) mean selectivity across channels, in [0, 1] (use as saturation)
    """
    preferred, selectivity = compute_orientation_per_channel(
        acts, channel_indices, num_freq, sf_idx, sel_mode
    )
    # preferred: (C, X, Y) in [0, 2π) double-angle space
    # selectivity-weighted circular mean across channels
    cos_sum = np.sum(np.cos(preferred) * selectivity, axis=0)  # (X, Y)
    sin_sum = np.sum(np.sin(preferred) * selectivity, axis=0)
    pop_pref = np.arctan2(sin_sum, cos_sum) % (2 * math.pi)   # [0, 2π) double-angle
    # pop_sel  = selectivity.mean(axis=0)                        # mean over channels → (X, Y)
    # Population coherence: resultant magnitude / sum of selectivities.
    # Mirrors the per-channel vector_strength formula one level up: high only when
    # channels both (a) are individually selective AND (b) agree on orientation.
    resultant = np.sqrt(cos_sum**2 + sin_sum**2)
    pop_sel   = resultant / (np.sum(selectivity, axis=0) + 1e-8)  # (X, Y) in [0, 1]
    return pop_pref, pop_sel


def plot_unfolded_ori_map(acts, channel_indices, num_freq, sf_idx, sf_label, output_path, max_unfolded_px=2048, sel_mode="vector_strength", sat_mode="uniform", px_per_channel=4):
    """Unfold channels into a spatial sheet (Lu et al. 2025 style) and save as HSV image.

    Each spatial position becomes a c0×c1 tile, one tile per channel.

    sat_mode: "uniform"     – all pixels fully saturated (vivid, matches Lu et al. 2025 paper)
              "selectivity" – saturation encodes vector strength (desaturated = untuned)
    """
    preferred, selectivity = compute_orientation_per_channel(acts, channel_indices, num_freq, sf_idx, sel_mode)
    # (C, X, Y) → (X, Y, C) = (H, W, C)
    preferred_hwc   = preferred.transpose(1, 2, 0)
    selectivity_hwc = selectivity.transpose(1, 2, 0)

    H, W, C = preferred_hwc.shape
    c0, c1  = _channel_dims(C)
    scale   = max(1, H * c0 // max_unfolded_px)
    p_ds    = preferred_hwc[::scale, ::scale, :]
    s_ds    = selectivity_hwc[::scale, ::scale, :]

    sheet_pref = channels_to_sheet(p_ds)                      # (H·c0, W·c1)
    sheet_sel  = channels_to_sheet(s_ds)

    # Upsample each channel square to px_per_channel×px_per_channel pixels so
    # individual channels are visible at normal zoom levels (px_per_channel=1 → no-op).
    if px_per_channel > 1:
        sheet_pref = np.repeat(np.repeat(sheet_pref, px_per_channel, axis=0), px_per_channel, axis=1)
        sheet_sel  = np.repeat(np.repeat(sheet_sel,  px_per_channel, axis=0), px_per_channel, axis=1)
    grid_c0 = c0 * px_per_channel
    grid_c1 = c1 * px_per_channel

    if sat_mode == "uniform":
        sat = np.ones_like(sheet_sel)
    elif sat_mode == "selectivity":
        sat = np.clip(sheet_sel, 0, 1)
    else:
        raise ValueError(f"Unknown sat_mode: '{sat_mode}'. Choose 'uniform' or 'selectivity'.")
    rgb_sheet  = make_hsv_ori_map(sheet_pref, sat)

    Hd, Wd = p_ds.shape[:2]
    sheet_h, sheet_w = rgb_sheet.shape[:2]
    dpi = 100
    cb_h_in = 0.4
    fig, axes = plt.subplots(2, 1, figsize=(sheet_w / dpi, sheet_h / dpi + cb_h_in),
                              gridspec_kw={"height_ratios": [sheet_h / dpi, cb_h_in]})
    ax_map, ax_cb = axes
    ax_map.imshow(rgb_sheet, interpolation="nearest")
    # --- XY-position grid overlay (comment out to remove) ---
    #for xi in range(1, Wd):
    #    ax_map.axvline(xi * grid_c1 - 0.5, color="black", linewidth=0.7, alpha=0.8)
    #for yi in range(1, Hd):
    #    ax_map.axhline(yi * grid_c0 - 0.5, color="black", linewidth=0.7, alpha=0.8)
    # ---------------------------------------------------------
    ax_map.set_title(
        f"Unfolded orientation map | {sf_label}\n"
        f"{Hd}×{Wd} spatial × {C} channels → {sheet_h}×{sheet_w} sheet "
        f"(tile={c0}×{c1}, px/channel={px_per_channel}, scale=1/{scale})",
        fontsize=9,
    )
    ax_map.axis("off")
    _ori_colorbar(ax_cb)
    fig.tight_layout(h_pad=1.5)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def compute_sf_preference_map(acts, channel_indices, num_freq, freqs):
    """
    Compute spatial frequency preference map (argmax over SF after median over orientations).

    Args:
        acts: (NUM_ORI * num_freq, C, X, Y) raw activations
        channel_indices: list of channel indices to use
        num_freq: number of spatial frequencies
        freqs: array of frequency values (cpp) to map indices to

    Returns:
        sf_pref_map: (X, Y) preferred SF values in cpp
    """
    acts = acts[:, channel_indices, :, :]  # filter channels

    # Reshape to (O, F, C, X, Y)
    acts = acts.reshape(NUM_ORI, num_freq, *acts.shape[1:])

    # Collapse channels (median)
    acts = np.median(acts, axis=2)  # (O, F, X, Y)

    # Median over orientations -> (F, X, Y)
    acts_sf = np.median(acts, axis=0)  # (F, X, Y)

    # Argmax over SF -> (X, Y) indices
    sf_idx = np.argmax(acts_sf, axis=0)  # (X, Y)

    # Map indices to actual frequency values
    sf_pref_map = freqs[sf_idx]

    return sf_pref_map

def plot_orientation_per_channel(acts, num_freq, output_path,
                                 channels=None, ncols=6, sf_idx=0, sf_label="", sel_mode="vector_strength"):

    num_channels_total = acts.shape[1]

    if channels is None:
        channels = list(range(num_channels_total))

    n_ch = len(channels)
    nrows = int(np.ceil(n_ch / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.6 * ncols, 2.6 * nrows + 1.5)  # extra height for legend
    )

    axes = np.atleast_2d(axes)

    for i, ch in enumerate(channels):
        r, c = divmod(i, ncols)
        ax = axes[r, c]

        ori_map, _ = compute_orientation_preference_map(
            acts, [ch], num_freq, sf_idx, sel_mode
        )
        # ori_map is in [0, 2π); divide by 2 to get [0, π) for the hsv colormap
        im = ax.imshow(ori_map / 2, cmap="hsv",
                       origin="lower",
                       vmin=0, vmax=np.pi)

        ax.set_title(f"Ch {ch}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty axes
    for j in range(i + 1, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    if sf_label:
        fig.suptitle(sf_label, fontsize=9, y=0.98)

    # --- GLOBAL ORIENTATION LEGEND ---
    fig.subplots_adjust(bottom=0.15)  # reserve space

    ax_legend = fig.add_axes([0.2, 0.03, 0.6, 0.08])

    num_bars = 9
    orientations = np.linspace(0, np.pi, num_bars, endpoint=True)
    cmap = plt.cm.hsv
    bar_length = 0.4

    ax_legend.set_xlim(-0.5, num_bars - 0.5)
    ax_legend.set_ylim(-0.9, 0.9)
    ax_legend.set_aspect("equal")
    ax_legend.axis("off")

    for i, theta in enumerate(orientations):
        color = cmap(theta / np.pi)
        dx = bar_length * np.cos(theta)
        dy = bar_length * np.sin(theta)

        ax_legend.plot([i - dx, i + dx], [-dy, dy],
                       color=color, linewidth=3)

        deg = int(np.degrees(theta))
        ax_legend.text(i, -0.75, f"{deg}°",
                       ha="center", va="top", fontsize=8)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)

def plot_sf_per_channel(acts, num_freq, freqs, output_path,
                        channels=None, ncols=6):
    """
    Plot SF preference map for each channel individually.

    Args:
        acts: (NUM_ORI * num_freq, C, X, Y) raw activations
        num_freq: number of spatial frequencies
        freqs: array of frequency values (cpp)
        output_path: path to save the figure
        channels: list of channel indices to plot. If None, plot all channels.
        ncols: number of columns in the grid
    """
    num_channels_total = acts.shape[1]

    if channels is None:
        channels = list(range(num_channels_total))

    n_ch = len(channels)
    nrows = int(np.ceil(n_ch / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.6 * ncols, 2.6 * nrows + 1.0)  # extra height for colorbar
    )

    axes = np.atleast_2d(axes)

    # Compute all SF preference maps first to get common vmin/vmax
    sf_maps = []
    for ch in channels:
        sf_map = compute_sf_preference_map(acts, [ch], num_freq, freqs)
        sf_maps.append(sf_map)

    # Common colorbar range
    vmin = min(m.min() for m in sf_maps)
    vmax = max(m.max() for m in sf_maps)

    for i, (ch, sf_map) in enumerate(zip(channels, sf_maps)):
        r, c = divmod(i, ncols)
        ax = axes[r, c]

        im = ax.imshow(sf_map, cmap="viridis",
                       origin="lower",
                       vmin=vmin, vmax=vmax)

        ax.set_title(f"Ch {ch}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty axes
    for j in range(i + 1, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    # Add colorbar
    fig.subplots_adjust(bottom=0.1, right=0.95)
    cbar_ax = fig.add_axes([0.96, 0.1, 0.02, 0.8])
    fig.colorbar(im, cax=cbar_ax, label="SF (cpp)")

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def load_frequencies_from_manifest(manifest_csv: str):
    """Load dataset shape from manifest CSV.

    Returns
    -------
    freqs       : np.ndarray – unique frequency values (one phase's worth)
    num_phases  : int        – number of phases in the dataset
    num_ori     : int        – number of orientations in the dataset
    ori_radians : np.ndarray – sorted unique orientation values in radians
                               (use 2*ori_radians as the double-angle oris for
                               the circular mean, regardless of whether the dataset
                               spans [0,π) or [0,2π))
    """
    with open(manifest_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]

    first_ori = rows[0]["orientation_deg"]
    first_phase = rows[0]["phase_rad"]

    # Collect frequencies from the first (orientation, phase) block only
    freqs = []
    for row in rows:
        if row["orientation_deg"] != first_ori or row["phase_rad"] != first_phase:
            break
        freqs.append(float(row["frequency_cpp"]))

    # Count unique phases in the first orientation block
    phases_seen = set()
    for row in rows:
        if row["orientation_deg"] != first_ori:
            break
        phases_seen.add(row["phase_rad"])
    num_phases = len(phases_seen)

    # Unique orientations — preserve loop order (sorted by appearance = ascending)
    ori_rad_seen = {}
    for row in rows:
        ori_rad_seen[row["orientation_deg"]] = float(row["orientation_rad"])
    ori_radians = np.array(list(ori_rad_seen.values()), dtype=np.float64)

    freqs = np.array(freqs, dtype=np.float32)
    return freqs, num_phases, len(ori_radians), ori_radians


def compute_eccentricity_map(X, Y, *, scale: float = 1.0):
    """
    Compute eccentricity (distance from center) for each position.

    Units are in "layer pixels" by default. Use `scale` to convert to another unit
    (e.g., approximate input-pixel units) by multiplying the distances.
    """
    # Use half-pixel center for even-sized maps (symmetry across center).
    cx, cy = (X - 1) / 2.0, (Y - 1) / 2.0
    xx, yy = np.meshgrid(np.arange(X, dtype=np.float32), np.arange(Y, dtype=np.float32), indexing="ij")
    ecc = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return ecc * float(scale)


def compute_activation_vs_sf_per_eccentricity(
    acts,
    channel_indices,
    num_freq,
    num_ecc_bins=20,
    *,
    ecc_scale: float = 1.0,
    ecc_units: str = "layer_px",
):
    """
    Compute median activation at each SF for different eccentricity bins.

    Args:
        acts: (NUM_ORI * num_freq, C, X, Y) raw activations
        channel_indices: list of channel indices to use
        num_freq: number of spatial frequencies
        num_ecc_bins: number of eccentricity bins

    Returns:
        ecc_bin_labels: labels for each eccentricity bin
        activation_curves: (num_ecc_bins, num_freq) median activation per SF per eccentricity
    """
    # Filter channels
    acts = acts[:, channel_indices, :, :]  # (N, C_filtered, X, Y)

    # Reshape to (O, F, C, X, Y)
    acts = acts.reshape(NUM_ORI, num_freq, *acts.shape[1:])
    print(f"Reshaped activations: {acts.shape}")

    # Max over orientations and channels -> (F, X, Y)
    acts_sf = np.max(acts, axis=0)  # (F, C, X, Y) # over orientations
    print(f"Max over orientations: {acts_sf.shape}")
    acts_sf = np.median(acts_sf, axis=1)  # (F, X, Y) # over channels
    print(f"Median over channels: {acts_sf.shape}")

    X, Y = acts_sf.shape[1], acts_sf.shape[2]
    ecc_map = compute_eccentricity_map(X, Y, scale=ecc_scale)

    # Create eccentricity bins
    max_ecc = ecc_map.max()
    bin_edges = np.linspace(10, max_ecc, num_ecc_bins + 1)

    activation_curves = []
    activation_q1 = []  # 25th percentile
    activation_q3 = []  # 75th percentile
    ecc_bin_labels = []

    for i in range(num_ecc_bins):
        mask = (ecc_map >= bin_edges[i]) & (ecc_map < bin_edges[i + 1])

        # For each SF, get median and IQR across positions in this eccentricity bin
        sf_medians = []
        sf_q1 = []
        sf_q3 = []
        for f in range(num_freq):
            vals = acts_sf[f][mask]
            if len(vals) > 0:
                sf_medians.append(np.median(vals))
                sf_q1.append(np.percentile(vals, 25))
                sf_q3.append(np.percentile(vals, 75))
            else:
                sf_medians.append(np.nan)
                sf_q1.append(np.nan)
                sf_q3.append(np.nan)

        activation_curves.append(sf_medians)
        activation_q1.append(sf_q1)
        activation_q3.append(sf_q3)
        if ecc_units == "normalized":
            # Keep a couple decimals if normalized
            ecc_bin_labels.append(f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}")
        else:
            ecc_bin_labels.append(f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}")

    return ecc_bin_labels, np.array(activation_curves), np.array(activation_q1), np.array(activation_q3)

def plot_orientation_heatmap(pop_pref, pop_sel, title, output_path):
    """Population orientation heatmap matching Lu et al. 2025.

    Args:
        pop_pref: (X, Y) in [0, 2π) double-angle space — from compute_orientation_preference_map
        pop_sel:  (X, Y) mean selectivity across channels — used as saturation
        title:    figure title
        output_path: where to save
    """
    fig, axes = plt.subplots(2, 1, figsize=(6, 6.5),
                              gridspec_kw={"height_ratios": [20, 1]})
    ax_map, ax_cb = axes

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


def plot_sf_special_channels(model_name: str, manifest_csv: str, special_channels=None, layer_name="stages.0.0.dwconv", model_outpath=None, ori_sf_indices=None, sel_mode="vector_strength", sat_mode="uniform", plots_nickname="overridden"):
    """
    Plot SF preference for channels in first layer.

    Args:
        model_name: Name identifier for the model (used to locate grid_search directory)
        special_channels: List of channel indices to compare against others.
                         If None, plot all channels together.
    """
    layer_dir = model_outpath
    os.makedirs(layer_dir, exist_ok=True)
    plots_dir = os.path.join(layer_dir, f"plots-{layer_name}-{plots_nickname}")
    os.makedirs(plots_dir, exist_ok=True)

    global NUM_ORI, NUM_PHASES, ORI_RADIANS

    # Load dataset shape from manifest
    sf_values, num_phases, num_ori, ori_radians = load_frequencies_from_manifest(manifest_csv)
    NUM_PHASES = num_phases
    NUM_ORI = num_ori
    ORI_RADIANS = ori_radians
    num_freq = len(sf_values)
    print(f"Manifest: {num_freq} frequencies, {NUM_PHASES} phases, {NUM_ORI} orientations "
          f"(range {np.degrees(ori_radians[0]):.0f}°–{np.degrees(ori_radians[-1]):.0f}°)")

    # Load raw activations
    acts_path = os.path.join(model_outpath, "acts.npy")
    acts = np.load(acts_path)
    print(f"Loaded activations: {acts.shape}")


    # Collapse phase dimension via max — loop order is ori × phase × freq
    # acts: (NUM_ORI * NUM_PHASES * num_freq, C, X, Y)
    acts = acts.reshape(NUM_ORI, NUM_PHASES, num_freq, *acts.shape[1:])
    acts = acts.max(axis=1)                           # max over phases → (NUM_ORI, num_freq, C, X, Y)
    acts = acts.reshape(NUM_ORI * num_freq, *acts.shape[2:])  # → (NUM_ORI * num_freq, C, X, Y)
    print(f"After phase-max: {acts.shape}")

    # Determine if we're comparing special vs other, or plotting all channels
    compare_mode = special_channels is not None

    # --- PER-CHANNEL SF PREFERENCE FIGURES (one PNG per channel) ---
    num_channels = acts.shape[1]

    # COMMENT OUT IF U DON'T WANT THIS
    # for ch in range(num_channels):
    #     sf_per_channel_outfile = os.path.join(
    #         plots_dir,
    #         f"sf_pref_ch{ch}_{layer_name}_{model_name}.png",
    #     )
    #     # If outfile exists, skip recomputation
    #     if os.path.exists(sf_per_channel_outfile):
    #         continue
    #     else:
    #         plot_sf_per_channel(
    #             acts,
    #             num_freq,
    #             sf_values,
    #             sf_per_channel_outfile,
    #             channels=[ch],
    #             ncols=1,
    #         )

    # --- ORIENTATION PREFERENCE FIGURES (one PNG per channel per SF, one heatmap per SF) ---

    sf_iter = ori_sf_indices if ori_sf_indices is not None else range(num_freq)

    if compare_mode:
        other_channels = [c for c in range(num_channels) if c not in special_channels]
        for fi in sf_iter:
            sf_label = f"SF {sf_values[fi]:.3f} cpp"
            for ch in special_channels:
                outfile = os.path.join(plots_dir, f"orientation_pref_ch{ch}_sf{fi:03d}_{layer_name}_{model_name}.png")
                if not os.path.exists(outfile):
                    plot_orientation_per_channel(acts, num_freq, outfile, channels=[ch], ncols=1, sf_idx=fi, sf_label=sf_label, sel_mode=sel_mode)
            for ch in other_channels:
                outfile = os.path.join(plots_dir, f"orientation_pref_ch{ch}_sf{fi:03d}_{layer_name}_{model_name}.png")
                if not os.path.exists(outfile):
                    plot_orientation_per_channel(acts, num_freq, outfile, channels=[ch], ncols=1, sf_idx=fi, sf_label=sf_label, sel_mode=sel_mode)
            ori_special, sel_special = compute_orientation_preference_map(acts, special_channels, num_freq, fi, sel_mode)
            ori_other, sel_other = compute_orientation_preference_map(
                acts, [c for c in range(num_channels) if c not in special_channels], num_freq, fi, sel_mode,
            )
            # plot_orientation_heatmap(
                # ori_special, sel_special,
                # f"Orientation Preference – Special Channels {special_channels} | {sf_label}",
                # os.path.join(plots_dir, f"orientation_pref_special_sf{fi:03d}_{model_name}.png"),
            # )
            # plot_orientation_heatmap(
                # ori_other, sel_other,
                # f"Orientation Preference – Other Channels | {sf_label}",
                # os.path.join(plots_dir, f"orientation_pref_other_sf{fi:03d}_{model_name}.png"),
            # )
            for ch_list, tag in [(special_channels, "special"), (other_channels, "other")]:
                unfolded_outfile = os.path.join(plots_dir, f"orientation_unfolded_{tag}_sf{fi:03d}_{model_name}.png")
                plot_unfolded_ori_map(acts, ch_list, num_freq, fi, sf_label, unfolded_outfile, sel_mode=sel_mode, sat_mode=sat_mode)

    else:
        all_channels = list(range(num_channels))
        for fi in sf_iter:
            sf_label = f"SF {sf_values[fi]:.3f} cpp"
            # # COMMENT OUT IF U DON'T WANT THIS
            # for ch in range(num_channels):
            #     outfile = os.path.join(plots_dir, f"orientation_pref_ch{ch}_sf{fi:03d}_{layer_name}_{model_name}.png")
            #     if not os.path.exists(outfile):
            #         plot_orientation_per_channel(acts, num_freq, outfile, channels=[ch], ncols=1, sf_idx=fi, sf_label=sf_label, sel_mode=sel_mode)
            ori_all, sel_all = compute_orientation_preference_map(acts, all_channels, num_freq, fi, sel_mode)
            plot_orientation_heatmap(
                ori_all, sel_all,
                f"Orientation Preference – All Channels | {sf_label}",
                os.path.join(plots_dir, f"orientation_pref_all_sf{fi:03d}_{model_name}.png"),
            )
            unfolded_outfile = os.path.join(plots_dir, f"orientation_unfolded_sf{fi:03d}_{model_name}.png")
            plot_unfolded_ori_map(acts, all_channels, num_freq, fi, sf_label, unfolded_outfile, sel_mode=sel_mode, sat_mode=sat_mode)


    print(f"SF values ({num_freq}): {sf_values}")

    if compare_mode:
        # Get channel indices
        num_channels = acts.shape[1]
        other_channels = [c for c in range(num_channels) if c not in special_channels]

        # Load the SF preference maps for heatmaps
        sf_special_path = os.path.join(model_outpath, f"sf_pref_{layer_name}_special_channels.npy")
        sf_other_path = os.path.join(model_outpath, f"sf_pref_{layer_name}_other_channels.npy")

        # These files already contain SF values (frequency in cpp), not wavelength
        sf_special = np.load(sf_special_path)
        sf_other = np.load(sf_other_path)

        # Create figure: 3 rows x 2 columns - heatmaps, curves, SF preference vs eccentricity
        fig = plt.figure(figsize=(14, 16))

        # Top row: heatmaps
        ax_heat1 = fig.add_subplot(3, 2, 1)
        ax_heat2 = fig.add_subplot(3, 2, 2)
        # Middle row: activation curves
        ax_curve1 = fig.add_subplot(3, 2, 3)
        ax_curve2 = fig.add_subplot(3, 2, 4)
        # Bottom row: SF preference vs eccentricity
        ax_pref1 = fig.add_subplot(3, 2, 5)
        ax_pref2 = fig.add_subplot(3, 2, 6)
    else:
        # Plot all channels together - single column layout
        # Load SF preference map for all channels
        sf_all_path = os.path.join(model_outpath, f"sf_pref_{layer_name}_median.npy")
        sf_all = np.load(sf_all_path)

        # Create figure: 3 rows x 1 column
        fig = plt.figure(figsize=(7, 16))

        # Top: heatmap
        ax_heat1 = fig.add_subplot(3, 1, 1)
        # Middle: activation curves
        ax_curve1 = fig.add_subplot(3, 1, 2)
        # Bottom: SF preference vs eccentricity
        ax_pref1 = fig.add_subplot(3, 1, 3)

        # Set dummy axes for compatibility (won't be used)
        ax_heat2 = None
        ax_curve2 = None
        ax_pref2 = None

    # Plot heatmaps
    if compare_mode:
        # Common colorbar range for heatmaps
        vmin = min(sf_special.min(), sf_other.min())
        vmax = max(sf_special.max(), sf_other.max())

        # Heatmap 1: Special channels
        im1 = ax_heat1.imshow(sf_special, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
        ax_heat1.set_title(f"SF Preference - Special Channels\n{special_channels}")
        ax_heat1.set_xlabel("X")
        ax_heat1.set_ylabel("Y")
        fig.colorbar(im1, ax=ax_heat1, shrink=0.7, label="SF (cpp)")

        # Heatmap 2: Other channels
        im2 = ax_heat2.imshow(sf_other, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
        ax_heat2.set_title("SF Preference - Other Channels")
        ax_heat2.set_xlabel("X")
        ax_heat2.set_ylabel("Y")
        fig.colorbar(im2, ax=ax_heat2, shrink=0.7, label="SF (cpp)")
    else:
        # Single heatmap for all channels
        im1 = ax_heat1.imshow(sf_all, cmap="viridis", origin="lower")
        ax_heat1.set_title("SF Preference - All Channels")
        ax_heat1.set_xlabel("X")
        ax_heat1.set_ylabel("Y")
        fig.colorbar(im1, ax=ax_heat1, shrink=0.7, label="SF (cpp)")

    """
    # Compute activation vs SF curves for each eccentricity bin
    num_ecc_bins = 20
    # Eccentricity is computed on the *current layer's* spatial grid (acts.shape[-2:]),
    # so deeper layers automatically have a smaller eccentricity range.
    ecc_units = "layer_px"
    if compare_mode:
        ecc_labels_special, curves_special, q1_special, q3_special = compute_activation_vs_sf_per_eccentricity(
            acts, special_channels, num_freq, num_ecc_bins, ecc_units=ecc_units
        )
        ecc_labels_other, curves_other, q1_other, q3_other = compute_activation_vs_sf_per_eccentricity(
            acts, other_channels, num_freq, num_ecc_bins, ecc_units=ecc_units
        )
    else:
        # Use all channels
        num_channels = acts.shape[1]
        all_channels = list(range(num_channels))
        ecc_labels_all, curves_all, q1_all, q3_all = compute_activation_vs_sf_per_eccentricity(
            acts, all_channels, num_freq, num_ecc_bins, ecc_units=ecc_units
        )

    # Color map for eccentricity bins (center=warm, periphery=cool)
    # Use num_ecc_bins - 1 since we exclude the last bin from plots
    colors = plt.cm.coolwarm(np.linspace(0, 1, num_ecc_bins - 1))

    if compare_mode:
        # Sort by SF ascending (low SF on left, high SF on right) for intuitive reading
        sort_idx = np.argsort(sf_values)
        sf_values_sorted = sf_values[sort_idx]
        curves_special_sorted = curves_special[:, sort_idx]
        curves_other_sorted = curves_other[:, sort_idx]
        q1_special_sorted = q1_special[:, sort_idx]
        q3_special_sorted = q3_special[:, sort_idx]
        q1_other_sorted = q1_other[:, sort_idx]
        q3_other_sorted = q3_other[:, sort_idx]

        # Plot curves for special channels with IQR shading (exclude last eccentricity bin)
        for i, (label, curve, q1, q3) in enumerate(zip(ecc_labels_special[:-1], curves_special_sorted[:-1], q1_special_sorted[:-1], q3_special_sorted[:-1])):
            # ax_curve1.fill_between(sf_values_sorted, q1, q3, color=colors[i], alpha=0.2)
            ax_curve1.plot(sf_values_sorted, curve, 'o-', color=colors[i], label=f"Ecc {label} ({ecc_units})", linewidth=2, markersize=4)

        ax_curve1.set_xlabel("Spatial Frequency (cpp)", fontsize=11)
        ax_curve1.set_ylabel("Median Activation", fontsize=11)
        ax_curve1.set_title(f"Special Channels {special_channels}", fontsize=12)
        ax_curve1.legend(fontsize=9)
        ax_curve1.grid(True, alpha=0.3)
        ax_curve1.set_xscale('log')

        # Use sorted SF values as x-axis ticks (subsample to avoid crowding)
        xticks = sf_values_sorted[::5]  # every 5th value
        ax_curve1.set_xticks(xticks)
        ax_curve1.set_xticklabels([f"{x:.2f}" for x in xticks], rotation=45, ha='right')

        # Plot curves for other channels with IQR shading (exclude last eccentricity bin)
        for i, (label, curve, q1, q3) in enumerate(zip(ecc_labels_other[:-1], curves_other_sorted[:-1], q1_other_sorted[:-1], q3_other_sorted[:-1])):
            # ax_curve2.fill_between(sf_values_sorted, q1, q3, color=colors[i], alpha=0.2)
            ax_curve2.plot(sf_values_sorted, curve, 'o-', color=colors[i], label=f"Ecc {label} ({ecc_units})", linewidth=2, markersize=4)

        ax_curve2.set_xlabel("Spatial Frequency (cpp)", fontsize=11)
        ax_curve2.set_ylabel("Median Activation", fontsize=11)
        ax_curve2.set_title("Other Channels", fontsize=12)
        ax_curve2.legend(fontsize=9)
        ax_curve2.grid(True, alpha=0.3)
        ax_curve2.set_xscale('log')
        ax_curve2.set_xticks(xticks)
        ax_curve2.set_xticklabels([f"{x:.2f}" for x in xticks], rotation=45, ha='right')
    else:
        # Sort by SF ascending
        sort_idx = np.argsort(sf_values)
        sf_values_sorted = sf_values[sort_idx]
        curves_all_sorted = curves_all[:, sort_idx]
        q1_all_sorted = q1_all[:, sort_idx]
        q3_all_sorted = q3_all[:, sort_idx]

        # Plot curves for all channels (exclude last eccentricity bin)
        for i, (label, curve, q1, q3) in enumerate(zip(ecc_labels_all[:-1], curves_all_sorted[:-1], q1_all_sorted[:-1], q3_all_sorted[:-1])):
            ax_curve1.plot(sf_values_sorted, curve, 'o-', color=colors[i], label=f"Ecc {label} ({ecc_units})", linewidth=2, markersize=4)

        ax_curve1.set_xlabel("Spatial Frequency (cpp)", fontsize=11)
        ax_curve1.set_ylabel("Median Activation", fontsize=11)
        ax_curve1.set_title("All Channels", fontsize=12)
        ax_curve1.legend(fontsize=9)
        ax_curve1.grid(True, alpha=0.3)
        ax_curve1.set_xscale('log')

        # Use sorted SF values as x-axis ticks (subsample to avoid crowding)
        xticks = sf_values_sorted[::5]  # every 5th value
        ax_curve1.set_xticks(xticks)
        ax_curve1.set_xticklabels([f"{x:.2f}" for x in xticks], rotation=45, ha='right')

    # --- Bottom row: SF preference (peak) vs eccentricity ---
    # For each eccentricity bin, find the SF that gives max activation

    if compare_mode:
        # Get eccentricity bin centers (midpoints) for x-axis
        # Parse eccentricity labels to get bin centers
        ecc_centers = []
        for label in ecc_labels_special:
            parts = label.split('-')
            low, high = float(parts[0]), float(parts[1])
            ecc_centers.append((low + high) / 2)
        ecc_centers = np.array(ecc_centers)

        # Find preferred SF (argmax) for each eccentricity curve
        pref_sf_special = []
        for curve in curves_special_sorted:
            max_idx = np.argmax(curve)
            pref_sf_special.append(sf_values_sorted[max_idx])
        pref_sf_special = np.array(pref_sf_special)

        pref_sf_other = []
        for curve in curves_other_sorted:
            max_idx = np.argmax(curve)
            pref_sf_other.append(sf_values_sorted[max_idx])
        pref_sf_other = np.array(pref_sf_other)
    else:
        # Get eccentricity bin centers for all channels
        ecc_centers = []
        for label in ecc_labels_all:
            parts = label.split('-')
            low, high = float(parts[0]), float(parts[1])
            ecc_centers.append((low + high) / 2)
        ecc_centers = np.array(ecc_centers)

        # Find preferred SF (argmax) for each eccentricity curve
        pref_sf_all = []
        for curve in curves_all_sorted:
            max_idx = np.argmax(curve)
            pref_sf_all.append(sf_values_sorted[max_idx])
        pref_sf_all = np.array(pref_sf_all)

    # --- Fit exponential curve: a * exp(b * x) + c ---
    # Exclude last eccentricity bin from fit
    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c

    # Helper function to compute R-squared
    def compute_r_squared(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)

    if compare_mode:
        # Fit for special channels (exclude last point)
        ecc_fit = ecc_centers[:-1]
        pref_sf_special_fit = pref_sf_special[:-1]
        try:
            popt_special, _ = curve_fit(exp_func, ecc_fit, pref_sf_special_fit,
                                         p0=[0.1, -0.01, 0.01], maxfev=5000)
            a_s, b_s, c_s = popt_special
            y_pred_special = exp_func(ecc_fit, *popt_special)
            r2_special = compute_r_squared(pref_sf_special_fit, y_pred_special)
            fit_success_special = True
            print(f"Special channels fit: SF = {a_s:.4f} * exp({b_s:.6f} * ecc) + {c_s:.4f}, R² = {r2_special:.4f}")
        except Exception as e:
            print(f"Could not fit special channels: {e}")
            fit_success_special = False
            r2_special = None

        # Fit for other channels (exclude last point)
        pref_sf_other_fit = pref_sf_other[:-1]
        try:
            popt_other, _ = curve_fit(exp_func, ecc_fit, pref_sf_other_fit,
                                       p0=[0.1, -0.01, 0.01], maxfev=5000)
            a_o, b_o, c_o = popt_other
            y_pred_other = exp_func(ecc_fit, *popt_other)
            r2_other = compute_r_squared(pref_sf_other_fit, y_pred_other)
            fit_success_other = True
            print(f"Other channels fit: SF = {a_o:.4f} * exp({b_o:.6f} * ecc) + {c_o:.4f}, R² = {r2_other:.4f}")
        except Exception as e:
            print(f"Could not fit other channels: {e}")
            fit_success_other = False
            r2_other = None

        # Generate smooth x values for plotting the fit (use ecc_fit range, excluding last bin)
        ecc_smooth = np.linspace(ecc_fit.min(), ecc_fit.max(), 100)

        # Plot SF preference vs eccentricity - Special channels (exclude last bin)
        ax_pref1.plot(ecc_fit, pref_sf_special_fit, 'o', color='tab:blue', markersize=8, label='Data')
        if fit_success_special:
            ax_pref1.plot(ecc_smooth, exp_func(ecc_smooth, *popt_special), '--',
                          color='tab:blue', linewidth=2, alpha=0.7,
                          label=f'Fit (R²={r2_special:.3f})')
        ax_pref1.set_xlabel(f"Eccentricity ({ecc_units})", fontsize=11)
        ax_pref1.set_ylabel("Preferred SF (cpp)", fontsize=11)
        ax_pref1.set_title(f"SF Preference vs Eccentricity\nSpecial Channels {special_channels}", fontsize=12)
        ax_pref1.legend(fontsize=8)
        ax_pref1.grid(True, alpha=0.3)
        ax_pref1.set_yscale('linear')

        # Plot SF preference vs eccentricity - Other channels (exclude last bin)
        ax_pref2.plot(ecc_fit, pref_sf_other_fit, 'o', color='tab:orange', markersize=8, label='Data')
        if fit_success_other:
            ax_pref2.plot(ecc_smooth, exp_func(ecc_smooth, *popt_other), '--',
                          color='tab:orange', linewidth=2, alpha=0.7,
                          label=f'Fit (R²={r2_other:.3f})')
        ax_pref2.set_xlabel(f"Eccentricity ({ecc_units})", fontsize=11)
        ax_pref2.set_ylabel("Preferred SF (cpp)", fontsize=11)
        ax_pref2.set_title("SF Preference vs Eccentricity\nOther Channels", fontsize=12)
        ax_pref2.legend(fontsize=8)
        ax_pref2.grid(True, alpha=0.3)
        ax_pref2.set_yscale('linear')
    else:
        # Fit for all channels (exclude last point)
        ecc_fit = ecc_centers[:-1]
        pref_sf_all_fit = pref_sf_all[:-1]
        try:
            popt_all, _ = curve_fit(exp_func, ecc_fit, pref_sf_all_fit,
                                     p0=[0.1, -0.01, 0.01], maxfev=5000)
            a_a, b_a, c_a = popt_all
            y_pred_all = exp_func(ecc_fit, *popt_all)
            r2_all = compute_r_squared(pref_sf_all_fit, y_pred_all)
            fit_success_all = True
            print(f"All channels fit: SF = {a_a:.4f} * exp({b_a:.6f} * ecc) + {c_a:.4f}, R² = {r2_all:.4f}")
        except Exception as e:
            print(f"Could not fit all channels: {e}")
            fit_success_all = False
            r2_all = None

        # Generate smooth x values for plotting the fit
        ecc_smooth = np.linspace(ecc_fit.min(), ecc_fit.max(), 100)

        # Plot SF preference vs eccentricity - All channels (exclude last bin)
        ax_pref1.plot(ecc_fit, pref_sf_all_fit, 'o', color='tab:green', markersize=8, label='Data')
        if fit_success_all:
            ax_pref1.plot(ecc_smooth, exp_func(ecc_smooth, *popt_all), '--',
                          color='tab:green', linewidth=2, alpha=0.7,
                          label=f'Fit (R²={r2_all:.3f})')
        ax_pref1.set_xlabel(f"Eccentricity ({ecc_units})", fontsize=11)
        ax_pref1.set_ylabel("Preferred SF (cpp)", fontsize=11)
        ax_pref1.set_title("SF Preference vs Eccentricity\nAll Channels", fontsize=12)
        ax_pref1.legend(fontsize=8)
        ax_pref1.grid(True, alpha=0.3)
        ax_pref1.set_yscale('linear')


    plt.tight_layout()
    output_path = os.path.join(
        plots_dir,
        f"sf_activation_curves_{layer_name}_{model_name}.png",
    )
    fig.savefig(output_path, dpi=150)
    # print(f"Saved: sf_activation_curves_{model_name}.png")

    # plt.show()
    """


def netStats_viz(model_name: str, manifest_csv: str, special_channels=None, layer_name="stages.0.0.dwconv", model_outpath=None, ori_sf_indices=None, sel_mode="vector_strength", sat_mode="uniform", plots_nickname="heather"):
    """
    Plot SF and orientation preferences for channels in first layer.

    Args:
        model_name: Name identifier for the model (used to locate grid_search directory)
        manifest_csv: Path to the manifest CSV matching the acts.npy files
        special_channels: List of channel indices to compare against others.
                         If None, plot all channels together.
        ori_sf_indices: List of SF indices (into the manifest frequency list) for which
                        to compute orientation maps. None means all SFs.
        sel_mode: How selectivity is computed — passed to compute_orientation_per_channel.
                  "vector_strength" (default): resultant / sum(acts), matches Lu et al. 2025.
                  "resultant": raw √(x²+y²), unnormalized.
        sat_mode: Saturation encoding for the unfolded sheet plot.
                  "uniform" (default): all pixels fully saturated, matches Lu et al. 2025 paper.
                  "selectivity": saturation encodes vector strength (desaturated = untuned).
    """
    plot_sf_special_channels(model_name=model_name, manifest_csv=manifest_csv, special_channels=special_channels, layer_name=layer_name, model_outpath=model_outpath, ori_sf_indices=ori_sf_indices, sel_mode=sel_mode, sat_mode=sat_mode, plots_nickname=plots_nickname)

if __name__ == "__main__":
    model_name = "ekuyzlow"
    model_outpath = "overridden"
    manifest_csv = "overridden"
    SPECIAL_CHANNELS = "overridden"
    plots_nickname = "overridden"
    netStats_viz(model_name=model_name, manifest_csv=manifest_csv, special_channels=SPECIAL_CHANNELS, layer_name="stages.0.0.dwconv", model_outpath=model_outpath, plots_nickname=plots_nickname)
