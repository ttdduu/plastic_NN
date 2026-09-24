import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.widgets import Slider
from matplotlib.patches import Circle


# Ensure we can import from src/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data.transforms.log_polar_viejo import LogPolarTransform, calculate_cartesian_radius  # noqa: E402


def build_canvas_with_square(img_size: int,
                             center_x: int,
                             center_y: int,
                             patch_size: int = 10) -> np.ndarray:
    canvas = np.zeros((img_size, img_size), dtype=np.float32)
    half = patch_size // 2
    y0 = max(0, center_y - half)
    x0 = max(0, center_x - half)
    y1 = min(img_size, center_y - half + patch_size)
    x1 = min(img_size, center_x - half + patch_size)
    if y0 < y1 and x0 < x1:
        canvas[y0:y1, x0:x1] = 1.0
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Place squares on Euclidean and fisheye grids; "
                    "visualise forward, inverse_fisheye, and round-trip."
    )
    parser.add_argument('--img-size', type=int, default=500,
                        help='Euclidean canvas size (square).')
    parser.add_argument('--patch-size', type=int, default=10)
    parser.add_argument('--center-x', type=int, default=300,
                        help='Initial X of Euclidean square (column).')
    parser.add_argument('--center-y', type=int, default=250,
                        help='Initial Y of Euclidean square (row).')

    # Log-polar params
    parser.add_argument('--rho0', type=float, default=80.0)
    parser.add_argument('--k', type=float, default=-0.56)
    parser.add_argument('--rmax', type=float, default=250.0,
                        help='Max Cartesian radius sampled (pixels).')
    parser.add_argument('--nrows', type=int, default=224,
                        help='Fisheye / log-polar resolution (rows & cols).')

    return parser.parse_args()


def main():
    args = parse_args()

    transform = LogPolarTransform(
        rho_0_px=args.rho0,
        k=args.k,
        R_max_cart_px=args.rmax,
        N_rows=args.nrows,
        N_cols=args.nrows,
    )

    # ---- mutable state ----
    state = {
        'img_size': args.img_size,
        'euc_x': args.center_x,
        'euc_y': args.center_y,
        'fish_x': args.nrows // 2 + 30,
        'fish_y': args.nrows // 2,
    }

    nrows = args.nrows
    img_size = state['img_size']

    # ---- initial computation ----
    euc_base = build_canvas_with_square(img_size, state['euc_x'],
                                        state['euc_y'], args.patch_size)
    fish_base = build_canvas_with_square(nrows, state['fish_x'],
                                         state['fish_y'], args.patch_size)

    with torch.no_grad():
        euc_t = torch.from_numpy(euc_base).unsqueeze(0).unsqueeze(0)
        fwd_t = transform(euc_t, output_size=nrows)
        rt_t  = transform.inverse_fisheye(fwd_t, output_size=img_size)

        fish_t = torch.from_numpy(fish_base).unsqueeze(0).unsqueeze(0)
        inv_t  = transform.inverse_fisheye(fish_t, output_size=img_size)

    fwd_np  = fwd_t.squeeze().cpu().numpy()
    rt_np   = rt_t.squeeze().cpu().numpy()
    inv_np  = inv_t.squeeze().cpu().numpy()

    # ---- figure layout ----
    fig = plt.figure(figsize=(24, 15))

    cx0, cx1, cx2 = 0.03, 0.35, 0.67
    aw = 0.26
    ah = 0.28

    y_top = 0.68
    y_bot = 0.35

    ax_fish_in = fig.add_axes([cx0, y_top, aw, ah])
    ax_inv     = fig.add_axes([cx1, y_top, aw, ah])
    ax_euc_in  = fig.add_axes([cx0, y_bot, aw, ah])
    ax_fwd     = fig.add_axes([cx1, y_bot, aw, ah])
    ax_round   = fig.add_axes([cx2, y_bot, aw, ah])
    ax_curve   = fig.add_axes([0.55, 0.03, 0.40, 0.24])

    imshow_kw = dict(cmap='RdBu_r', origin='upper', vmin=-1, vmax=1)

    def _dress(ax, size):
        mid = size / 2.0
        ax.axvline(mid, color='k', ls='--', lw=0.6, alpha=0.5)
        ax.axhline(mid, color='k', ls='--', lw=0.6, alpha=0.5)
        ax.set_xticks([]); ax.set_yticks([])

    # ---- circles ----
    # Euclidean axes get rho0 (lime) + R_max (cyan); fisheye axis gets only R_max.
    euc_ctr = (img_size - 1) / 2.0
    fish_ctr = (nrows - 1) / 2.0

    rho0_circ_euc = Circle((euc_ctr, euc_ctr), radius=transform.rho_0_px,
                            fill=False, color='lime', lw=1.2, alpha=0.9, zorder=3)
    rmax_circ_euc = Circle((euc_ctr, euc_ctr), radius=transform.R_max_cart_px,
                            fill=False, color='cyan', lw=1.0, ls='--', alpha=0.7, zorder=3)

    rho0_circ_inv = Circle((euc_ctr, euc_ctr), radius=transform.rho_0_px,
                            fill=False, color='lime', lw=1.2, alpha=0.9, zorder=3)
    rmax_circ_inv = Circle((euc_ctr, euc_ctr), radius=transform.R_max_cart_px,
                            fill=False, color='cyan', lw=1.0, ls='--', alpha=0.7, zorder=3)

    rmax_circ_fish = Circle((fish_ctr, fish_ctr), radius=transform.R_max_cart_px,
                             fill=False, color='cyan', lw=1.0, ls='--', alpha=0.7, zorder=3)

    # -- Top-left: Fisheye input --
    im_fish_in = ax_fish_in.imshow(fish_base, **imshow_kw)
    ax_fish_in.set_title(f'Fisheye input ({nrows}×{nrows})  —  click')
    _dress(ax_fish_in, nrows)
    ax_fish_in.add_patch(rmax_circ_fish)

    # -- Top-right: inverse result --
    im_inv = ax_inv.imshow(inv_np, **imshow_kw)
    ax_inv.set_title(f'inverse_fisheye → Euclidean ({img_size}×{img_size})')
    _dress(ax_inv, img_size)
    ax_inv.add_patch(rho0_circ_inv)
    ax_inv.add_patch(rmax_circ_inv)

    # -- Bottom-left: Euclidean input --
    im_euc_in = ax_euc_in.imshow(euc_base, **imshow_kw)
    ax_euc_in.set_title(f'Euclidean input ({img_size}×{img_size})  —  click')
    _dress(ax_euc_in, img_size)
    ax_euc_in.add_patch(rho0_circ_euc)
    ax_euc_in.add_patch(rmax_circ_euc)

    # -- Bottom-mid: forward fisheye --
    im_fwd = ax_fwd.imshow(fwd_np, **imshow_kw)
    ax_fwd.set_title(f'forward → fisheye ({nrows}×{nrows})')
    _dress(ax_fwd, nrows)

    # -- Bottom-right: round-trip --
    im_round = ax_round.imshow(rt_np, **imshow_kw)
    ax_round.set_title(f'inv(fwd(input)) ({img_size}×{img_size})')
    _dress(ax_round, img_size)

    # -- Mapping curve --
    def draw_mapping(ax, rho0, k, rmax, nr):
        ax.clear()
        j_vals = np.arange(1, int(nr) + 1)
        r_vals = calculate_cartesian_radius(j_vals, rho0, k, rmax, int(nr))
        ax.plot(r_vals, j_vals, lw=2, color='C0')
        ax.set_xlabel('r(j) [px]')
        ax.set_ylabel('LP row j')
        ax.set_title('Mapping r(j) vs j')
        ax.grid(True)
        ax.set_xlim(0, float(rmax))
        ax.set_ylim(0, int(nr))
        ax.invert_yaxis()
        try:
            ax.set_box_aspect(1)
        except Exception:
            pass

    draw_mapping(ax_curve, transform.rho_0_px, transform.k,
                 transform.R_max_cart_px, transform.N_rows)

    # ---- sliders ----
    sl_l, sl_w, sl_h, sl_gap = 0.06, 0.40, 0.025, 0.032
    sl_base = 0.03

    slider_imgsize_ax = fig.add_axes([sl_l, sl_base + 4*sl_gap, sl_w, sl_h])
    slider_rho0_ax    = fig.add_axes([sl_l, sl_base + 3*sl_gap, sl_w, sl_h])
    slider_k_ax       = fig.add_axes([sl_l, sl_base + 2*sl_gap, sl_w, sl_h])
    slider_rmax_ax    = fig.add_axes([sl_l, sl_base + 1*sl_gap, sl_w, sl_h])
    slider_nrows_ax   = fig.add_axes([sl_l, sl_base + 0*sl_gap, sl_w, sl_h])

    slider_imgsize = Slider(slider_imgsize_ax, 'img_size', 64, 2000,
                            valinit=args.img_size, valstep=1)
    slider_rho0    = Slider(slider_rho0_ax,  r'$\rho_0$', 1, 1000, valinit=args.rho0)
    slider_k       = Slider(slider_k_ax,     'k',         -3.0, 1.0, valinit=args.k)
    slider_rmax    = Slider(slider_rmax_ax,  'R_max',     1, 1000, valinit=args.rmax)
    slider_nrows   = Slider(slider_nrows_ax, 'N_rows',    16, 1000, valinit=args.nrows, valstep=1)

    # ---- helpers to update Euclidean-sized axes ----
    def _update_euc_axis(ax, im_obj, data, sz):
        """Set new data & rescale axis for a changed img_size."""
        im_obj.set_data(data)
        im_obj.set_extent([-0.5, sz - 0.5, sz - 0.5, -0.5])
        ax.set_xlim(-0.5, sz - 0.5)
        ax.set_ylim(sz - 0.5, -0.5)

    def _update_euc_circles(sz):
        """Move circle centres to the middle of the new Euclidean canvas."""
        ctr = (sz - 1) / 2.0
        for c in (rho0_circ_euc, rmax_circ_euc):
            c.center = (ctr, ctr)
        for c in (rho0_circ_inv, rmax_circ_inv):
            c.center = (ctr, ctr)

    # ---- recompute functions ----
    def recompute_euc():
        sz = state['img_size']
        cx, cy = state['euc_x'], state['euc_y']
        nr = int(slider_nrows.val)
        new_base = build_canvas_with_square(sz, cx, cy, args.patch_size)
        with torch.no_grad():
            t = torch.from_numpy(new_base).unsqueeze(0).unsqueeze(0)
            fwd = transform(t, output_size=nr)
            rt  = transform.inverse_fisheye(fwd, output_size=sz)
        _update_euc_axis(ax_euc_in, im_euc_in, new_base, sz)
        im_fwd.set_data(fwd.squeeze().cpu().numpy())
        _update_euc_axis(ax_round, im_round, rt.squeeze().cpu().numpy(), sz)
        # titles
        ax_euc_in.set_title(f'Euclidean input ({sz}×{sz})  —  click')
        ax_round.set_title(f'inv(fwd(input)) ({sz}×{sz})')

    def recompute_fish():
        sz = state['img_size']
        nr = int(slider_nrows.val)
        cx, cy = state['fish_x'], state['fish_y']
        new_fish = build_canvas_with_square(nr, cx, cy, args.patch_size)
        with torch.no_grad():
            t = torch.from_numpy(new_fish).unsqueeze(0).unsqueeze(0)
            inv = transform.inverse_fisheye(t, output_size=sz)
        im_fish_in.set_data(new_fish)
        ax_fish_in.set_xlim(-0.5, nr - 0.5)
        ax_fish_in.set_ylim(nr - 0.5, -0.5)
        _update_euc_axis(ax_inv, im_inv, inv.squeeze().cpu().numpy(), sz)
        # titles
        ax_fish_in.set_title(f'Fisheye input ({nr}×{nr})  —  click')
        ax_inv.set_title(f'inverse_fisheye → Euclidean ({sz}×{sz})')

        # diagnostics
        inv_np = inv.squeeze().cpu().numpy()
        ys, xs = np.where(np.abs(inv_np) > 0.01)
        if len(ys) > 0:
            dx = xs - sz / 2.0
            dy = ys - sz / 2.0
            radii = np.sqrt(dx**2 + dy**2)
            angles = np.degrees(np.arctan2(dy, dx))
            print(f"  Fish square at ({cx},{cy}) in {nr}×{nr}  →  "
                  f"inv to {sz}×{sz}: {len(ys)} px  "
                  f"|  r=[{radii.min():.1f}, {radii.max():.1f}]  "
                  f"|  θ=[{angles.min():.1f}°, {angles.max():.1f}°]  "
                  f"|  bbox {ys.max()-ys.min()+1}×{xs.max()-xs.min()+1}")

    # ---- click handler ----
    def on_click(event):
        if event.xdata is None:
            return
        if event.inaxes == ax_euc_in:
            sz = state['img_size']
            state['euc_x'] = max(0, min(sz - 1, int(round(event.xdata))))
            state['euc_y'] = max(0, min(sz - 1, int(round(event.ydata))))
            recompute_euc()
            fig.canvas.draw_idle()
        elif event.inaxes == ax_fish_in:
            nr = int(slider_nrows.val)
            state['fish_x'] = max(0, min(nr - 1, int(round(event.xdata))))
            state['fish_y'] = max(0, min(nr - 1, int(round(event.ydata))))
            recompute_fish()
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)

    # ---- slider handler ----
    def on_slider_change(val):
        nr = int(slider_nrows.val)
        new_sz = int(slider_imgsize.val)

        transform.update_params(
            rho_0_px=float(slider_rho0.val),
            k=float(slider_k.val),
            R_max_cart_px=float(slider_rmax.val),
            N_rows=nr,
            N_cols=nr,
        )
        # Invalidate cached grids
        transform._inv_fisheye_grid = None
        transform._inv_fisheye_cache_key = None
        transform._fisheye_grid = None
        transform._fisheye_cache_key = None

        # Update img_size state
        state['img_size'] = new_sz
        # Clamp Euclidean click position
        state['euc_x'] = min(state['euc_x'], new_sz - 1)
        state['euc_y'] = min(state['euc_y'], new_sz - 1)
        # Clamp fisheye click position
        state['fish_x'] = min(state['fish_x'], nr - 1)
        state['fish_y'] = min(state['fish_y'], nr - 1)

        # Update circles
        for c in (rho0_circ_euc, rho0_circ_inv):
            c.set_radius(float(slider_rho0.val))
        for c in (rmax_circ_fish, rmax_circ_euc, rmax_circ_inv):
            c.set_radius(float(slider_rmax.val))
        # Move fisheye circle centre
        fish_c = (nr - 1) / 2.0
        rmax_circ_fish.center = (fish_c, fish_c)
        # Move Euclidean circle centres
        _update_euc_circles(new_sz)

        # Update titles
        ax_fwd.set_title(f'forward → fisheye ({nr}×{nr})')

        draw_mapping(ax_curve, transform.rho_0_px, transform.k,
                     transform.R_max_cart_px, transform.N_rows)

        recompute_euc()
        recompute_fish()
        fig.canvas.draw_idle()

    slider_imgsize.on_changed(on_slider_change)
    slider_rho0.on_changed(on_slider_change)
    slider_k.on_changed(on_slider_change)
    slider_rmax.on_changed(on_slider_change)
    slider_nrows.on_changed(on_slider_change)

    plt.show()


if __name__ == '__main__':
    main()
