import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
from matplotlib.patches import Rectangle
import torch
from src.data.transforms.fisheye import InverseFisheyeTransform

# Hardcoded configuration
INPUT_SIZE = 111+10  # input-space resolution (square)
# Fisheye parameters (same convention as in find_relevant_neurons)
FE_C = 1.0
FE_K = -7.0
FE_RFOV = 20.0

# Layers, RF sizes (in input-space pixels), and feature map sizes (H, W)
LAYER_NAMES = [
    "stages.0.0.dwconv",
    "stages.0.1.dwconv",
    "stages.1.0.dwconv",
    "stages.1.1.dwconv",
    "stages.2.0.dwconv",
    "stages.2.1.dwconv",
    "stages.2.2.dwconv",
    "stages.2.3.dwconv",
    "stages.2.4.dwconv",
    "stages.2.5.dwconv",
    "stages.3.0.dwconv",
    "stages.3.1.dwconv",
]

# Example RF sizes (edit to match your model)
LAYER_RF_SIZE = {
    "stages.0.0.dwconv":10, 
    "stages.0.1.dwconv":16,
    "stages.1.0.dwconv":29,
    "stages.1.1.dwconv":41,
    "stages.2.0.dwconv":67,
    "stages.2.1.dwconv":91,
    "stages.2.2.dwconv":115,
    "stages.2.3.dwconv":139,
    "stages.2.4.dwconv":163,
    "stages.2.5.dwconv":187,
    "stages.3.0.dwconv":239,
    "stages.3.1.dwconv":287
}                        

# Example feature map sizes (edit to match your model)
LAYER_FMAP_HW = {
    "stages.0.0.dwconv": (INPUT_SIZE, INPUT_SIZE),
    "stages.0.1.dwconv": (INPUT_SIZE, INPUT_SIZE),
    "stages.1.0.dwconv": (INPUT_SIZE // 2, INPUT_SIZE // 2),
    "stages.1.1.dwconv": (INPUT_SIZE // 2, INPUT_SIZE // 2),
    "stages.2.0.dwconv": (INPUT_SIZE // 4, INPUT_SIZE // 4),
    "stages.2.1.dwconv": (INPUT_SIZE // 4, INPUT_SIZE // 4),
    "stages.2.2.dwconv": (INPUT_SIZE // 4, INPUT_SIZE // 4),
    "stages.2.3.dwconv": (INPUT_SIZE // 4, INPUT_SIZE // 4),
    "stages.2.4.dwconv": (INPUT_SIZE // 4, INPUT_SIZE // 4),
    "stages.2.5.dwconv": (INPUT_SIZE // 4, INPUT_SIZE // 4),
    "stages.3.0.dwconv": (INPUT_SIZE // 8, INPUT_SIZE // 8),
    "stages.3.1.dwconv": (INPUT_SIZE // 8, INPUT_SIZE // 8),
}


def build_square_canvas(img_size: int, center_y: int, center_x: int, side: int) -> np.ndarray:
    side = max(1, int(round(side)))
    canvas = np.zeros((img_size, img_size), dtype=np.float32)
    half = side // 2
    y0 = max(0, int(center_y - half))
    y1 = min(img_size, int(center_y - half + side))
    x0 = max(0, int(center_x - half))
    x1 = min(img_size, int(center_x - half + side))
    if y1 > y0 and x1 > x0:
        canvas[y0:y1, x0:x1] = 1.0
    return canvas


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inv_transform = InverseFisheyeTransform(C=FE_C, K=FE_K, rfov=FE_RFOV).to(device)

    # Initial state
    state = {
        "layer": LAYER_NAMES[0],
        "center_y": INPUT_SIZE // 2,
        "center_x": INPUT_SIZE // 2,
        "out_size": INPUT_SIZE * 2,
    }

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.7, 1.2, 1.2], wspace=0.25)

    ax_layer = fig.add_subplot(gs[0, 0])
    ax_in = fig.add_subplot(gs[0, 1])
    ax_out = fig.add_subplot(gs[0, 2])

    # Layer selector
    ax_layer.set_title("Layer")
    radio = RadioButtons(ax_layer, LAYER_NAMES, active=0)

    # Input-space axis
    ax_in.set_title("Input space")
    ax_in.set_aspect('equal')
    ax_in.set_xlim(-0.5, INPUT_SIZE - 0.5)
    ax_in.set_ylim(INPUT_SIZE - 0.5, -0.5)
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    in_im = ax_in.imshow(np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.float32),
                         cmap='gray', vmin=0.0, vmax=1.0)

    # Transformed axis
    ax_out.set_title(f"Inverse fisheye transform ({state['out_size']}x{state['out_size']})")
    ax_out.set_aspect('equal')
    ax_out.set_xticks([])
    ax_out.set_yticks([])
    ax_out.set_facecolor('black')
    ax_out.set_xlim(-0.5, state['out_size'] - 0.5)
    ax_out.set_ylim(state['out_size'] - 0.5, -0.5)
    out_im = ax_out.imshow(np.zeros((state['out_size'], state['out_size']), dtype=np.float32),
                           cmap='gray', vmin=0.0, vmax=1.0)
    out_im.set_extent((-0.5, state['out_size'] - 0.5, state['out_size'] - 0.5, -0.5))

    # Centered border-only squares
    # Input axis: 111x111 centered
    in_side = 111
    in_mid = (INPUT_SIZE - 1) / 2.0
    in_x0 = in_mid - in_side / 2.0
    in_y0 = in_mid - in_side / 2.0
    in_square_patch = Rectangle((in_x0, in_y0), in_side, in_side,
                                fill=False, edgecolor='lime', linewidth=1.5)
    ax_in.add_patch(in_square_patch)

    # Output axis: 224x224 centered (updates if out_size changes)
    out_side = 224
    out_mid = (state['out_size'] - 1) / 2.0
    out_x0 = out_mid - out_side / 2.0
    out_y0 = out_mid - out_side / 2.0
    out_square_patch = Rectangle((out_x0, out_y0), out_side, out_side,
                                 fill=False, edgecolor='orange', linewidth=1.5)
    ax_out.add_patch(out_square_patch)

    txt_info = ax_out.text(0.02, 0.98, "", transform=ax_out.transAxes,
                           va='top', ha='left', color='w', fontsize=9,
                           bbox=dict(facecolor='black', alpha=0.4, edgecolor='none'))

    def current_rf_size() -> int:
        return int(LAYER_RF_SIZE.get(state["layer"], 7))

    def current_fmap_hw():
        return LAYER_FMAP_HW.get(state["layer"], (INPUT_SIZE, INPUT_SIZE))

    def render():
        side = current_rf_size()
        canvas = build_square_canvas(INPUT_SIZE, state["center_y"], state["center_x"], side)
        in_im.set_data(canvas)
        with torch.no_grad():
            t = torch.from_numpy(canvas).to(device)
            # Apply inverse fisheye (output size kept equal to INPUT_SIZE)
            y = inv_transform(t, out_hw=(state['out_size'], state['out_size']))
            y_np = y.detach().cpu().numpy()
        out_im.set_data(y_np)
        out_im.set_extent((-0.5, state['out_size'] - 0.5, state['out_size'] - 0.5, -0.5))
        nz = int(np.count_nonzero(y_np > 1e-6))
        fmap_h, fmap_w = current_fmap_hw()
        txt_info.set_text(
            f"Layer: {state['layer']}\nRF side: {side}px\nFMAP: {fmap_h}x{fmap_w}\nNonzero (invFE): {nz}"
        )
        fig.canvas.draw_idle()

    # Out size control
    out_size_ax = fig.add_axes([0.82, 0.90, 0.12, 0.06])  # [left, bottom, width, height]
    out_size_box = RadioButtons(out_size_ax, [str(state['out_size'])], active=0)
    # Replace RadioButtons label with a TextBox-like behavior by listening to clicks is awkward;
    # create a proper TextBox instead:
    out_size_ax.clear()
    from matplotlib.widgets import TextBox  # local import to avoid clutter at top
    out_size_box = TextBox(out_size_ax, 'Out:', initial=str(state['out_size']))

    def on_out_size_change(text):
        try:
            new_size = int(float(text))
        except Exception:
            return
        new_size = max(1, new_size)
        state['out_size'] = new_size
        ax_out.set_title(f"Inverse fisheye transform ({new_size}x{new_size})")
        ax_out.set_xlim(-0.5, new_size - 0.5)
        ax_out.set_ylim(new_size - 0.5, -0.5)
        # Resize current image buffer to avoid shape mismatch artifacts
        out_im.set_data(np.zeros((new_size, new_size), dtype=np.float32))
        out_im.set_extent((-0.5, new_size - 0.5, new_size - 0.5, -0.5))
        # Recenter the 224x224 border on the new canvas size
        out_mid_local = (new_size - 1) / 2.0
        new_x0 = out_mid_local - out_side / 2.0
        new_y0 = out_mid_local - out_side / 2.0
        out_square_patch.set_xy((new_x0, new_y0))
        out_square_patch.set_width(out_side)
        out_square_patch.set_height(out_side)
        render()

    out_size_box.on_submit(on_out_size_change)

    def on_radio(label):
        state["layer"] = label
        render()

    def on_click(event):
        if event.inaxes != ax_in:
            return
        if event.xdata is None or event.ydata is None:
            return
        cx = int(round(event.xdata))
        cy = int(round(event.ydata))
        cx = max(0, min(INPUT_SIZE - 1, cx))
        cy = max(0, min(INPUT_SIZE - 1, cy))
        state["center_x"] = cx
        state["center_y"] = cy
        render()

    radio.on_clicked(on_radio)
    fig.canvas.mpl_connect('button_press_event', on_click)

    render()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

