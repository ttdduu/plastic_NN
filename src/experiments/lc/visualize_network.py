import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Drawing Helpers ---

def draw_block(ax, center, width, height, label):
    """Draws a standard layer block."""
    x, y = center
    ax.add_patch(patches.FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.1,rounding_size=0.1",
        ec='black', fc='white', lw=1.5
    ))
    ax.text(x, y, label, ha='center', va='center', fontsize=12, fontweight='bold')

def draw_vertical_arrow(ax, start_y, end_y, x, label=None, label_color='gray', side='right'):
    """Draws a vertical arrow with a side label."""
    ax.add_patch(patches.FancyArrowPatch(
        (x, start_y), (x, end_y),
        arrowstyle='->', mutation_scale=20, lw=1.5, ec='black'
    ))
    if label:
        y_center = (start_y + end_y) / 2
        x_offset = 0.3 if side == 'right' else -0.3
        ha = 'left' if side == 'right' else 'right'
        # Ensure vertical alignment is centered for multi-line labels
        ax.text(x + x_offset, y_center, label, ha=ha, va='center', fontsize=11, color=label_color)

def draw_droppath_connection(ax, y_start, y_end, x_center, width, label):
    """Draws a right-angled residual connection, like in the ConvNeXt paper."""
    x_side = x_center - width / 2 - 0.8  # Position for the vertical line

    # Path for FancyArrowPatch
    path_points = [
        (x_center, y_start),  # 1. Start on main path
        (x_side, y_start),    # 2. Go left
        (x_side, y_end),      # 3. Go down
        (x_center, y_end),    # 4. Go right to merge (arrow will be here)
    ]
    path = patches.Path(path_points, [
        patches.Path.MOVETO, patches.Path.LINETO,
        patches.Path.LINETO, patches.Path.LINETO,
    ])

    pp = patches.FancyArrowPatch(
        path=path, arrowstyle='->', mutation_scale=20,
        lw=1.5, ec='black', fill=False, zorder=9
    )
    ax.add_patch(pp)
    
    # Add a '+' sign inside a circle at the merge point
    ax.add_patch(patches.Circle((x_center, y_end), 0.2, fc='white', ec='black', lw=1.5, zorder=10))
    ax.text(x_center, y_end, '+', ha='center', va='center', fontsize=14, fontweight='bold', zorder=11)

    # Add the 'DropPath' label
    ax.text(x_side - 0.1, (y_start + y_end) / 2, label, ha='right', va='center', fontsize=12, color='blue')

# --- Main Plotting Functions ---

def draw_flow(ax, title, dw_label, has_layernorm, has_droppath):
    """Draws one of the two vertical block diagrams."""
    ax.set_title(title, fontsize=16, pad=20)
    
    # Layout constants
    x_center = 2.5
    block_w, block_h = 4, 1.2 # Increased height for more text
    y_start = 10
    gap = 1.0

    # Block positions
    y1 = y_start - block_h
    y2 = y1 - gap - block_h
    y3 = y2 - gap - block_h
    
    # --- Draw Main Blocks and Connecting Arrows ---
    y_input_point = y1 + block_h/2
    draw_vertical_arrow(ax, y_start + 1, y_input_point, x_center, label="C x H x W", side='left')
    
    draw_block(ax, (x_center, y1), block_w, block_h, dw_label)
    
    ln_label = "LayerNorm\n" if has_layernorm else ""
    draw_vertical_arrow(ax, y1 - block_h/2, y2 + block_h/2, x_center, label=f"{ln_label}C x H x W")

    draw_block(ax, (x_center, y2), block_w, block_h, "Pointwise Conv\n(1x1, C → 4C)")

    draw_vertical_arrow(ax, y2 - block_h/2, y3 + block_h/2, x_center, label="GELU + GRN\n4C x H x W", label_color='green')

    draw_block(ax, (x_center, y3), block_w, block_h, "Pointwise Conv\n(1x1, 4C → C)")
    
    # --- Handle Output and Residual Path ---
    y_output_from_blocks = y3 - block_h/2
    
    if has_droppath:
        y_merge_point = y_output_from_blocks - gap
        
        # Arrow from last block to the merge point
        draw_vertical_arrow(ax, y_output_from_blocks, y_merge_point, x_center, label="C x H x W")

        # Draw the DropPath connection from input to merge point
        draw_droppath_connection(ax, y_input_point, y_merge_point, x_center, block_w, "DropPath")
        
        # Final arrow emerging from the merge point
        y_final_output = y_merge_point - gap
        draw_vertical_arrow(ax, y_merge_point, y_final_output, x_center)
        
        ax.set_ylim(y_final_output - 0.5, y_start + 1.5)
    else:
        # For the custom block, flow ends at the last block.
        ax.set_ylim(y_output_from_blocks - 0.5, y_start + 1.5)

    ax.set_xlim(0, 5)
    ax.axis('off')

def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
    fig.suptitle("ConvNeXt Block Architectures", fontsize=20, y=0.98)
    
    # Panel 1: Original Block
    draw_flow(ax1, 
              title="Original ConvNeXt V2 Block", 
              dw_label="DW Conv (7x7)\n(Params: 49 x C)", 
              has_layernorm=True, 
              has_droppath=True)
              
    # Panel 2: Custom Block
    draw_flow(ax2, 
              title="Custom Locally Connected Block", 
              dw_label="LC DW Conv (7x7)\n(Params: 49 x H x W x C)", 
              has_layernorm=False, 
              has_droppath=False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = "block_comparison.png"
    plt.show()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Diagram saved to {output_path}")

if __name__ == '__main__':
    main()
