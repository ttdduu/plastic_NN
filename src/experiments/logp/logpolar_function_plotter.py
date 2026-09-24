import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Nomenclature:
# j: The row index in the log-polar image.
# r: The corresponding radius in the original Cartesian image (in pixels).
# rho_0: The pivot radius where the mapping has unit magnification (dj/dr = 1).
# k: The magnification strength exponent. k=-1 corresponds to a pure log transform.
# r_max: The maximum Cartesian radius sampled (maps to j=j_max).
# j_max: The total number of rows in the log-polar image.

def calculate_cartesian_radius(j, rho_0, k, r_max, j_max):
    """
    Calculates the Cartesian radius r(j) using a generalized logarithmic transform.
    
    This transform is defined by its derivative: dj/dr = (r / rho_0)^k.
    It allows for independent control over the pivot point (rho_0) and the
    magnification strength (k).

    Args:
        j (np.ndarray): Array of log-polar row indices.
        rho_0 (float): The pivot radius (in pixels).
        k (float): The magnification strength exponent. k=-1 is the classic log transform.
        r_max (float): The maximum sampled Cartesian radius (in pixels).
        j_max (int): The total number of rows in the log-polar image.

    Returns:
        np.ndarray: The corresponding Cartesian radii.
    """
    # Handle edge cases for rho_0 or if r_max is not set properly
    if rho_0 <= 0 or r_max <= 0:
        return np.zeros_like(j, dtype=float)

    # The mapping is derived by integrating dj/dr and solving for r(j).
    # This leads to a special case for k=-1 (the pure log transform).
    if np.isclose(k, -1.0):
        # Classic log transform: r(j) = r_max * exp((j - j_max) / rho_0)
        # We add a small epsilon to rho_0 to avoid division by zero if it's ever 0.
        r = r_max * np.exp((j - j_max) / (rho_0 + 1e-9))
        # Shift to ensure center is always sampled: subtract r(1) from all radii
        r_at_j1 = r_max * np.exp((1.0 - j_max) / (rho_0 + 1e-9))
        r = r - r_at_j1
        r = np.maximum(r, 0.0)
    else:
        # Generalized power-law transform
        k_plus_1 = k + 1.0
        # Formula: r(j) = [ r_max^(k+1) + (j - j_max) * rho_0^k * (k+1) ] ^ (1 / (k+1))
        
        # We must handle the base of the power carefully to avoid taking roots of negative numbers.
        # This can happen if j is small and k is such that the term becomes negative.
        base = np.power(r_max, k_plus_1) + (j - j_max) * np.power(rho_0, k) * k_plus_1
        
        # For any invalid regions (e.g., trying to sample below 0 radius), we'll clamp to 0.
        r = np.power(np.maximum(0, base), 1.0 / k_plus_1)
        # Shift to ensure center is always sampled: subtract r(1) from all radii
        base_at_j1 = np.power(r_max, k_plus_1) + (1.0 - j_max) * np.power(rho_0, k) * k_plus_1
        r_at_j1 = np.power(np.maximum(0, base_at_j1), 1.0 / k_plus_1)
        r = r - r_at_j1
        r = np.maximum(r, 0.0)

    return r

# --- Main script ---
if __name__ == "__main__":
    # Initial parameters
    init_rho_0 = 40.0
    init_k = -1.0  # Start with the classic log transform
    init_r_max = 128.0
    init_j_max = 256

    # Create the figure and axes for the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(left=0.1, bottom=0.4)

    # Generate initial data
    j_values = np.arange(1, init_j_max + 1)
    r_values = calculate_cartesian_radius(j_values, init_rho_0, init_k, init_r_max, init_j_max)

    # Create the plot
    line, = ax.plot(j_values, r_values, lw=2)
    ax.set_xlabel("Log-Polar Row Index (j)")
    ax.set_ylabel("Cartesian Radius r(j) [pixels]")
    ax.set_title("Generalized Log-Polar Transform")
    ax.grid(True)
    ax.set_xlim(0, init_j_max)
    ax.set_ylim(0, init_r_max * 1.1)

    # Add a horizontal line at r = rho_0, the pivot point
    rho0_line = ax.axhline(y=init_rho_0, color='r', linestyle='--', label=f'r = ρ₀ (pivot)')
    ax.legend()
    
    # --- Create Sliders for interactive control ---
    
    # Define slider axes positions
    ax_rho0 = plt.axes([0.25, 0.25, 0.65, 0.03])
    ax_k = plt.axes([0.25, 0.20, 0.65, 0.03])
    ax_rmax = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_jmax = plt.axes([0.25, 0.10, 0.65, 0.03])

    # Create Slider objects
    rho0_slider = Slider(
        ax=ax_rho0, label='Pivot ρ₀', valmin=1, valmax=200, valinit=init_rho_0)
    k_slider = Slider(
        ax=ax_k, label='Strength k', valmin=-3, valmax=1, valinit=init_k)
    rmax_slider = Slider(
        ax=ax_rmax, label='r_max', valmin=1, valmax=1024, valinit=init_r_max)
    jmax_slider = Slider(
        ax=ax_jmax, label='j_max (rows)', valmin=2, valmax=1024, valinit=init_j_max, valstep=1)

    # --- Update function that runs when a slider is changed ---
    def update(val):
        # Get current slider values
        rho_0 = rho0_slider.val
        k = k_slider.val
        r_max = rmax_slider.val
        j_max = int(jmax_slider.val)

        # Recalculate data with new parameters
        j_new = np.arange(1, j_max + 1)
        r_new = calculate_cartesian_radius(j_new, rho_0, k, r_max, j_max)
        
        # Update the plot data
        line.set_xdata(j_new)
        line.set_ydata(r_new)
        
        # Update the pivot line
        rho0_line.set_ydata([rho_0, rho_0])
        
        # Update axes limits
        ax.set_xlim(0, j_max)
        ax.set_ylim(0, r_max * 1.1)
        
        # Redraw the canvas
        fig.canvas.draw_idle()

    # Register the update function with each slider
    rho0_slider.on_changed(update)
    k_slider.on_changed(update)
    rmax_slider.on_changed(update)
    jmax_slider.on_changed(update)

    plt.show() 