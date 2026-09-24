import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image
from .polar_plot import rho_theta_to_cartesian
from src.data.transforms.log_polar_viejo import LogPolarTransform, calculate_cartesian_radius, rows_needed_to_cover_center

class TransformPlotter:
    def __init__(self, image_path, second_image_path=None):
        # Load and preprocess image
        self.original_img = np.array(Image.open(image_path))
        # Ensure image_size is set for later use
        if self.original_img.ndim == 3: # Color image (H, W, C)
            self.image_size = self.original_img.shape[0] # Assuming H is the primary size reference
        elif self.original_img.ndim == 2: # Grayscale image (H, W)
            self.image_size = self.original_img.shape[0]
        else:
            raise ValueError(f"Unexpected image dimensions: {self.original_img.shape}")

        # Resize image to 224x224 if needed
        if self.original_img.shape[0] != 224 or self.original_img.shape[1] != 224:
            # Convert to PIL Image for resizing
            pil_img = Image.fromarray(self.original_img)
            # Use LANCZOS resampling for high quality resizing
            pil_img = pil_img.resize((224, 224), Image.Resampling.LANCZOS)
            # Convert back to numpy array
            self.original_img = np.array(pil_img)
            self.image_size = 224

        img_height = self.original_img.shape[0]

        # Optionally load a second image
        self.second_img = None
        if second_image_path is not None:
            self.second_img = np.array(Image.open(second_image_path))
            if self.second_img.ndim == 3 and (self.second_img.shape[0] != 224 or self.second_img.shape[1] != 224):
                pil_second = Image.fromarray(self.second_img)
                self.second_img = np.array(pil_second.resize((224, 224), Image.Resampling.LANCZOS))
            elif self.second_img.ndim == 2 and (self.second_img.shape[0] != 224 or self.second_img.shape[1] != 224):
                pil_second = Image.fromarray(self.second_img)
                self.second_img = np.array(pil_second.resize((224, 224), Image.Resampling.LANCZOS))

        # Create figure with a 2-row, 4-column layout for plots, plus space for sliders
        self.fig = plt.figure(figsize=(22, 12)) # Slightly wider to fit the extra column

        # Create a gridspec: 2 rows for plots, 4 for sliders. 4 columns.
        gs = self.fig.add_gridspec(2 + 4, 4, height_ratios=[1, 1, 0.05, 0.05, 0.05, 0.05], width_ratios=[1, 1, 1, 1], hspace=0.3, wspace=0.3)

        # Main 2x2 image plots in the first two columns
        self.axs = [
            [self.fig.add_subplot(gs[0, 0]), self.fig.add_subplot(gs[0, 1])],
            [self.fig.add_subplot(gs[1, 0]), self.fig.add_subplot(gs[1, 1])]
        ]

        # Mapping function plot in the third column (single row) to keep it square
        self.ax_mapping_function = self.fig.add_subplot(gs[0, 2])

        # Extra column (fourth) for optional second image: original (top) and log-polar (bottom)
        self.ax_second_orig = self.fig.add_subplot(gs[0, 3])
        self.ax_second_logpolar = self.fig.add_subplot(gs[1, 3])

        # Axes for sliders, spanning all columns at the bottom
        self.slider_rho0_ax = self.fig.add_subplot(gs[2, :])
        self.slider_k_ax = self.fig.add_subplot(gs[3, :])
        self.slider_R_max_cart_px_ax = self.fig.add_subplot(gs[4, :])
        self.slider_N_rows_ax = self.fig.add_subplot(gs[5, :])

        # Initial values for sliders
        initial_rho0_px = 80.0
        initial_k = -1.0 # Classic log transform
        initial_R_max_cart_px = self.image_size // 2
        initial_N_rows = self.image_size

        # Instantiate the transform object that will be updated
        self.log_polar_transform = LogPolarTransform(
            rho_0_px=initial_rho0_px,
            k=initial_k,
            R_max_cart_px=initial_R_max_cart_px,
            N_rows=initial_N_rows,
            N_cols=initial_N_rows # Keep it square
        )

        # Create sliders
        self.slider_rho0 = Slider(
            ax=self.slider_rho0_ax,
            label='Pivot ρ₀',
            valmin=1,
            valmax=self.image_size,
            valinit=initial_rho0_px
        )
        self.slider_k = Slider(
            ax=self.slider_k_ax,
            label='Strength k',
            valmin=-3.0,
            valmax=1.0,
            valinit=initial_k
        )
        self.slider_R_max_cart_px = Slider(
            ax=self.slider_R_max_cart_px_ax,
            label='R_max_px (Sample Extent)',
            valmin=1,
            valmax=self.image_size,
            valinit=initial_R_max_cart_px
        )
        self.slider_N_rows = Slider(
            ax=self.slider_N_rows_ax,
            label='N_rows (LP Height)',
            valmin=16,
            valmax=self.image_size * 10,
            valinit=initial_N_rows,
            valstep=1
        )

        # Connect sliders to update function
        self.slider_rho0.on_changed(self._update_plots)
        self.slider_k.on_changed(self._update_plots)
        self.slider_R_max_cart_px.on_changed(self._update_plots)
        self.slider_N_rows.on_changed(self._update_plots)

        # Removed: plt.subplots_adjust(wspace=0.3, hspace=0.3) as hspace is in gridspec
        # self.fig.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to prevent overlap, may need tuning

        # Store parameters that need to be accessed
        self.scotoma_radius = None
        self.input_tensor = None
        # Store current slider values
        self.current_rho0_px = initial_rho0_px
        self.current_k = initial_k
        self.current_R_max_cart_px = initial_R_max_cart_px
        self.current_N_rows = initial_N_rows

    def apply_scotoma(self, input_tensor, radius):
        """Apply scotoma to input tensor"""
        H, W = input_tensor.shape[-2:]
        center = (W // 2, H // 2)

        y, x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=input_tensor.device),
            torch.arange(W, dtype=torch.float32, device=input_tensor.device),
            indexing='ij'
        )
        distance = torch.sqrt((x - center[0])**2 + (y - center[1])**2)

        sharpness = 6
        radius_pixels = (radius/100) * W
        scotoma = torch.sigmoid(sharpness*(distance - radius_pixels))

        mask = scotoma.unsqueeze(0).unsqueeze(0)
        if input_tensor.shape[1] == 3:
            # Create mask that only affects non-red channels
            mask = torch.stack([
                torch.ones_like(scotoma),  # Red channel unaffected
                scotoma,  # Apply scotoma to green channel
                scotoma   # Apply scotoma to blue channel
            ]).unsqueeze(0)

        return input_tensor * mask

    def _update_plots(self, event=None): # event is passed by on_changed but not always used
        # Update current values from sliders
        self.current_rho0_px = self.slider_rho0.val
        self.current_k = self.slider_k.val
        self.current_R_max_cart_px = self.slider_R_max_cart_px.val
        self.current_N_rows = int(self.slider_N_rows.val)

        # Clear all plot axes
        for row in self.axs:
            for ax in row:
                ax.clear()
        self.ax_mapping_function.clear()

        # Replot with new slider values
        self._plot_all(
            self.scotoma_radius,
            rho0_px_val=self.current_rho0_px,
            k_val=self.current_k,
            R_max_cart_px_val=self.current_R_max_cart_px,
            N_rows_val=self.current_N_rows
        )
        self.fig.canvas.draw_idle()

    def _plot_all(self, scotoma_radius, rho0_px_val, k_val, R_max_cart_px_val, N_rows_val):
        # 1. Original image
        self.axs[0][0].imshow(self.original_img)
        self.axs[0][0].set_title('Original Image')
        self.axs[0][0].axis('off')

        # 2. Apply scotoma
        #scotoma_img = self.apply_scotoma(self.input_tensor, scotoma_radius)
        scotoma_img = self.input_tensor
        scotoma_np = scotoma_img[0].permute(1, 2, 0).detach().numpy()
        self.axs[0][1].imshow(scotoma_np)
        self.axs[0][1].set_title(f'Scotoma (radius={scotoma_radius}%)')
        self.axs[0][1].axis('off')

        # 3. Convert scotoma image to log-polar with current parameters
        _current_N_cols = int(N_rows_val) # Keep LP image square for now

        # Update the transform object with the new slider values
        self.log_polar_transform.update_params(
            rho_0_px=rho0_px_val,
            k=k_val,
            R_max_cart_px=R_max_cart_px_val,
            N_rows=int(N_rows_val),
            N_cols=_current_N_cols
        )

        log_polar_img = self.log_polar_transform(scotoma_img)
        log_polar_np = log_polar_img[0].permute(1, 2, 0).detach().numpy()
        self.axs[1][0].imshow(log_polar_np)
        self.axs[1][0].set_title(f'Log-polar (ρ₀={rho0_px_val:.1f}, k={k_val:.2f}, R_max={R_max_cart_px_val:.0f}px)')

        # --- Y-axis labels ---
        # The y-axis now simply corresponds to the row index j.
        # We can label the top, bottom, and center for context.
        ax_logpolar = self.axs[1][0]
        num_rows_lp = log_polar_np.shape[0]

        if num_rows_lp > 0:
            tick_positions = [0, num_rows_lp // 2, num_rows_lp - 1]
            tick_labels = ['j=1', f'j={num_rows_lp // 2 + 1}', f'j={num_rows_lp}']
            ax_logpolar.set_yticks(tick_positions)
            ax_logpolar.set_yticklabels(tick_labels)

        ax_logpolar.set_ylabel("Log-Polar Row Index (j)")
        self.axs[1][0].set_xticks([])

        # 4. Convert back to Cartesian
        cartesian = rho_theta_to_cartesian(Image.fromarray((log_polar_np * 255).astype(np.uint8)))
        cart_np = cartesian.permute(1, 2, 0).detach().numpy()
        self.axs[1][1].imshow(cart_np)
        self.axs[1][1].set_title('Back to Cartesian')
        self.axs[1][1].axis('off')

        # 4b. If second image present, show it and its log-polar
        if self.second_input_tensor is not None:
            # Original second image
            if self.second_img.ndim == 3:
                self.ax_second_orig.imshow(self.second_img)
            elif self.second_img.ndim == 2:
                self.ax_second_orig.imshow(self.second_img, cmap='gray')
            self.ax_second_orig.set_title('Second Image')
            self.ax_second_orig.axis('off')

            # Log-polar second image
            log_polar_img_2 = self.log_polar_transform(self.second_input_tensor)
            log_polar_np_2 = log_polar_img_2[0].permute(1, 2, 0).detach().numpy()
            self.ax_second_logpolar.imshow(log_polar_np_2)
            self.ax_second_logpolar.set_title('Second: Log-polar')
            self.ax_second_logpolar.set_xticks([])
            self.ax_second_logpolar.set_yticks([])

        # 5. Add the mapping function plot to the right-hand column
        ax_map = self.ax_mapping_function
        try:
            ax_map.set_aspect('equal', adjustable='box')
        except Exception:
            pass

        # Generate data for the plot
        j_values = np.arange(1, int(N_rows_val) + 1)
        r_values = calculate_cartesian_radius(j_values, rho0_px_val, k_val, R_max_cart_px_val, N_rows_val)

        # Plot our transform
        ax_map.plot(r_values, j_values, lw=2, label="New Transform")

        # Add log2 transform for comparison
        r_values_log2 = np.linspace(1, R_max_cart_px_val, int(N_rows_val))
        j_values_log2 = np.log2(r_values_log2) * (N_rows_val / np.log2(R_max_cart_px_val))
        ax_map.plot(r_values_log2, j_values_log2, lw=2, linestyle='--', color='red', label="log2 Transform")

        # Add a VERTICAL line at r = rho_0
        ax_map.axvline(x=rho0_px_val, color='r', linestyle='--', label=f'r = ρ₀ (pivot)')

        # Set labels and title
        ax_map.set_ylabel("Log-Polar Row Index (j)")
        ax_map.set_xlabel("Cartesian Radius r(j) [pixels]")
        ax_map.set_title("Mapping: r(j) vs j")
        ax_map.legend(loc='lower right')
        ax_map.grid(True)

        # Set axis limits based on current parameters
        ax_map.set_ylim(0, N_rows_val)
        ax_map.set_xlim(0, R_max_cart_px_val * 1.1)

        # Invert y-axis to match log-polar image (row 0 at top)
        ax_map.invert_yaxis()

        # --- Display Parameters on the Plot ---
        # Compute and display recommended N_rows to include radii from 1px
        recommended_rows = rows_needed_to_cover_center(rho0_px_val, k_val, R_max_cart_px_val, r_min_target=1.0)
        recommendation_note = ""
        if recommended_rows > N_rows_val:
            recommendation_note = f"\n  Recommended N_rows ≥ {recommended_rows} to include r≥1px"
        else:
            recommendation_note = f"\n  Coverage OK for r≥1px (need ≤ {recommended_rows})"
        # Console message so it's visible even if the text box is overlooked
        print(f"[TransformPlotter] ρ₀={rho0_px_val:.2f}, k={k_val:.2f}, R_max={R_max_cart_px_val:.0f}px -> recommended N_rows ≥ {recommended_rows} (current={int(N_rows_val)}) to include r≥1px")
        # Reinforce in the log-polar image title
        lp_title = f"Log-polar (ρ₀={rho0_px_val:.1f}, k={k_val:.2f}, R_max={R_max_cart_px_val:.0f}px)"
        if recommended_rows > N_rows_val:
            lp_title += f" | need N_rows≥{recommended_rows}"
        else:
            lp_title += f" | OK (≤{recommended_rows})"
        self.axs[1][0].set_title(lp_title)
        param_text = (
            f"Parameters:\n"
            f"  Pivot ρ₀ = {rho0_px_val:.2f}\n"
            f"  Strength k = {k_val:.2f}\n"
            f"  R_max_px = {R_max_cart_px_val:.2f}\n"
            f"  N_rows = {int(N_rows_val)}" + recommendation_note
        )

        # Position the text box in the plot
        # 'transform=ax_map.transAxes' makes coordinates relative to the axes (0,0 is bottom-left, 1,1 is top-right)
        ax_map.text(0.95, 0.95, param_text, transform=ax_map.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    def plot_transforms(self, scotoma_radius=30):
        # Convert original image to tensor once
        if self.original_img.ndim == 3:
            if self.original_img.shape[2] == 3: # RGB
                img_tensor = torch.from_numpy(self.original_img).permute(2,0,1).float()
            elif self.original_img.shape[2] == 4: # RGBA
                # Take only RGB channels, ignore alpha
                img_tensor = torch.from_numpy(self.original_img[:,:,:3]).permute(2,0,1).float()
            else:
                raise ValueError(f"Unsupported number of channels: {self.original_img.shape[2]}")
        elif self.original_img.ndim == 2: # Grayscale
            img_tensor = torch.from_numpy(self.original_img).unsqueeze(0).float()
        else:
            raise ValueError(f"Unsupported image dimensions: {self.original_img.ndim}")

        if img_tensor.max() > 1.0: # Ensure normalization if not already 0-1
            img_tensor = img_tensor / 255.0

        # Add batch dimension if it's not there (C,H,W -> B,C,H,W)
        if img_tensor.dim() == 3:
            self.input_tensor = img_tensor.unsqueeze(0)
        elif img_tensor.dim() == 4:
            self.input_tensor = img_tensor # Already has batch
        else:
            raise ValueError(f"Input tensor should be 3D or 4D, got {img_tensor.dim()}")

        # Prepare optional second image tensor
        self.second_input_tensor = None
        if self.second_img is not None:
            if self.second_img.ndim == 3:
                if self.second_img.shape[2] == 3:
                    img2_tensor = torch.from_numpy(self.second_img).permute(2,0,1).float()
                elif self.second_img.shape[2] == 4:
                    img2_tensor = torch.from_numpy(self.second_img[:,:,:3]).permute(2,0,1).float()
                else:
                    raise ValueError(f"Unsupported number of channels in second image: {self.second_img.shape[2]}")
            elif self.second_img.ndim == 2:
                img2_tensor = torch.from_numpy(self.second_img).unsqueeze(0).float()
            else:
                raise ValueError(f"Unsupported second image dimensions: {self.second_img.ndim}")
            if img2_tensor.max() > 1.0:
                img2_tensor = img2_tensor / 255.0
            if img2_tensor.dim() == 3:
                self.second_input_tensor = img2_tensor.unsqueeze(0)
            elif img2_tensor.dim() == 4:
                self.second_input_tensor = img2_tensor
            else:
                raise ValueError(f"Second input tensor should be 3D or 4D, got {img2_tensor.dim()}")

        self.scotoma_radius = scotoma_radius
        self._update_plots()
        plt.show()

class ConcentricCirclesPlotter:
    def __init__(self, initial_rho0_px, initial_k, initial_R_max_cart_px, initial_N_rows, side=256, second_image_path=None):
        self.side = side
        # Define fixed radii and colors for the circles image
        self.fixed_radii_for_drawing_px = [6, 16, 26, 36, 46, 56, 66, 76, 86, 96, 106, 116]
        self.circle_colors = [
            (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0),
            (0.7, 0.3, 0.0), (0.3, 0.7, 0.0), (0.0, 0.7, 0.3),
            (0.7, 0.0, 0.3), (0.3, 0.0, 0.7), (0.0, 0.3, 0.7)
        ]
        if len(self.fixed_radii_for_drawing_px) > len(self.circle_colors):
            self.circle_colors = (self.circle_colors * (len(self.fixed_radii_for_drawing_px) // len(self.circle_colors) + 1))

        # Create the Cartesian image with circles (once)
        self.cartesian_img_circles_tensor = self._create_cartesian_circles_image()

        # Optionally load a second image (to show only its transformed view)
        self.second_input_tensor_circles = None
        if second_image_path is not None:
            second_img_np = np.array(Image.open(second_image_path))
            if second_img_np.ndim == 3:
                pil2 = Image.fromarray(second_img_np)
                second_img_np = np.array(pil2.resize((self.side, self.side), Image.Resampling.LANCZOS))
                img2_tensor = torch.from_numpy(second_img_np).permute(2, 0, 1).float()
                if img2_tensor.shape[0] > 3:
                    img2_tensor = img2_tensor[:3]
            elif second_img_np.ndim == 2:
                pil2 = Image.fromarray(second_img_np)
                second_img_np = np.array(pil2.resize((self.side, self.side), Image.Resampling.LANCZOS))
                img2_tensor = torch.from_numpy(second_img_np).unsqueeze(0).float()
            else:
                raise ValueError(f"Unsupported second image dimensions: {second_img_np.shape}")
            if img2_tensor.max() > 1.0:
                img2_tensor = img2_tensor / 255.0
            if img2_tensor.dim() == 3:
                self.second_input_tensor_circles = img2_tensor.unsqueeze(0)
            elif img2_tensor.dim() == 4:
                self.second_input_tensor_circles = img2_tensor

        # Setup figure and axes: top row for 4 plots, 2 rows for sliders in 2x2
        self.fig_circles = plt.figure(figsize=(20, 10))
        gs_circles = self.fig_circles.add_gridspec(1 + 2, 4, height_ratios=[1, 0.08, 0.08], width_ratios=[1, 1, 1, 1], hspace=0.6, wspace=0.5)

        # Top row visuals
        self.ax_cartesian_circles = self.fig_circles.add_subplot(gs_circles[0, 0])
        self.ax_logpolar_circles = self.fig_circles.add_subplot(gs_circles[0, 1])
        self.ax_second_logpolar_circles = self.fig_circles.add_subplot(gs_circles[0, 2])
        self.ax_mapping_circles = self.fig_circles.add_subplot(gs_circles[0, 3])

        # Sliders arranged 2x2
        self.slider_rho0_ax_circles = self.fig_circles.add_subplot(gs_circles[1, 0:2])
        self.slider_k_ax_circles = self.fig_circles.add_subplot(gs_circles[1, 2:4])
        self.slider_R_max_cart_px_ax_circles = self.fig_circles.add_subplot(gs_circles[2, 0:2])
        self.slider_N_rows_ax_circles = self.fig_circles.add_subplot(gs_circles[2, 2:4])

        # Store and set initial slider values
        self.current_rho0_px = initial_rho0_px
        self.current_k = initial_k
        self.current_R_max_cart_px = initial_R_max_cart_px
        self.current_N_rows = initial_N_rows

        # Instantiate the transform object for this plotter
        self.log_polar_transform_circles = LogPolarTransform(
            rho_0_px=self.current_rho0_px,
            k=self.current_k,
            R_max_cart_px=self.current_R_max_cart_px,
            N_rows=int(self.current_N_rows),
            N_cols=int(self.current_N_rows)
        )

        # Create sliders
        self.slider_rho0_circles = Slider(
            ax=self.slider_rho0_ax_circles, label='Pivot ρ₀',
            valmin=1, valmax=self.side,
            valinit=self.current_rho0_px
        )
        self.slider_k_circles = Slider(
            ax=self.slider_k_ax_circles, label='Strength k',
            valmin=-30.0, valmax=1.0,
            valinit=self.current_k
        )
        self.slider_R_max_cart_px_circles = Slider(
            ax=self.slider_R_max_cart_px_ax_circles, label='R_max_px (Sample Extent)',
            valmin=1,
            valmax=self.side,
            valinit=self.current_R_max_cart_px
        )
        self.slider_N_rows_circles = Slider(
            ax=self.slider_N_rows_ax_circles, label='N_rows (LP Height)',
            valmin=16,
            valmax=self.side * 10,
            valinit=self.current_N_rows, valstep=1
        )

        # Avoid overlapping labels by anchoring them to the left within each slider axis
        for s in [self.slider_rho0_circles, self.slider_k_circles, self.slider_R_max_cart_px_circles, self.slider_N_rows_circles]:
            try:
                s.label.set_fontsize(9)
                s.label.set_ha('left')
                s.label.set_x(0.01)
                if hasattr(s, 'valtext') and s.valtext is not None:
                    s.valtext.set_fontsize(9)
                    s.valtext.set_ha('right')
                    s.valtext.set_x(0.99)
            except Exception:
                pass

        # Connect sliders to update function
        self.slider_rho0_circles.on_changed(self._update_plot_circles)
        self.slider_k_circles.on_changed(self._update_plot_circles)
        self.slider_R_max_cart_px_circles.on_changed(self._update_plot_circles)
        self.slider_N_rows_circles.on_changed(self._update_plot_circles)

    def _create_cartesian_circles_image(self):
        yy, xx = torch.meshgrid(torch.arange(self.side, dtype=torch.float32),
                                torch.arange(self.side, dtype=torch.float32),
                                indexing='ij')
        center_coord = (self.side - 1) / 2.0
        cartesian_img_circles = torch.zeros(3, self.side, self.side)
        for r_idx, r_val in enumerate(self.fixed_radii_for_drawing_px):
            # Draw thin rings by checking if distance is very close to r_val
            distance_from_center = torch.sqrt((xx - center_coord)**2 + (yy - center_coord)**2)
            mask = (distance_from_center >= r_val - 0.5) & (distance_from_center < r_val + 0.5)
            # mask = torch.round(torch.sqrt((xx - center_coord)**2 + (yy - center_coord)**2)) == r_val # old version
            cartesian_img_circles[:, mask] = torch.tensor(self.circle_colors[r_idx]).view(3, 1)
        return cartesian_img_circles.unsqueeze(0) # Add batch dimension

    def _update_plot_circles(self, event=None):
        # Update current values from sliders
        self.current_rho0_px = self.slider_rho0_circles.val
        self.current_k = self.slider_k_circles.val
        self.current_R_max_cart_px = self.slider_R_max_cart_px_circles.val
        self.current_N_rows = int(self.slider_N_rows_circles.val)

        # Clear axes
        self.ax_cartesian_circles.clear()
        self.ax_logpolar_circles.clear()
        if hasattr(self, 'ax_second_logpolar_circles'):
            self.ax_second_logpolar_circles.clear()
        if hasattr(self, 'ax_mapping_circles'):
            self.ax_mapping_circles.clear()

        # Re-plot Cartesian circles (it's static)
        self.ax_cartesian_circles.imshow(self.cartesian_img_circles_tensor[0].permute(1, 2, 0).numpy())
        self.ax_cartesian_circles.set_title('Cartesian Image with Concentric Circles')
        self.ax_cartesian_circles.axis('off')

        # Apply LogPolarTransform
        _N_cols_c = int(self.current_N_rows) # Keep LP image square

        self.log_polar_transform_circles.update_params(
            rho_0_px=self.current_rho0_px,
            k=self.current_k,
            R_max_cart_px=self.current_R_max_cart_px,
            N_rows=int(self.current_N_rows),
            N_cols=_N_cols_c
        )

        log_polar_circles_tensor = self.log_polar_transform_circles(self.cartesian_img_circles_tensor)
        log_polar_circles_np = log_polar_circles_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

        self.ax_logpolar_circles.imshow(log_polar_circles_np)
        self.ax_logpolar_circles.set_title('Log-Polar Transformed')

        # If a second image was provided, show only its transformed view
        if self.second_input_tensor_circles is not None:
            log_polar_second_tensor = self.log_polar_transform_circles(self.second_input_tensor_circles)
            log_polar_second_np = log_polar_second_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            self.ax_second_logpolar_circles.imshow(log_polar_second_np)
            self.ax_second_logpolar_circles.set_title('Second: Log-Polar')
            self.ax_second_logpolar_circles.set_xticks([])
            self.ax_second_logpolar_circles.set_yticks([])

        # Compute recommendation for the circles figure as well
        recommended_rows_c = rows_needed_to_cover_center(self.current_rho0_px, self.current_k, self.current_R_max_cart_px, r_min_target=1.0)
        print(f"[CirclesPlotter] ρ₀={self.current_rho0_px:.2f}, k={self.current_k:.2f}, R_max={self.current_R_max_cart_px:.0f}px -> recommended N_rows ≥ {recommended_rows_c} (current={self.current_N_rows}) to include r≥1px")
        suffix = f" | need N_rows≥{recommended_rows_c}" if recommended_rows_c > self.current_N_rows else f" | OK (≤{recommended_rows_c})"
        self.fig_circles.suptitle(f'Log-Polar Mapping of Specific Radii\nρ₀={self.current_rho0_px:.1f}, k={self.current_k:.2f}, R_max={self.current_R_max_cart_px:.0f}px, N_rows={self.current_N_rows}{suffix}', fontsize=10)

        # Mapping plot (r(j) vs j)
        try:
            self.ax_mapping_circles.set_box_aspect(1)
        except Exception:
            try:
                self.ax_mapping_circles.set_aspect('equal', adjustable='box')
            except Exception:
                pass
        j_vals = np.arange(1, int(self.current_N_rows) + 1)
        r_vals = calculate_cartesian_radius(j_vals, self.current_rho0_px, self.current_k, self.current_R_max_cart_px, self.current_N_rows)
        self.ax_mapping_circles.plot(r_vals, j_vals, lw=2, label="New Transform")
        # log2 reference
        r_values_log2 = np.linspace(1, self.current_R_max_cart_px, int(self.current_N_rows))
        j_values_log2 = np.log2(np.maximum(r_values_log2, 2e-6)) * (self.current_N_rows / np.log2(max(self.current_R_max_cart_px, 2)))
        self.ax_mapping_circles.plot(r_values_log2, j_values_log2, lw=2, linestyle='--', color='red', label="log2 Transform")
        self.ax_mapping_circles.axvline(x=self.current_rho0_px, color='r', linestyle='--', label='r = ρ₀ (pivot)')
        self.ax_mapping_circles.set_ylabel("Log-Polar Row Index (j)")
        self.ax_mapping_circles.set_xlabel("Cartesian Radius r(j) [pixels]")
        self.ax_mapping_circles.set_title("Mapping: r(j) vs j")
        self.ax_mapping_circles.legend(loc='lower right')
        self.ax_mapping_circles.grid(True)
        self.ax_mapping_circles.set_ylim(0, self.current_N_rows)
        self.ax_mapping_circles.set_xlim(0, self.current_R_max_cart_px * 1.1)
        self.ax_mapping_circles.invert_yaxis()

        # --- Quantitative Spacing Analysis (printed to console) ---
        print("\n--- Quantitative Spacing Analysis (Interactive Circles Figure) ---")
        mapped_radii_info = []
        num_rows_lp_circles = log_polar_circles_np.shape[0]

        for r_val in self.fixed_radii_for_drawing_px:
            # We need the inverse function j(r) to find where a radius lands.
            # j(r) = j_max + (r^(k+1) - r_max^(k+1)) / (rho_0^k * (k+1))
            # Let's calculate it.
            k_p_1 = self.current_k + 1.0
            is_valid_for_transform = False
            j_float = float('nan')

            if np.isclose(self.current_k, -1.0):
                # Inverse for classic log transform: j = j_max + rho_0 * log(r / r_max)
                if r_val > 0 and self.current_R_max_cart_px > 0:
                     j_float = self.current_N_rows + self.current_rho0_px * np.log(r_val / self.current_R_max_cart_px)
            else:
                # Inverse for generalized transform
                num = np.power(r_val, k_p_1) - np.power(self.current_R_max_cart_px, k_p_1)
                den = np.power(self.current_rho0_px, self.current_k) * k_p_1
                if np.abs(den) > 1e-9:
                    j_float = self.current_N_rows + num / den

            if 1 <= j_float <= self.current_N_rows:
                is_valid_for_transform = True

            mapped_radii_info.append({
                'r_px': r_val,
                'j_float': j_float, # Keep it float for now
                'valid': is_valid_for_transform
            })

        print(f"Log-polar image rows (N_rows): {num_rows_lp_circles}")
        print(f"Transform parameters: ρ₀={self.current_rho0_px:.2f}, k={self.current_k:.2f}, R_max={self.current_R_max_cart_px:.2f}, N_rows={self.current_N_rows}")
        print("Mapping of original radii to log-polar image rows (1-indexed, float):")

        last_valid_j_rounded = None
        last_valid_r_px = None

        for info in mapped_radii_info:
            r_px, j_float, is_valid = info['r_px'], info['j_float'], info['valid']
            if is_valid and not np.isnan(j_float):
                j_rounded = int(round(j_float))
                print(f"  R={r_px:>3}px -> j={j_float:7.2f} (maps to LP row ~{j_rounded})", end="")
                if last_valid_j_rounded is not None:
                    row_diff = j_rounded - last_valid_j_rounded
                    radius_diff = r_px - last_valid_r_px
                    print(f" (Sep from {last_valid_r_px}px: {row_diff:>3} rows for {radius_diff}px diff)")
                else:
                    print()
                last_valid_j_rounded = j_rounded
                last_valid_r_px = r_px
            else:
                print(f"  R={r_px:>3}px -> (Outside sampled range)")
        print("-----------------------------------------------------")

        # --- Y-axis ticks for ConcentricCirclesPlotter ---
        # The y-axis now simply corresponds to the row index j.
        if num_rows_lp_circles > 0:
            tick_positions = [0, num_rows_lp_circles // 2, num_rows_lp_circles - 1]
            tick_labels = ['j=1', f'j={num_rows_lp_circles // 2 + 1}', f'j={num_rows_lp_circles}']
            self.ax_logpolar_circles.set_yticks(tick_positions)
            self.ax_logpolar_circles.set_yticklabels(tick_labels)

        self.ax_logpolar_circles.set_ylabel("Log-Polar Row Index (j)")
        self.ax_logpolar_circles.set_xticks([]) # Hide x-ticks for log-polar

        self.fig_circles.canvas.draw_idle()

    def plot(self):
        self._update_plot_circles() # Initial plot based on initial slider values
        plt.show()


if __name__ == "__main__":
    #image_path = "/home/tomasdu/repos/datasets/eth80/val/dog/dog1-022-180.png"
    # image_path = "/home/tomasdu/repos/datasets/eth80-padded-circular/val/dog/dog1-022-180.png"
    #image_path = "/home/tomasdu/logp_original_perrinet.png"
    image_path = "/home/ttdduu/logp_original.png"

    print("\n--- Launching Interactive Log-Polar Figure (Circles + Mapping + Sliders) --- ")

    # Initial values for the interactive figure
    initial_rho0_px = 80.0
    initial_k = -1.0
    initial_R_max_cart_px = 112  # for 224x224
    initial_N_rows = 224

    # Optional: provide a second image path to see its transformed view alongside the circles
    second_image_path_circles = image_path  # or set to None

    circles_plotter = ConcentricCirclesPlotter(
        initial_rho0_px=initial_rho0_px,
        initial_k=initial_k,
        initial_R_max_cart_px=initial_R_max_cart_px,
        initial_N_rows=initial_N_rows,
        side=224,
        second_image_path=second_image_path_circles
    )
    circles_plotter.plot()

    print("\n--- Figure closed. Script finished. ---")
