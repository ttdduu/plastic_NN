import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np

def rho_theta_to_cartesian(log_polar_img, output_size=221, rho_0_px=80.0, k=-1.0, R_max_cart_px=112, N_rows=221):
    """
    Use inverse mapping with bilinear interpolation using torch.nn.functional.grid_sample
    """
    # Convert image to tensor
    img_tensor = transforms.ToTensor()(log_polar_img)

    # Create output coordinates
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, output_size),
        torch.linspace(-1, 1, output_size),
        indexing='ij'
    )

    # Convert to polar coordinates (r is radius in Cartesian space)
    r_cartesian = torch.sqrt(x**2 + y**2) * (output_size / 2.0)  # Scale to actual pixel radius
    theta = torch.atan2(-y, x)  # Negative y for clockwise
    theta = torch.where(theta < 0, theta + 2*np.pi, theta)

    # Apply inverse log-polar transform: rho(r) = rho_max + (r**(k+1) - rmax**(k+1))/(rho0**k * (k+1))
    k_plus_1 = k + 1.0
    if torch.isclose(torch.tensor(k), torch.tensor(-1.0)):
        # Inverse of classic log transform
        rho = N_rows + (torch.log(r_cartesian / R_max_cart_px) * rho_0_px)
    else:
        # Inverse of generalized power-law transform
        rho = N_rows + (r_cartesian**(k_plus_1) - R_max_cart_px**(k_plus_1)) / (rho_0_px**k * k_plus_1)
    
    # Normalize rho to [0, 1] for grid_sample
    rho_norm = rho / N_rows

    # Create sampling grid
    grid = torch.stack([
        2 * theta / (2*np.pi) - 1,  # Map theta from [0,2π] to [-1,1]
        2 * rho_norm - 1            # Map rho from [0,1] to [-1,1]
    ], dim=-1)

    # Sample using grid_sample
    output = torch.nn.functional.grid_sample(
        img_tensor.unsqueeze(0),  # Add batch dimension
        grid.unsqueeze(0),        # Add batch dimension
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )

    # Mask outside the circle
    mask = torch.ones_like(r_cartesian).unsqueeze(0).unsqueeze(0)
    return (output * mask)[0]  # Remove batch dimension properly

def process_image(image_path, output_size=224):
    # Load image
    img = Image.open(image_path).convert('RGB')

    # Transform to Cartesian
    cartesian = rho_theta_to_cartesian(img, output_size)

    # Create figure with white background
    plt.figure(figsize=(12, 5), facecolor='white')

    # Original
    plt.subplot(211)
    plt.imshow(img)
    plt.title('Input (θ-ρ space)')
    plt.xlabel('θ (angle)')
    plt.ylabel('ρ (radius)')

    # Transformed
    plt.subplot(212)
    cart_np = cartesian.permute(1, 2, 0).detach().numpy()

    # Create circular mask
    y, x = np.ogrid[:output_size, :output_size]
    center = output_size // 2
    mask = (x - center)**2 + (y - center)**2 <= center**2

    # Apply mask to make outside transparent
    cart_np_masked = np.zeros((output_size, output_size, 4))  # RGBA
    cart_np_masked[..., :3] = cart_np
    cart_np_masked[..., 3] = mask  # Alpha channel

    # Plot with transparent background
    plt.imshow(cart_np_masked)
    plt.title('Output (Cartesian space)')

    # Remove axes
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add concentric circles
    for r in np.arange(0.2, 1.2, 0.2):
        circle = plt.Circle((center, center),
                          r * center,
                          fill=False,
                          color='gray',
                          alpha=0.5,
                          linestyle='--')
        ax.add_artist(circle)

    # Add angle lines and labels clockwise
    for angle in range(0, 360, 45):
        rad = np.radians(angle)
        dx = np.cos(rad) * center
        dy = -np.sin(rad) * center  # Negative sin for clockwise

        # Draw line from center to edge
        plt.plot([center, center + dx], [center, center + dy],
                color='gray', alpha=0.5, linestyle='--')

        # Add angle labels outside the circle
        label_r = 1.15  # Push labels slightly further out
        plt.text(center + label_r*dx, center + label_r*dy,
                f"{angle}°",
                color='black',
                horizontalalignment='center',
                verticalalignment='center')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = "/home/tomasdu/donk.png"
 #   image_path = "/home/ttdduu/Downloads/20230331-_3310122-edit-ftsmithphotos.jpg"
    process_image(image_path, output_size=224)
