import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from src.experiments.logp.PolarTransform import PolarTransform

def log_polar_transform(image_path):
    # Load and preprocess image
    img = Image.open(image_path)
    img = img.convert('RGB')
    
    # Use the exact same parameters as in cortical_magnification.py
    img_size = 224
    out_size = 224
    input_h, input_w = img_size, img_size
    output_h, output_w = out_size, out_size
    
    # These values are already in log space, so we don't need to take log2
    min_radius = -5  # rmin
    max_radius = 0   # rmax
    n_samples = img_size
    
    # Create exponentially spaced radii (converting from log space back to linear)
    dists = np.exp2(np.linspace(min_radius, max_radius, n_samples))
    
    # Create angle bins - use n_samples + 1 points and exclude the last one
    # This ensures we don't have duplicate angles at 0 and 2π
    angle_bins = np.linspace(0, 2*np.pi, n_samples + 1)[:-1]
    
    radius_bins = dists
    
    # Resize image to match paper's input size
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0)
    
    # Create transform with paper's parameters
    transform = PolarTransform(
        input_h=input_h,
        input_w=input_w,
        output_h=output_h,
        output_w=output_w,
        radius_bins=radius_bins,
        angle_bins=angle_bins,
        interpolation='linear',
        subbatch_size=1
    )
    
    # Apply transform
    with torch.no_grad():
        transformed = transform(img_tensor)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12, 6))
    
    # Original image with foveal radius circle
    ax1.imshow(img_tensor.squeeze(0).permute(1, 2, 0))
    circle = plt.Circle((img_size//2, img_size//2), foveal_radius, 
                       color='blue', fill=False)
    ax1.add_patch(circle)
    ax1.plot(img_size//2, img_size//2, 'b.')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Transformed image
    transformed_img = transformed.squeeze(0).permute(1, 2, 0).numpy()
    # transformed_img = np.flipud(transformed_img)
    ax2.imshow(transformed_img)
    ax2.set_title('Log-Polar Transformed')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = "/home/tomasdu/imgnet.JPEG"
    #image_path = "/home/tomasdu/aa.png"
    log_polar_transform(image_path)
