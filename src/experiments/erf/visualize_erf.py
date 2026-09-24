import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from src.config import BaseConfig
from src.scotoma import ScotomaApplier
from src.data.transforms.log_polar import LogPolarTransform
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

def load_eth80_image(data_dir):
    # Define image transformations (same as in training pipeline)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Get first image from first class
    data_dir = Path(data_dir)
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if class_dirs:
        img_files = list(class_dirs[0].glob('*.png'))
        if img_files:
            img_path = img_files[0]
            img = Image.open(img_path)
            # Save the original image for display
            display_img = np.array(img.resize((224, 224)))
            # Transform for model (includes normalization)
            img_tensor = transform(img)
            return display_img, img_tensor
    return None, None

def calculate_region_averages(matrix, n_regions=3):
    """Calculate average values for nxn grid regions in the matrix"""
    h, w = matrix.shape
    region_h = h // n_regions
    region_w = w // n_regions
    averages = np.zeros((n_regions, n_regions))
    
    for i in range(n_regions):
        for j in range(n_regions):
            region = matrix[i*region_h:(i+1)*region_h, 
                          j*region_w:(j+1)*region_w]
            averages[i, j] = np.mean(region)
    
    return averages, region_h, region_w

def calculate_concentric_quarter_averages(matrix, n_circles=5):
    """Calculate average values for concentric quarter-circles in the matrix"""
    h, w = matrix.shape
    center_y, center_x = h // 2, w // 2
    max_radius = min(h, w) // 2
    radius_step = max_radius / n_circles
    
    # Initialize averages: [n_circles, 4] array for 4 quarters in each circle
    averages = np.zeros((n_circles, 4))
    
    # Create coordinate grids
    y, x = np.ogrid[:h, :w]
    # Center the coordinates
    y = y - center_y
    x = x - center_x
    
    # Calculate radius for each point
    radius = np.sqrt(x*x + y*y)
    # Calculate angle for each point
    angle = np.arctan2(y, x)
    
    # Define quarter regions (-π to π)
    quarter_bounds = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
    
    # Calculate averages for each concentric quarter
    for i in range(n_circles):
        inner_r = i * radius_step
        outer_r = (i + 1) * radius_step
        ring_mask = (radius >= inner_r) & (radius < outer_r)
        
        for q in range(4):
            # Create quarter mask
            if q == 3:  # Handle the wraparound case
                quarter_mask = (angle >= quarter_bounds[q]) | (angle < quarter_bounds[0])
            else:
                quarter_mask = (angle >= quarter_bounds[q]) & (angle < quarter_bounds[q+1])
            
            # Combine ring and quarter masks
            region_mask = ring_mask & quarter_mask
            if np.any(region_mask):
                averages[i, q] = np.mean(matrix[region_mask])
    
    return averages, radius_step

def plot_erf_with_regions(ax, matrix, n_regions=3, vmin=None, vmax=None):
    """Plot ERF heatmap with region averages"""
    # Plot the basic heatmap with normalized values
    im = ax.imshow(matrix, cmap='hot', vmin=vmin, vmax=vmax)
    
    # Calculate region averages
    averages, region_h, region_w = calculate_region_averages(matrix, n_regions)
    
    # Draw grid lines and add average values
    h, w = matrix.shape
    
    # Draw vertical and horizontal lines
    for i in range(1, n_regions):
        ax.axvline(x=i*region_w - 0.5, color='white', linestyle='-', linewidth=0.5)
        ax.axhline(y=i*region_h - 0.5, color='white', linestyle='-', linewidth=0.5)
    
    # Add average values in each region
    for i in range(n_regions):
        for j in range(n_regions):
            center_y = i * region_h + region_h/2 - 0.5
            center_x = j * region_w + region_w/2 - 0.5
            
            ax.text(center_x, center_y, 
                   f'{averages[i,j]:.3f}',
                   ha='center', va='center',
                   color='white', fontsize=12,  # Increased font size
                   bbox=dict(facecolor='black', alpha=0.5, pad=1))
    
    return im

def plot_erf_with_concentric_quarters(ax, matrix, n_circles=5, vmin=None, vmax=None):
    """Plot ERF heatmap with concentric quarter regions"""
    # Plot the basic heatmap
    im = ax.imshow(matrix, cmap='hot', vmin=vmin, vmax=vmax)
    
    # Calculate region averages
    averages, radius_step = calculate_concentric_quarter_averages(matrix, n_circles)
    
    h, w = matrix.shape
    center_y, center_x = h // 2, w // 2
    
    # Draw circles
    for i in range(n_circles):
        radius = (i + 1) * radius_step
        circle = plt.Circle((center_x, center_y), radius, 
                          fill=False, color='white', linestyle='-', linewidth=0.5)
        ax.add_artist(circle)
    
    # Draw quarter lines
    max_radius = min(h, w) // 2
    ax.plot([center_x, center_x], [center_y - max_radius, center_y + max_radius], 
            color='white', linestyle='-', linewidth=0.5)
    ax.plot([center_x - max_radius, center_x + max_radius], [center_y, center_y], 
            color='white', linestyle='-', linewidth=0.5)
    
    # Add average values in each quarter
    for i in range(n_circles):
        radius = (i + 0.5) * radius_step  # Middle of the ring
        for q in range(4):
            angle = np.pi/4 + q * np.pi/2  # Middle of each quarter
            text_x = center_x + radius * np.cos(angle)
            text_y = center_y + radius * np.sin(angle)
            
            ax.text(text_x, text_y, 
                   f'{averages[i,q]:.3f}',
                   ha='center', va='center',
                   color='white', fontsize=8,
                   bbox=dict(facecolor='black', alpha=0.5, pad=1))
    
    return im

def plot_erf(ax, erf_matrix, title=None, vmin=None, vmax=None):
    """
    Plot just the ERF heatmap without any grid overlays.
    """
    im = ax.imshow(erf_matrix, cmap='hot', vmin=0, vmax=vmax)
    ax.axis('off')
    if title:
        ax.set_title(title)
    return im

def visualize_erf(source_path, save_path, input_type='constant', data_dir=None, 
                 apply_scotoma=False, apply_logpolar=False, config=None):
    # Load the ERF matrix
    erf_matrix = np.load(source_path)
    
    # Initialize config and transformations if needed
    config = config or BaseConfig(parse_args=False)
    scotoma_applier = ScotomaApplier(config) if apply_scotoma else None
    logpolar_transform = LogPolarTransform() if apply_logpolar else None
    
    if data_dir is None:
        raise ValueError("data_dir must be provided for eth80 input type")
    display_img, x = load_eth80_image(data_dir)
    if display_img is None:
        raise ValueError("Could not load ETH80 image")
    x = x.unsqueeze(0)  # Already normalized by load_eth80_image
    title = 'Input Image\n(ETH80)'
    
    # Apply transformations if requested (on normalized tensor)
    transformed_img = x.clone()
    if scotoma_applier:
        transformed_img = scotoma_applier.apply_scotoma(transformed_img)
        title += ' + Scotoma'
    
    if logpolar_transform:
        transformed_img = logpolar_transform(transformed_img)
        title += ' + LogPolar'
    
    # Convert transformed tensor to display image (un-normalize)
    display_transformed = transformed_img[0].clone()
    display_transformed = transforms.Normalize(
        mean=[-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
        std=[1/s for s in [0.229, 0.224, 0.225]]
    )(display_transformed)
    display_transformed = display_transformed.permute(1, 2, 0).numpy()
    
    # Create figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot original input image
    ax1.imshow(display_img)
    ax1.set_title('Original Input')
    ax1.axis('on')
    
    # Plot transformed input image
    ax2.imshow(display_transformed)
    ax2.set_title(title)
    ax2.axis('on')
    
    # Plot ERF heatmap with regions
    im = plot_erf_with_concentric_quarters(ax3, erf_matrix, n_circles=5)
    ax3.set_title('Effective Receptive Field\nwith Region Averages')
    plt.colorbar(im, ax=ax3)
    
    # Add overall title
    plt.suptitle('Input Image, Transformed Input, and ERF Analysis', fontsize=14)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_all_erfs(config=None, model_dir=None, eth80_dir=None):
    # Use provided parameters or defaults
    #if config is None:
    #    config = BaseConfig(parse_args=False)
    #if model_dir is None:
    #    model_dir = '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-mar19/WS-S/wandb/offline-run-20250321_002408-gk29fe5c/files/model'
    #if eth80_dir is None:
    #    eth80_dir = "/home/tomasdu/repos/datasets/eth80-padded-circular/val"
    
    visualize_erf(
        f"{model_dir}/erf_eth80.npy",
        f"{model_dir}/heatmap_eth80.png",
        input_type='eth80',
        data_dir=eth80_dir,
        config=config
    )
    
    visualize_erf(
        f"{model_dir}/erf_eth80_scotoma.npy",
        f"{model_dir}/heatmap_eth80_scotoma.png",
        input_type='eth80',
        data_dir=eth80_dir,
        apply_scotoma=True,
        config=config
    )
    
    visualize_erf(
        f"{model_dir}/erf_eth80_logpolar.npy",
        f"{model_dir}/heatmap_eth80_logpolar.png",
        input_type='eth80',
        data_dir=eth80_dir,
        apply_logpolar=True,
        config=config
    )
    
    visualize_erf(
        f"{model_dir}/erf_eth80_scotoma_logpolar.npy",
        f"{model_dir}/heatmap_eth80_scotoma_logpolar.png",
        input_type='eth80',
        data_dir=eth80_dir,
        apply_scotoma=True,
        apply_logpolar=True,
        config=config
    )

def create_combined_figure(config=None, model_dir=None, eth80_dir=None):
    # Extract wandb ID and create suffixes
    wandb_id = model_dir.split('offline-run-')[-1].split('/')[0]
    
    # Create two different suffixes:
    # One for ERF matrices (includes both target and input class)
    erf_suffix = f"-{wandb_id}-{config.data.target_class}-{config.data.eth80_class}"
    # One for input images (includes only input class)
    input_suffix = f"-{wandb_id}-{config.data.eth80_class}"
    
    # Input configurations with correct suffixes
    input_configs = {
        'original': {
            'img_path': f"{model_dir}/img_eth80{input_suffix}.pt",
            'erf_path': f"{model_dir}/erf_eth80{erf_suffix}.npy",
            'title': 'Original'
        },
        'scotoma': {
            'img_path': f"{model_dir}/img_eth80_scotoma{input_suffix}.pt",
            'erf_path': f"{model_dir}/erf_eth80_scotoma{erf_suffix}.npy",
            'title': 'Scotoma'
        },
        'logpolar': {
            'img_path': f"{model_dir}/img_eth80_logpolar{input_suffix}.pt",
            'erf_path': f"{model_dir}/erf_eth80_logpolar{erf_suffix}.npy",
            'title': 'Log-polar'
        },
        'both': {
            'img_path': f"{model_dir}/img_eth80_scotoma_logpolar{input_suffix}.pt",
            'erf_path': f"{model_dir}/erf_eth80_scotoma_logpolar{erf_suffix}.npy",
            'title': 'Scotoma + Log-polar'
        }
    }
    
    # Create figure
    n_inputs = 4  # eth80, eth80+scotoma, eth80+logpolar, eth80+scotoma+logpolar
    fig = plt.figure(figsize=(12, 2*n_inputs))
    
    for idx, (input_type, input_config) in enumerate(input_configs.items()):
        # Load ERF matrix and normalize to 0-1
        erf_matrix = np.load(input_config['erf_path'])
        erf_matrix = (erf_matrix - erf_matrix.min()) / (erf_matrix.max() - erf_matrix.min())
        
        # Create two subplots for this input (removed original input)
        ax1 = plt.subplot(n_inputs, 2, idx*2 + 1)
        ax2 = plt.subplot(n_inputs, 2, idx*2 + 2)
        
        # Load the transformed image
        transformed_img = torch.load(input_config['img_path'])
        display_transformed = transformed_img[0].clone()
        #display_transformed = transforms.Normalize(
        #    mean=[-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
        #    std=[1/s for s in [0.229, 0.224, 0.225]]
        #)(display_transformed)
        display_transformed = display_transformed.permute(1, 2, 0).numpy()
        
        # Plot transformed input
        ax1.imshow(display_transformed)
        ax1.set_title(f"{input_config['title']}\nInput", fontsize=14)
        ax1.axis('off')
        
        # Plot ERF heatmap with normalized values
        im = plot_erf_with_concentric_quarters(ax2, erf_matrix, n_circles=5, vmin=0, vmax=1)
        ax2.set_title(f"{input_config['title']}\nERF", fontsize=14)
        plt.colorbar(im, ax=ax2)
        
        # Increase tick label size
        ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # Adjust layout
    #plt.suptitle('ERF Analysis', fontsize=16, y=0.95)
    plt.tight_layout()
    
    # Save with suffix
    #save_path = f"{model_dir}/combined_erf_analysis{erf_suffix}.png"
    #plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #general_path = f'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-mar19/WS-S/erf/last_conv_layer/combined_erf_analysis{erf_suffix}'
    general_path = f'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc5/WS-S/erf/head/nologp/combined_erf_analysis{erf_suffix}'
    plt.savefig(general_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_neuron_grid(model_dir, target_class, radius):
    """
    Creates a 4x4 grid showing how a specific neuron responds to all input classes.
    For each input class, shows:
    - Original input ERF
    - Scotoma input ERF
    All ERF values are normalized between 0 and 1
    """
    wandb_id = model_dir.split('offline-run-')[-1].split('/')[0]
    classes = ['apple', 'car', 'cow', 'cup', 'dog', 'horse', 'pear', 'tomato']
    
    # Create figure with 4x4 grid (4 input classes per row, 2 rows)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle(f'ERF Analysis: {target_class.capitalize()} Output Neuron, scotoma radius {radius}', fontsize=16)
    
    # First, load all ERF matrices and find global min/max for normalization
    all_erfs = []
    for input_class in classes:
        erf_suffix = f"-{wandb_id}-{target_class}-{input_class}"
        
        # Load original and scotoma ERFs
        erf = np.load(f"{model_dir}/erf_eth80{erf_suffix}.npy")
        erf_scotoma = np.load(f"{model_dir}/erf_eth80_scotoma{erf_suffix}.npy")
        
        all_erfs.extend([erf, erf_scotoma])
    
    # Calculate global min/max for normalization
    global_min = min(erf.min() for erf in all_erfs)
    global_max = max(erf.max() for erf in all_erfs)
    
    # For each input class
    for i, input_class in enumerate(classes):
        row = i // 2  # Integer division to determine row
        col = (i % 2) * 2  # Multiply by 2 to leave space for scotoma
        
        erf_suffix = f"-{wandb_id}-{target_class}-{input_class}"
        
        # Load and normalize original ERF
        erf = np.load(f"{model_dir}/erf_eth80{erf_suffix}.npy")
        erf = (erf - global_min) / (global_max - global_min)  # normalize to [0,1]
        plot_erf(axes[row, col], erf, title=f"{input_class.capitalize()} - Original", 
                vmin=0, vmax=1)
        
        # Load and normalize scotoma ERF
        erf_scotoma = np.load(f"{model_dir}/erf_eth80_scotoma{erf_suffix}.npy")
        erf_scotoma = (erf_scotoma - global_min) / (global_max - global_min)  # normalize to [0,1]
        plot_erf(axes[row, col+1], erf_scotoma, title=f"{input_class.capitalize()} - Scotoma", 
                vmin=0, vmax=1)
    
    # Add colorbar without overlapping
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='hot', norm=norm)
    fig.colorbar(sm, cax=cbar_ax)
    
    # Save both PNG and SVG versions
    plt.savefig(f"{model_dir}/erf_grid_{wandb_id}_{target_class}_neuron.png", 
                dpi=300, bbox_inches='tight')
    plt.savefig(f"{model_dir}/erf_grid_{wandb_id}_{target_class}_neuron.svg", 
                format='svg', bbox_inches='tight')
    plt.close()

def load_erfs_from_model(model_dir, target_class):
    """Load ERF matrix for a specific target class from a model directory"""
    wandb_id = model_dir.split('offline-run-')[-1].split('/')[0]
    erf_path = f"{model_dir}/erf_eth80_scotoma-{wandb_id}-{target_class}.npy"
    
    if os.path.exists(erf_path):
        return np.load(erf_path)
    else:
        print(f"Warning: ERF file not found: {erf_path}")
        return None

def f_of_radius(model_radius_list):
    classes = ['apple', 'car', 'cow', 'cup', 'dog', 'horse', 'pear', 'tomato']
    n_models = len(model_radius_list[0])  # Number of different scotoma radii
    n_classes = len(classes)
    
    # Create a figure with n_models rows and n_classes columns
    fig, axes = plt.subplots(n_models, n_classes, figsize=(20, 4*n_models))

    if model_radius_list[2] == 'og':
        architecture_substring = 'original ConvNeXt'
    elif model_radius_list[2] == 'lc':
        architecture_substring = 'LC ConvNeXt'

    if model_radius_list[3] == 'LH-S':
        exp_substring = 'finetuning without scotoma'
    elif model_radius_list[3] == 'WS-S':
        exp_substring = 'finetuned with scotoma'

    fig.suptitle(f'ERF Analysis: ERF of each output neuron of the {architecture_substring} architecture after {exp_substring}', fontsize=16)
    
    # Find min and max for each class for consistent colormap within classes
    class_min_max = {}
    for class_name in classes:  # For each output neuron
        class_min = float('inf')
        class_max = float('-inf')
        for pair in model_radius_list[0]:  # For each radius
            erf = load_erfs_from_model(pair[1], class_name)
            if erf is not None:
                class_min = min(class_min, erf.min())
                class_max = max(class_max, erf.max())
        class_min_max[class_name] = (class_min, class_max)
    
    # Plot ERFs
    for row, pair in enumerate(model_radius_list[0]):  # For each radius
        for col, class_name in enumerate(classes):  # For each output neuron
            erf = load_erfs_from_model(pair[1], class_name)
            if erf is not None:
                print(f"Plotting ERF for radius {pair[0]}%, class {class_name}")
                vmin, vmax = class_min_max[class_name]
                plot_erf(axes[col], erf, #changed from axes[row,col] to axes[col] in order to do it with only one model
                        title=f"r={pair[0]}%, {class_name}",
                        #vmin=vmin, vmax=vmax
                        )
            else:
                axes[row, col].text(0.5, 0.5, 'No data', 
                                  horizontalalignment='center',
                                  verticalalignment='center')
                axes[row, col].axis('off')
    
    # Add colorbar
    fig.subplots_adjust(right=0.9)
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    #norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='hot')
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.set_title('ERF Intensity')

    """
    model_radius_list looks like this:
    [[[0, '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/LH-S/wandb/offline-run-20250408_183732-n8fzbmp0/files/model'], 
    [5, '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/LH-S/wandb/offline-run-20250408_185337-nhjsewh4/files/model'], 
    [10, '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/LH-S/wandb/offline-run-20250408_190942-exa7jf7z/files/model'], 
    [15, '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/LH-S/wandb/offline-run-20250408_192543-yig9iuod/files/model'], 
    [20, '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/LH-S/wandb/offline-run-20250408_194152-ue4x5my7/files/model'], 
    [21, '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/LH-S/wandb/offline-run-20250408_195807-zyhfl6sn/files/model'], 
    [22, '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/LH-S/wandb/offline-run-20250408_201421-w8y7jul3/files/model'], 
    [23, '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/LH-S/wandb/offline-run-20250408_203036-tw1iv4kp/files/model'], 
    [24, '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/LH-S/wandb/offline-run-20250408_204654-1s0uz5t9/files/model'], 
    [25, '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/LH-S/wandb/offline-run-20250408_210313-yj5kpgwa/files/model'], 
    [30, '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/LH-S/wandb/offline-run-20250408_182124-o5gv847y/files/model']], False, 'lc', 'LH-S']

    
    """

    save_dir_base = f'{model_radius_list[0][0][1].split(model_radius_list[3])[0]}erf/head/nologp/ps00_without_scotoma/output_f_rad-{model_radius_list[2]}_atto_nologp-{model_radius_list[3]}-nonorm'
    # Create save directory if it doesn't exist
    os.makedirs(os.path.dirname(save_dir_base), exist_ok=True)
    
    plt.savefig(f"{save_dir_base}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir_base}.svg", format='svg', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # Default execution if run directly
    #config = BaseConfig(parse_args=False)
    #config.data.scotoma_radius = 20
    #config.data.scotoma_method = 'nice'
    #config.data.scotoma_strength = 1.0
    #config.data.scotoma_sharpness = 10
    
    visualize_all_erfs()
    #create_combined_figure()