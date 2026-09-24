import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.patches import Rectangle

from types import SimpleNamespace
from src.models.convnext_atto import CustomConvNeXtAtto

def calculate_receptive_field(kernel_sizes, strides, paddings=None):
    """
    Calculate receptive field size using the formula:
    r_0 = sum_{i=1}^{L} [(k_i - 1) * product_{j=1}^{i-1} s_j] + 1
    
    Args:
        kernel_sizes: List of kernel sizes for each layer
        strides: List of strides for each layer
        paddings: List of paddings for each layer (not used in the basic formula)
    
    Returns:
        List of receptive field sizes for each layer
    """
    if len(kernel_sizes) != len(strides):
        raise ValueError("kernel_sizes and strides must have the same length")
    
    if paddings is None:
        paddings = [0] * len(kernel_sizes)
    
    # Calculate cumulative product of strides
    cum_strides = [1]
    for i in range(len(strides)):
        cum_strides.append(cum_strides[-1] * strides[i])
    cum_strides.pop(0)  # Remove the first 1
    
    # Calculate receptive field for each layer
    receptive_fields = [1]  # Start with 1 for the input layer
    for i in range(len(kernel_sizes)):
        rf = 1
        for j in range(i + 1):
            rf += (kernel_sizes[j] - 1) * (cum_strides[j] if j > 0 else 1)
        receptive_fields.append(rf)
    
    return receptive_fields

def get_model_layers():
    """
    Returns lists of kernel sizes, strides, and paddings for the ConvNeXtAtto model.
    """
    # ConvNeXt-Atto stem and stages
    kernel_sizes = [4, 7, 7, 2, 7, 2, 7, 2, 7]
    strides = [4, 1, 1, 2, 1, 2, 1, 2, 1]
    paddings = [0, 3, 3, 0, 3, 0, 3, 0, 3]
    
    layer_names = [
        "stem",
        "stage0_block1",
        "stage0_block2",
        "stage1_down",
        "stage1_block1",
        "stage2_down",
        "stage2_block1",
        "stage3_down",
        "stage3_block1"
    ]
    
    return kernel_sizes, strides, paddings, layer_names

def plot_rf_on_featuremap(ax, fmap, neuron_pos, rf_size, color='red'):
    """
    Plots a square receptive field on the feature map.
    neuron_pos: (y, x) in feature map coordinates
    rf_size: size of the receptive field (in pixels of the previous feature map)
    """
    y, x = neuron_pos
    half = rf_size // 2
    rect = plt.Rectangle(
        (x - half, y - half), rf_size, rf_size,
        linewidth=2, edgecolor=color, facecolor='none'
    )
    ax.add_patch(rect)
    ax.plot(x, y, 'o', color=color)
    ax.set_xlim(-0.5, fmap.shape[1] - 0.5)
    ax.set_ylim(fmap.shape[0] - 0.5, -0.5)

def draw_feature_map(ax, size, rf_size=None, center=None, title=None):
    """Draw a feature map with optional receptive field highlight"""
    # Draw the feature map as a blue rectangle
    ax.add_patch(Rectangle((0, 0), size, size, facecolor='lightblue'))
    
    # If receptive field is specified, draw it
    if rf_size is not None and center is not None:
        cx, cy = center
        rf_start = (cx - rf_size/2, cy - rf_size/2)
        ax.add_patch(Rectangle(rf_start, rf_size, rf_size, 
                             facecolor='pink', alpha=0.5, 
                             edgecolor='red'))
        # Draw center point
        ax.plot(cx, cy, 'ro')

    ax.set_xlim(-size*0.1, size*1.1)
    ax.set_ylim(-size*0.1, size*1.1)
    ax.set_aspect('equal')
    if title:
        ax.set_title(title)
    ax.axis('off')

def main():
    # Get model layer parameters
    kernel_sizes, strides, paddings, layer_names = get_model_layers()
    
    # Calculate receptive fields
    rf_sizes = calculate_receptive_field(kernel_sizes, strides, paddings)
    
    # Calculate feature map sizes
    feature_map_sizes = [224]  # Start with input size
    size = 224
    for i, stride in enumerate(strides):
        size = size // stride
        feature_map_sizes.append(size)
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Create stage information for plotting
    stages = []
    for i in range(len(layer_names)):
        stages.append((
            feature_map_sizes[i+1],  # Feature map size
            rf_sizes[i+1],           # Receptive field size
            f"{layer_names[i]}\n({kernel_sizes[i]}x{kernel_sizes[i]} {'conv' if i%3==0 else 'dwconv'}, stride {strides[i]})"
        ))
    
    # Add input stage
    stages.insert(0, (224, 1, "Input"))
    
    # Create subplot for each stage
    n_stages = len(stages)
    n_cols = 5
    n_rows = (n_stages + n_cols - 1) // n_cols
    
    for idx, (fmap_size, rf_size, title) in enumerate(stages):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        
        # Center the receptive field in the feature map
        center = (fmap_size/2, fmap_size/2)
        
        draw_feature_map(ax, fmap_size, rf_size, center, title)
        
        # Add size annotations
        ax.text(fmap_size/2, -fmap_size*0.05, 
                f'Feature Map: {fmap_size}x{fmap_size}\nRF size: {rf_size}x{rf_size}', 
                ha='center', va='top')
    
    # Add formula as a text annotation
    formula = r"$r_0 = \sum_{i=1}^{L} \left[(k_i - 1) \prod_{j=1}^{i-1} s_j \right] + 1$"
    fig.text(0.5, 0.01, formula, ha='center', fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)  # Make room for the formula
    plt.show()

if __name__ == "__main__":
    main()
