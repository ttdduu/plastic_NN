import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import os

def plot_layers(model, save_path):
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
            layers.append((name, module))

    fig, ax = plt.subplots(figsize=(10, len(layers)))
    ax.set_ylim(0, len(layers))
    ax.set_xlim(0, 1)
    ax.axis('off')

    for i, (name, layer) in enumerate(layers):
        if isinstance(layer, nn.Conv2d):
            color = 'lightblue'
            label = f'Conv2d: {layer.in_channels}x{layer.out_channels}x{layer.kernel_size[0]}x{layer.kernel_size[1]}'
        elif isinstance(layer, nn.Linear):
            color = 'lightgreen'
            label = f'Linear: {layer.in_features}x{layer.out_features}'
        elif isinstance(layer, nn.MaxPool2d):
            color = 'lightyellow'
            label = f'MaxPool2d: {layer.kernel_size}x{layer.kernel_size}'
        elif isinstance(layer, nn.AdaptiveAvgPool2d):
            color = 'lightpink'
            label = f'AdaptiveAvgPool2d: {layer.output_size}x{layer.output_size}'

        ax.add_patch(plt.Rectangle((0.1, len(layers)-i-0.9), 0.8, 0.8, fill=True, color=color))
        ax.text(0.5, len(layers)-i-0.5, label, ha='center', va='center', wrap=True)

    plt.title('Network Architecture')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    
