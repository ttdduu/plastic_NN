import matplotlib.pyplot as plt
from types import SimpleNamespace
from src.experiments.erf.convnext_atto_lc import ConvNeXtAttoLC

config = SimpleNamespace()
config.pretrained = False
config.num_classes = 8

weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/LH-S/wandb/offline-run-20250408_190942-exa7jf7z/files/model/best_model.pth"
model = ConvNeXtAttoLC(config, weights_path=weights_path)

def get_kernel(lc_layer, channel, x, y, visualize=True):
    # Convert (x,y) to flattened index in 56x56 grid, bc weights.shape = torch.Size([40, 3136, 49, 1])
    spatial_idx = y * 56 + x
    
    # Get the 7x7 kernel for this channel and location
    kernel = lc_layer.weights.data[channel, spatial_idx, :, 0].view(7, 7)
    
    if visualize:
        plt.figure(figsize=(5,5))
        plt.imshow(kernel.cpu().numpy(), cmap='bwr')
        plt.colorbar()
        plt.title(f'Kernel for Channel {channel} at position ({x},{y})')
        plt.show()
    
    return kernel


# example
first_block = model.stages[0][0]
lc_layer = first_block.dwconv
weights = lc_layer.weights.data # >>> weights.shape torch.Size([40, 3136, 49, 1]) this is the shape after matmul
channel = 19
x,y = 28,28 # center

# i could loop over all channels and positions
kernel = get_kernel(lc_layer, channel, x, y, visualize=True)