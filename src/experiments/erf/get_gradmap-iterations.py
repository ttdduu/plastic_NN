import torch
from src.experiments.erf.video_model import VideoNeuralModel
import os
import matplotlib.pyplot as plt
import numpy as np
from src.experiments.erf.convnext_atto_lc_new import ConvNeXtAttoLC
from src.experiments.erf.convnext_atto_new import ConvNeXtAtto
from types import SimpleNamespace

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_gradmap(weights_path, save_dir, model_is, radius,gradmap_iters=50):

    config = SimpleNamespace()
    config.pretrained = False
    config.num_classes = 8
    config.scotoma_apply = False  # For training
    config.scotoma_apply_val = False  # For validation

    # Initialize model first
    if model_is == 'lc':
        base_model = ConvNeXtAttoLC(config, weights_path=weights_path)
    elif model_is == 'og':
        base_model = ConvNeXtAtto(config, weights_path=weights_path)

    base_model = base_model.to(device)
    
    # Create video model wrapper
    spatial_resolution = 224
    video_model = VideoNeuralModel(
        base_model=base_model,
        spatial_resol=spatial_resolution,
        temporal_window_size=1,
        out_neurons=8
    )

    # Don't try to load state dict into video_model
    # The base_model already has the weights loaded
    video_model = video_model.to(device)
    video_model.eval()
    
    # Get gradmaps for all iterations
    gradmaps = video_model.STRF_gradmap(T=1, gradmap_iters=gradmap_iters)  # List of [B=8, C=3, H, W] tensors
    
    # Create directory for frames
    frames_dir = os.path.join(save_dir, f'gradmap_frames_{model_is}_r{radius}')
    os.makedirs(frames_dir, exist_ok=True)
    
    classes = ['apple', 'car', 'cow', 'cup', 'dog', 'horse', 'pear', 'tomato']
    
    # For each iteration, create and save a figure
    for iter_idx, gradmap in enumerate(gradmaps):
        # Convert gradmap to numpy
        gradmap_np = gradmap.detach().cpu().numpy()  # [8, 3, H, W]
        
        # Create a figure with subplots for each neuron
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        # Plot each neuron's gradmap
        for i in range(8):
            # Take mean across channels for visualization
            neuron_gradmap = np.mean(gradmap_np[i], axis=0)  # [H, W]
            
            # Normalize the gradmap to make patterns more visible
            neuron_gradmap = (neuron_gradmap - neuron_gradmap.min()) / (neuron_gradmap.max() - neuron_gradmap.min() + 1e-8)
            
            im = axes[i].imshow(neuron_gradmap, cmap='RdBu_r', vmin=0, vmax=1)
            fig.suptitle(f'Gradmap for each output neuron on the {model_is} model (r={radius})\nIteration {iter_idx+1}/{gradmap_iters}')
            axes[i].set_title(f'{classes[i]}')
            axes[i].axis('off')
        
        # Add a colorbar
        plt.colorbar(im, ax=axes.ravel().tolist())
        
        # Save frame
        frame_path = os.path.join(frames_dir, f'frame_{iter_idx:04d}.png')
        plt.savefig(frame_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Create video from frames using ffmpeg (if available)
    try:
        import subprocess
        video_path = os.path.join(save_dir, f'gradmap_video_{model_is}_r{radius}.mp4')
        cmd = [
            'ffmpeg', '-y', '-framerate', '8', 
            '-i', os.path.join(frames_dir, 'frame_%04d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', 
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # Ensure dimensions are even
            video_path
        ]
        subprocess.run(cmd)
        print(f"Video saved to {video_path}")
    except Exception as e:
        print(f"Could not create video: {e}")
        print(f"Individual frames saved to {frames_dir}")
    
    # Still save the final gradmap as before
    final_gradmap = gradmaps[-1]
    final_gradmap_np = final_gradmap.detach().cpu().numpy()
    
    # Create a figure with subplots for the final gradmap
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # Plot each neuron's gradmap
    for i in range(8):
        neuron_gradmap = np.mean(final_gradmap_np[i], axis=0)
        neuron_gradmap = (neuron_gradmap - neuron_gradmap.min()) / (neuron_gradmap.max() - neuron_gradmap.min() + 1e-8)
        
        im = axes[i].imshow(neuron_gradmap, cmap='RdBu_r', vmin=0, vmax=1)
        fig.suptitle(f'Final Gradmap for each output neuron on the {model_is} model finetuned with radius {radius}. {gradmap_iters} gradmap iterations.')
        axes[i].set_title(f'{classes[i]}')
        axes[i].axis('off')
    
    # Add a colorbar
    plt.colorbar(im, ax=axes.ravel().tolist())
    
    # Save as PNG and SVG
    plt.savefig(os.path.join(save_dir, f'gradmaps-{model_is}-r{radius}-{gradmap_iters}iters.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, f'gradmaps-{model_is}-r{radius}-{gradmap_iters}iters.svg'), format='svg', bbox_inches='tight')
    print(f"Final gradmaps saved to {save_dir}")
    
    return gradmaps[-1]  # Return the final gradmap for compatibility

if __name__ == "__main__":

    lc_models = [
       '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_182928-nzhez322/files/model',
       '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_184534-jscw2ie2/files/model',
       '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_190137-oamvro1o/files/model',
       '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_191741-aaha68yt/files/model',
       '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_193345-ps00a7p6/files/model',
       '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_194957-iu6zyye9/files/model',
       '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_200612-et6t50zz/files/model',
       '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_202226-twjf2x9l/files/model',
       '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_203841-p8nwysug/files/model',
       '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_205500-i8ty2jwt/files/model',
       '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_181315-uh2g47yq/files/model',
    ]

    og_models = [
        '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/WS-S/wandb/offline-run-20250409_160853-z7hirwq2/files/model',
        '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/WS-S/wandb/offline-run-20250409_161626-syihi52v/files/model',
        '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/WS-S/wandb/offline-run-20250409_162401-5y4t5xdv/files/model',
        '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/WS-S/wandb/offline-run-20250409_163134-z895rj3n/files/model',
        '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/WS-S/wandb/offline-run-20250409_163900-ld64on63/files/model',
        '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/WS-S/wandb/offline-run-20250409_164624-kmo0uoxj/files/model',
        '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/WS-S/wandb/offline-run-20250409_165349-luq143u2/files/model',
        '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/WS-S/wandb/offline-run-20250409_170007-zxmfkbrg/files/model',
        '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/WS-S/wandb/offline-run-20250409_170713-8vd6zhg4/files/model',
        '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/WS-S/wandb/offline-run-20250409_171457-k9z6d2tg/files/model',
        '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/WS-S/wandb/offline-run-20250409_172233-qpf6skuf/files/model',
    ]

    radii = [0,5,10,15,20,21,22,23,24,25,30]

    lc_save_dir = '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/gradmaps'
    os.makedirs(lc_save_dir, exist_ok=True)
    og_save_dir = '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/gradmaps'
    os.makedirs(og_save_dir, exist_ok=True)

    def loopit(models,model_is,save_dir,radii,gradmap_iters=50):
        for i, model_dir in enumerate(models):
            weights_path = os.path.join(model_dir, 'best_model.pth')
            get_gradmap(weights_path=weights_path, save_dir=save_dir, model_is=model_is,radius=radii[i],gradmap_iters=gradmap_iters)

    gradmap_iters = 53

    model_is = 'og'
    loopit(og_models, model_is,og_save_dir,radii,gradmap_iters)
    model_is = 'lc'
    loopit(lc_models, model_is=model_is, save_dir=lc_save_dir, radii=radii,gradmap_iters=gradmap_iters)