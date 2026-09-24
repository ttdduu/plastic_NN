import torch
from src.experiments.erf.video_model import VideoNeuralModel
import os
import matplotlib.pyplot as plt
import numpy as np
from src.experiments.erf.convnext_atto_lc_new import ConvNeXtAttoLC
from src.experiments.erf.convnext_atto_new import ConvNeXtAtto
from types import SimpleNamespace
from src.experiments.erf.convnext_atto_inner_layer import ConvNeXtAttoInnerLayer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_gradmap(weights_path, save_dir, model_is, radius, inner_layer_params=None):

    config = SimpleNamespace()
    config.pretrained = False
    config.num_classes = 8
    config.scotoma_apply = False  
    config.scotoma_apply_val = False 

    if model_is == 'lc':
        base_model = ConvNeXtAttoLC(config, weights_path=weights_path)
    elif model_is == 'og':
        base_model = ConvNeXtAtto(config, weights_path=weights_path)
    elif model_is == 'inner':
        target_stage = 0
        target_block = 0
        target_layer = 'dwconv'
        
        if inner_layer_params:
            target_stage = inner_layer_params.get('stage', 0)
            target_block = inner_layer_params.get('block', 0)
            target_layer = inner_layer_params.get('layer', 'dwconv')
            
        base_model = ConvNeXtAttoInnerLayer(
            config, 
            weights_path=weights_path,
            target_stage=target_stage,
            target_block=target_block,
            target_layer=target_layer
        )

    base_model = base_model.to(device)
    
    spatial_resolution = 224
    video_model = VideoNeuralModel(
        base_model=base_model,
        spatial_resol=spatial_resolution,
        temporal_window_size=1,
        out_neurons=8
    )

    video_model = video_model.to(device)
    video_model.eval()
    
    # Get gradmap with a single pass
    gradmap = video_model.STRF_gradmap(T=1)  # [B=8, C=3, H, W] tensor
    
    gradmap_np = gradmap.detach().cpu().numpy()  # [8, 3, H, W]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    classes = ['apple', 'car', 'cow', 'cup', 'dog', 'horse', 'pear', 'tomato']
    
    for i in range(8):
        # Take mean across channels 
        neuron_gradmap = np.mean(gradmap_np[i], axis=0)  # [H, W]
        
        neuron_gradmap = (neuron_gradmap - neuron_gradmap.min()) / (neuron_gradmap.max() - neuron_gradmap.min() + 1e-8)
        
        im = axes[i].imshow(neuron_gradmap, cmap='RdBu_r', vmin=0, vmax=1)
        fig.suptitle(f'Gradmap for each output neuron on the {model_is} model (r={radius})')
        axes[i].set_title(f'{classes[i]}')
        axes[i].axis('off')
    
    # Add a colorbar
    plt.colorbar(im, ax=axes.ravel().tolist())
    
    # plt.savefig(os.path.join(save_dir, f'gradmaps-{model_is}-r{radius}.png'), dpi=300, bbox_inches='tight')
    # plt.savefig(os.path.join(save_dir, f'gradmaps-{model_is}-r{radius}.svg'), format='svg', bbox_inches='tight')
    # print(f"Gradmaps saved to {save_dir}")
    
    return gradmap  

if __name__ == "__main__":

    lc_models = [
       #'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_182928-nzhez322/files/model',
       #'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_184534-jscw2ie2/files/model',
       #'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_190137-oamvro1o/files/model',
       #'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_191741-aaha68yt/files/model',
       '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_193345-ps00a7p6/files/model',
       #'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_194957-iu6zyye9/files/model',
       #'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_200612-et6t50zz/files/model',
       #'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_202226-twjf2x9l/files/model',
       #'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_203841-p8nwysug/files/model',
       #'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_205500-i8ty2jwt/files/model',
       #'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_181315-uh2g47yq/files/model',
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

    lc_save_dir = '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/gradmaps/4th_run'
    #os.makedirs(lc_save_dir, exist_ok=True)
    og_save_dir = '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/gradmaps/4th_run'
    #os.makedirs(og_save_dir, exist_ok=True)

    def loopit(models, model_is, save_dir, radii):
        for i, model_dir in enumerate(models):
            weights_path = os.path.join(model_dir, 'best_model.pth')
            get_gradmap(weights_path=weights_path, save_dir=save_dir, model_is=model_is, radius=radii[i])

    model_is = 'og'
    #loopit(og_models, model_is, og_save_dir, radii)
    model_is = 'lc'
    loopit(lc_models, model_is=model_is, save_dir=lc_save_dir, radii=radii)

    def loopit_inner_layer(models, save_dir, radii):
        for i, model_dir in enumerate(models):
            weights_path = os.path.join(model_dir, 'best_model.pth')
            # Example: visualize first stage, first block, dwconv layer
            inner_params = {'stage': 0, 'block': 0, 'layer': 'dwconv'}
            get_gradmap(
                weights_path=weights_path, 
                save_dir=save_dir, 
                model_is='inner', 
                radius=radii[i],
                inner_layer_params=inner_params
            )