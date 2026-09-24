import numpy as np
import torch


def infer_input_size_from_checkpoint(weights_path: str, fallback: int = 224) -> int:
    """Infer model input size from a checkpoint's stage-0 LC positions dimension."""
    try:
        ckpt = torch.load(weights_path, map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt)
        key_candidates = [
            'stages.0.0.dwconv.weights',
            'stages.0.0.dwconv.weight',
        ]
        key = None
        for k in state.keys():
            if any(k.endswith(cand) for cand in key_candidates):
                key = k
                break
        if key is None:
            for k in state.keys():
                if 'dwconv.weights' in k:
                    key = k
                    break
        if key is None:
            return fallback

        w = state[key]
        if not hasattr(w, 'shape') or len(w.shape) < 2:
            return fallback
        positions = int(w.shape[1])
        h0 = int(round(positions ** 0.5))
        return h0 + 3  # Stem is Conv2d k=4, s=1, p=0 -> H_out = H_in - 3
    except Exception:
        return fallback


def extract_nonzero_patch(gradmap_tensor, min_threshold=1e-6):
    """Extract the nonzero patch from a gradmap tensor and its bounding box."""
    if torch.is_tensor(gradmap_tensor):
        gradmap_np = gradmap_tensor.detach().cpu().numpy()
    else:
        gradmap_np = gradmap_tensor
    if isinstance(gradmap_np, np.ndarray) and gradmap_np.ndim == 3 and gradmap_np.shape[0] == 1:
        gradmap_np = gradmap_np.squeeze(0)

    nonzero_coords = np.where(np.abs(gradmap_np) > min_threshold)
    if len(nonzero_coords[0]) == 0:
        return gradmap_np, (0, 0, gradmap_np.shape[0], gradmap_np.shape[1])

    y_min, y_max = np.min(nonzero_coords[0]), np.max(nonzero_coords[0])
    x_min, x_max = np.min(nonzero_coords[1]), np.max(nonzero_coords[1])
    patch = gradmap_np[y_min:y_max + 1, x_min:x_max + 1]

    return patch, (y_min, x_min, y_max + 1, x_max + 1)

