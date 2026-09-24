import torch
from typing import Dict
def _load_checkpoint(
    path: str, device: torch.device, *, allow_pickle: bool = False
) -> Dict[str, torch.Tensor]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        if not allow_pickle:
            raise
        ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            return ckpt["model_state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        return ckpt
    raise ValueError(f"Unsupported checkpoint format at {path}")
