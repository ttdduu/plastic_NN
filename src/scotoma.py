import torch
import torch.nn.functional as F
import math

class ScotomaApplier:
    def __init__(self, config):
        self.config = config

    def _convert_radius_to_pixels(self, radius, image_width):
        """Convert radius from percentage to pixels"""
        return (radius/100) * image_width
        
    def apply_scotoma(self, images, center=None, radius=None, method=None, strength=None, sharpness=None):
        # Use config defaults if parameters not provided
        radius = radius if radius is not None else self.config.data.scotoma_radius
        strength = strength if strength is not None else self.config.data.scotoma_strength
        sharpness = sharpness if sharpness is not None else self.config.data.scotoma_sharpness
        method = method if method is not None else self.config.data.scotoma_method
        
        # Edge case: if radius is 0, return original images (no scotoma)
        if radius == 0:
            return images
        
        B, C, H, W = images.shape
        if center is None:
            center = (W // 2, H // 2)

        # Create distance matrix
        y, x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=images.device),
            torch.arange(W, dtype=torch.float32, device=images.device)
        )
        distance = torch.sqrt((x - center[0])**2 + (y - center[1])**2)  # Circular (Euclidean)
        # distance = torch.max(torch.abs(x - center[0]), torch.abs(y - center[1]))  # Square (Chebyshev)
        
        # Convert radius to pixels
        radius_pixels = self._convert_radius_to_pixels(radius, W)
        
        if method == 'nice':
            sharpness=1000
            
            # Raw sigmoid masks (no normalization - both should have exact 0 and 1 values with high sharpness)
            mask_center_zero = torch.sigmoid(sharpness*(distance - radius_pixels))  # 0 at center, 1 at periphery
            mask_center_one = 1 - mask_center_zero  # 1 at center, 0 at periphery
            
            # CENTRAL SCOTOMA (macular degeneration): mask the center, preserve periphery
            # Use mask_center_zero: multiplies center by 0, periphery by 1
            
            # PERIPHERAL SCOTOMA: preserve center, mask the periphery  
            # Use mask_center_one: multiplies center by 1, periphery by 0
            
            # Toggle:
            mask = mask_center_zero  # CENTRAL scotoma (original behavior)
            # mask = mask_center_one  # PERIPHERAL scotoma
            
            mask = mask.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
            scotomized_images = images * mask  # Black mask
            
        return scotomized_images
        # return torch.zeros_like(images) # DEBUG
