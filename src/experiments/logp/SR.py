from super_image import DrlnModel, ImageLoader
from PIL import Image
import torch
import torch.nn.functional as F

# Setup device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# path = '/home/ttdduu/sample_datasets/mini_net/train/n02165456/n02165456_298.JPEG'
# path = '/home/ttdduu/sample_datasets/imagenet_centered-2026-diff_0.7/train/n01537544/n01537544_1222.JPEG'
path = '/home/ttdduu/sample_datasets/imagenet_centered-2026-diff_0.7/train/n02280649/n02280649_995.JPEG'
image = Image.open(path)

scale = 4
model = DrlnModel.from_pretrained('eugenesiow/drln', scale=scale)
model = model.to(device)
model.eval()

inputs = ImageLoader.load_image(image).to(device)

with torch.no_grad():
    preds = model(inputs)

ImageLoader.save_image(preds, f'./scaled_{scale}x.png')
ImageLoader.save_compare(inputs, preds, f'./scaled_{scale}x_compare.png')
