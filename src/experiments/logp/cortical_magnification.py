import torch
import torch.nn as nn
from torchvision.models import resnet50
import numpy as np
from utils.PolarTransform import PolarTransform

def resnet50_cortical_magnification():
    img_size = 32
    out_size = 16
    input_h, input_w = img_size, img_size
    output_h, output_w = out_size, out_size
    foveal_radius = 12
    max_dist = np.sqrt((input_h/2)**2 + (input_w/2)**2)
    foveal_dists = np.arange(0, foveal_radius)
    periphery_dists = [12, 13, 14, 16, 18, 20, 23]
    radius_bins = list(foveal_dists) + list(periphery_dists)
    angle_bins = list(np.linspace(0, 2*np.pi, img_size+1-2))
    lpp = PolarTransform(input_h, input_w, output_h, output_w, radius_bins, angle_bins, interpolation='linear', subbatch_size=32)
    pretrained_model = resnet50()
    model = torch.nn.Sequential(
        pretrained_model.conv1,
        pretrained_model.bn1,
        pretrained_model.relu,
        lpp,
        pretrained_model.layer1,
        pretrained_model.layer2,
        pretrained_model.layer3,
        pretrained_model.layer4,
        pretrained_model.avgpool,
        torch.nn.Flatten(1),
        pretrained_model.fc
    )
    return model