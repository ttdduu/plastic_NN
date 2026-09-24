import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class SimpleConvNet(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.features = nn.Sequential()
        in_ch = config.in_channels
        
        for i, out_ch in enumerate(config.conv_layers):
            self.features.add_module(f'conv{i+1}', nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            self.features.add_module(f'bn{i+1}', nn.BatchNorm2d(out_ch))
            self.features.add_module(f'relu{i+1}', nn.ReLU(inplace=True))
            if i % 2 == 1:
                self.features.add_module(f'pool{i//2+1}', nn.MaxPool2d(2, 2))
            in_ch = out_ch

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential()
        fc_in = config.conv_layers[-1] * 7 * 7
        for i, fc_size in enumerate(config.fc_layers):
            self.classifier.add_module(f'fc{i+1}', nn.Linear(fc_in, fc_size))
            self.classifier.add_module(f'fc_relu{i+1}', nn.ReLU(inplace=True))
            self.classifier.add_module(f'dropout{i+1}', nn.Dropout(config.dropout_rate))
            fc_in = fc_size
        self.classifier.add_module('fc_out', nn.Linear(fc_in, config.num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        
    def get_model_specific_config(self):
        return {
            'in_channels': self.config.in_channels,
            'conv_layers': self.config.conv_layers,
            'fc_layers': self.config.fc_layers,
            'dropout_rate': self.config.dropout_rate,
            'num_classes': self.config.num_classes
        }

# Example usage:
# model = SimpleConvNet(num_classes=10)
# input_tensor = torch.randn(1, 3, 200, 200)
# output = model(input_tensor)
