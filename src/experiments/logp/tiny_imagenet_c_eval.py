import sys
sys.path.append('..')

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import resnet50
from models.center_surround import resnet50_center_surround
from models.local_rf import resnet50_local_rf
from models.cortical_magnification import resnet50_cortical_magnification
from models.composite import resnet50_div_norm, resnet50_tuned_norm, resnet50_composite_model


def evaluate(model, dataloader, criterion, device='cpu'):
    model.eval()
    criterion.to(device)
    running_nsamples, running_loss, running_correct = 0, 0, 0
    y_true, y_hat = [], []
    with torch.no_grad():
        for img_batch, label_batch in dataloader:
            img_batch = img_batch.to(device)
            label_batch = label_batch.to(device)
            output = model(img_batch)
            loss = criterion(output, label_batch)
            running_nsamples += len(img_batch)
            running_loss += (loss.item()*len(img_batch))
            predictions = torch.argmax(output, dim=-1)
            running_correct += torch.sum(predictions == label_batch).item()
            y_true.append(label_batch.cpu().numpy())
            y_hat.append(predictions.cpu().numpy())
    avg_loss = running_loss/running_nsamples
    acc = running_correct/running_nsamples
    print('-'*100)
    print('Evaluation:\t loss: {:.4f}\t accuracy: {:.4f}'.format(avg_loss, acc))
    print('-'*100)
    return np.hstack(y_true), np.hstack(y_hat), avg_loss, acc


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['tinyimagenet-c'], \
                    type=str, required=True, help='Dataset for evaluation')
parser.add_argument('--data_root', type=str, required=True, help='Path to data directory')
parser.add_argument('--model_type', choices=['resnet50', 'resnet50_center_surround', 'resnet50_local_rf', \
                                             'resnet50_div_norm', 'resnet50_tuned_norm', 'resnet50_cortical_magnification', \
                                             'resnet50_composite_a', 'resnet50_composite_b', 'resnet50_composite_c', \
                                             'resnet50_composite_d', 'resnet50_composite_e', 'resnet50_composite_f', \
                                             'resnet50_composite_g'], \
                    type=str, required=True, help='Model architecture')
parser.add_argument('--model_path', type=str, required=True, \
                    help='Path to saved model file')
parser.add_argument('--output_root', type=str, required=True, \
                    help='Path to directory where artifacts will be saved')
parser.add_argument('--batch_size', default=128, type=int, required=False, \
                    help='Data batch size')
parser.add_argument('--num_workers', default=16, type=int, required=False, help='Dataloader num_workers')                  
parser.add_argument('--device', default='cpu', type=str, required=False, help='Device to use for training/testing')
args = parser.parse_args()



torch.set_float32_matmul_precision('medium')

if __name__ == '__main__':
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    if args.model_type == 'resnet50':
        model = resnet50()
    elif args.model_type == 'resnet50_center_surround':
        # Center Surround only
        model = resnet50_center_surround()
    elif args.model_type == 'resnet50_local_rf':
        # Local RF only
        model = resnet50_local_rf()
    elif args.model_type == 'resnet50_div_norm':
        # Divisive normalization only
        model = resnet50_div_norm()
    elif args.model_type == 'resnet50_tuned_norm':
        # Tuned normalization only
        model = resnet50_tuned_norm()
    elif args.model_type == 'resnet50_cortical_magnification':
        # Cortical Magnification only
        model = resnet50_cortical_magnification()
    elif args.model_type == 'resnet50_composite_a':
        # All components (Center Surround, Local RF, Tuned Norm., Cortical Mag.)
        model = resnet50_composite_model('a')
    elif args.model_type == 'resnet50_composite_b':
        # Local RF, Tuned Norm., Cortical Mag.
        model = resnet50_composite_model('b')
    elif args.model_type == 'resnet50_composite_c':
        # Center Surround, Local RF, Cortical Mag.
        model = resnet50_composite_model('c')
    elif args.model_type == 'resnet50_composite_d':
        # Center Surround, Local RF, Tuned Norm.
        model = resnet50_composite_model('d')
    elif args.model_type == 'resnet50_composite_e':
        # Tuned Norm., Cortical Mag.
        model = resnet50_composite_model('e')
    elif args.model_type == 'resnet50_composite_f':
        # Local RF, Cortical Mag.
        model = resnet50_composite_model('f')
    elif args.model_type == 'resnet50_composite_g':
        # Local RF, Tuned Norm.
        model = resnet50_composite_model('g')

    if args.model_type == 'resnet50_polar':
        model[-1] = torch.nn.Linear(2048, 200)
    else:
        model.fc = torch.nn.Linear(2048, 200)

    if args.model_path is not None:
        print('Loading model from checkpoint')
        checkpoint = torch.load(args.model_path, map_location='cpu')
        val_acc = checkpoint['valid_acc']
        model.load_state_dict(checkpoint['model'])
        print('Tiny imagenet Clean Val Acc at checkpoint: {}'.format(val_acc))

    device = args.device
    model.to(device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    print('Evaluating...')
    if args.dataset == 'tinyimagenet-c':
        results = []
        for corruption_type in tqdm(['brightness', 'defocus_blur', 'fog', 'gaussian_noise', 'impulse_noise', \
                                     'motion_blur', 'shot_noise', 'zoom_blur', 'contrast', 'elastic_transform', \
                                     'frost', 'glass_blur', 'jpeg_compression', 'pixelate', 'snow']):
            for difficulty in [1, 2, 3, 4, 5]:
                valid_dataset = ImageFolder(os.path.join(args.data_root, corruption_type, str(difficulty)), transform=valid_transform)
                valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
                print((corruption_type, difficulty))
                y_true_valid, y_hat_valid, valid_loss, valid_acc = evaluate(model, valid_dataloader, criterion, device)
                results.append([corruption_type, difficulty, valid_acc])
        df = pd.DataFrame(np.array(results), columns=['corruption_type', 'difficulty', 'valid_acc'])
        df.to_csv(os.path.join(args.output_root, 'tinyimagenetc_results.csv'), index=False)