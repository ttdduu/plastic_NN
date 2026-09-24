import os
import glob
import cv2
import numpy as np
from collections import defaultdict
import re
import yaml

def extract_params_from_path(path):
    # Extract run directory which contains the parameters in its name
    run_dir = path.split('/wandb/')[1].split('/')[0]
    
    # Extract radius and sharpness from config file
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(path)))), 'config.yaml')
    
    if not os.path.exists(config_path):
        print(f"Config file not found at: {config_path}")
        return None, None
        
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
            # Extract values from the YAML structure
            radius = config.get('scotoma_radius', {}).get('value')
            sharpness = config.get('scotoma_sharpness', {}).get('value')
            
            if radius is not None and sharpness is not None:
                return float(radius), float(sharpness)
            else:
                print(f"Missing parameters in config file: radius={radius}, sharpness={sharpness}")
                return None, None
                
    except Exception as e:
        print(f"Error reading config file: {str(e)}")
        return None, None

def create_videos(experiment_type):
    base_dir = f'/home/tomasdu/repos/experiments/plastic_NNs/active/gradcam_tests/{experiment_type}/wandb'
    
    # Dictionary to store images by dataset, class, radius, and sharpness
    images_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    # Collect and organize all images
    for run_dir in glob.glob(os.path.join(base_dir, 'offline-run-*')):
        gradcam_dir = os.path.join(run_dir, 'files', 'extras', 'gradcam')
        if not os.path.exists(gradcam_dir):
            continue
            
        for dataset_dir in os.listdir(gradcam_dir):
            dataset_path = os.path.join(gradcam_dir, dataset_dir)
            if not os.path.isdir(dataset_path):
                continue
                
            for img_path in glob.glob(os.path.join(dataset_path, '*.png')):
                try:
                    # Extract class number from filename
                    class_match = re.search(r'class_(\d+)\.png', img_path)
                    if not class_match:
                        print(f"Could not extract class number from filename: {img_path}")
                        continue
                        
                    class_num = int(class_match.group(1))
                    
                    # Extract radius and sharpness
                    params = extract_params_from_path(img_path)
                    if params[0] is None or params[1] is None:
                        print(f"Skipping {img_path} due to missing parameters")
                        continue
                        
                    radius, sharpness = params
                    
                    # Read image
                    img = cv2.imread(img_path)
                    if img is not None:
                        images_dict[dataset_dir][class_num][radius][sharpness] = img
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    print(f"Error type: {type(e)}")
                    import traceback
                    traceback.print_exc()
    
    # Create videos for each dataset and class
    for dataset in images_dict:
        for class_num in images_dict[dataset]:
            output_path = f'/home/tomasdu/repos/experiments/plastic_NNs/active/gradcam_tests/{experiment_type}_{dataset}_class{class_num}.mp4'
            
            # Get first image to determine dimensions
            first_radius = next(iter(images_dict[dataset][class_num]))
            first_sharpness = next(iter(images_dict[dataset][class_num][first_radius]))
            first_img = images_dict[dataset][class_num][first_radius][first_sharpness]
            
            height, width = first_img.shape[:2]
            
            # Resize dimensions for smaller video
            target_width, target_height = width // 2, height // 2
            
            # Create video writer with lower frame rate and smaller resolution
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 1.0, (target_width, target_height))  # 1 FPS
            
            # Sort radii and sharpnesses
            radii = sorted(images_dict[dataset][class_num].keys())
            
            for radius in radii:
                sharpnesses = sorted(images_dict[dataset][class_num][radius].keys())
                for sharpness in sharpnesses:
                    img = images_dict[dataset][class_num][radius][sharpness]
                    
                    # Resize image
                    img_resized = cv2.resize(img, (target_width, target_height))
                    
                    # Add text with parameters
                    text_img = img_resized.copy()
                    cv2.putText(text_img, f'Radius: {radius}, Sharpness: {sharpness:.3f}', 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Write frame multiple times for longer display
                    for _ in range(2):  # Adjust this number to change display duration
                        out.write(text_img)
            
            out.release()
            print(f"Created video: {output_path}")

if __name__ == "__main__":
    for exp_type in ['AS-S', 'LH-S', 'LH-H']:
        print(f"Processing experiment: {exp_type}")
        create_videos(exp_type)