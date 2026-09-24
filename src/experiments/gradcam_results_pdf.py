import os
from PIL import Image
import math
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import glob

def create_gradcam_pdf(experiment_type):
    # Base directory where the wandb runs are
    base_dir = f'/home/tomasdu/repos/experiments/plastic_NNs/active/gradcam_tests/correcting_WS-S/{experiment_type}/wandb'
    
    # Output PDF path
    output_pdf = f'/home/tomasdu/repos/experiments/plastic_NNs/active/gradcam_tests/correcting_WS-S/{experiment_type}_gradcam_results.pdf'
    
    # Collect all image paths
    image_paths = []
    for run_dir in glob.glob(os.path.join(base_dir, 'offline-run-*')):
        gradcam_dir = os.path.join(run_dir, 'files', 'extras', 'gradcam')
        if os.path.exists(gradcam_dir):
            for dataset_dir in os.listdir(gradcam_dir):
                dataset_path = os.path.join(gradcam_dir, dataset_dir)
                if os.path.isdir(dataset_path):
                    image_paths.extend(glob.glob(os.path.join(dataset_path, '*.png')))
    
    if not image_paths:
        print(f"No images found for {experiment_type}")
        return
        
    # Sort images for consistent ordering
    image_paths.sort()
    
    # Create PDF with landscape orientation (swap width and height)
    c = canvas.Canvas(output_pdf, pagesize=(A4[1], A4[0]))  # Swap for landscape
    
    # Define grid
    cols = 2  # 2 columns
    images_per_page = 4  # 2 images per row, 2 rows
    rows = images_per_page // cols
    
    # Calculate dimensions
    page_width, page_height = A4[1], A4[0]  # Using swapped dimensions
    margin = 10  # Minimal margin
    column_spacing = 5  # Minimal spacing between columns
    row_spacing = 5  # Minimal spacing between rows
    
    # Calculate available space
    available_width = page_width - (2 * margin) - column_spacing
    available_height = page_height - (2 * margin) - row_spacing
    
    # Calculate image dimensions
    image_width = (available_width - column_spacing) / cols
    image_height = (available_height - row_spacing) / rows
    
    # Process images
    for i, img_path in enumerate(image_paths):
        if i % images_per_page == 0 and i > 0:
            c.showPage()
        
        print(f"Processing image {i+1}/{len(image_paths)}: {img_path}")
        
        # Calculate position in grid
        row = (i % images_per_page) // cols
        col = (i % images_per_page) % cols
        
        # Calculate position on page with minimal spacing
        x = margin + col * (image_width + column_spacing)
        y = page_height - (margin + (row + 1) * image_height + row * row_spacing)
        
        try:
            # Draw image
            c.drawImage(img_path, x, y, width=image_width, height=image_height, preserveAspectRatio=True)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    # Save PDF
    c.save()
    print(f"Created PDF: {output_pdf}")

if __name__ == "__main__":
    # Process each experiment type
    for exp_type in ['WS-S']:
        print(f"Starting PDF generation for {exp_type}")
        create_gradcam_pdf(exp_type)
        print(f"Finished PDF generation for {exp_type}")
