import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from super_image import DrlnModel

MAX_IMAGE_SIDE = 600
SR_SCALE = 4
SR_BATCH_SIZE = 8
SR_USE_FP16 = True
SR_ALLOW_TF32 = True
SR_SORT_BY_SIZE = True
RUN_CENTERING = False
RUN_SR = True
IMAGENET_ROOT = Path('/home/tomasdu/repos/datasets/imagenet-ILSVRC/ILSVRC')
CENTERED_DIR = Path('/home/tomasdu/repos/datasets/imagenet-centered')
SR_OUTPUT_DIR = Path('/home/tomasdu/repos/datasets/imagenet-sr-centered')

def get_class_from_xml(xml_path):
    """Extract class name from XML annotation."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objects = root.findall('object')
        if len(objects) > 0:
            return objects[0].find('name').text
        return None
    except Exception as e:
        print(f"Error getting class from {xml_path}: {e}")
        return None

def get_bbox_from_xml(xml_path):
    """Extract bounding box coordinates from XML (first object)."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get bounding box
        objects = root.findall('object')
        if len(objects) > 0:
            bbox = objects[0].find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            return (xmin, ymin, xmax, ymax)
        return None
    except Exception as e:
        print(f"Error getting bbox from {xml_path}: {e}")
        return None

def get_size_from_xml(xml_path):
    """Extract image size from XML."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        return width, height
    except Exception as e:
        print(f"Error getting size from {xml_path}: {e}")
        return None

def center_bbox_in_square(img, bbox):
    """
    Center the bounding box in a black square.
    Places the entire image in a black square such that the bbox center is at the square's center.

    Args:
        img: PIL Image (RGB)
        bbox: Tuple (xmin, ymin, xmax, ymax) in image coordinates

    Returns:
        PIL Image: Centered square image
    """
    try:
        xmin, ymin, xmax, ymax = bbox

        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        width, height = img.size

        # Calculate bbox center
        bbox_center_x = (xmin + xmax) / 2
        bbox_center_y = (ymin + ymax) / 2

        # Calculate distances from bbox center to image edges
        dist_left = bbox_center_x
        dist_right = width - bbox_center_x
        dist_top = bbox_center_y
        dist_bottom = height - bbox_center_y

        # Square size is 2x the max distance to ensure entire image fits
        square_size = int(2 * max(dist_left, dist_right, dist_top, dist_bottom))

        # Create a black square
        black_square = Image.new('RGB', (square_size, square_size), (0, 0, 0))

        # Calculate position to paste image so that bbox center aligns with square center
        square_center = square_size / 2
        paste_x = int(square_center - bbox_center_x)
        paste_y = int(square_center - bbox_center_y)

        # Paste the original image
        black_square.paste(img, (paste_x, paste_y))

        return black_square
    except Exception as e:
        print(f"Error centering image: {e}")
        return None


def process_image_center_only(img_path, xml_path, output_path):
    """
    Process a single image:
    1) Center using bbox (zero padding)
    2) Save centered image
    """
    try:
        bbox = get_bbox_from_xml(xml_path)
        if bbox is None:
            return False

        with Image.open(img_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            centered = center_bbox_in_square(img, bbox)

        if centered is None:
            return False
        centered.save(output_path, quality=95)
        return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

def pil_to_tensor(img):
    """Convert a PIL RGB image to a CHW float tensor in [0, 1]."""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor

def tensor_to_pil(tensor):
    """Convert a CHW float tensor in [0, 1] to a PIL Image."""
    tensor = tensor.clamp(0, 1)
    arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)

def process_sr_then_center_batch(image_paths, xml_paths, output_paths, model, device):
    """Super-resolve originals, then center using scaled bboxes."""
    tensors = []
    sizes = []
    bboxes = []
    valid_indices = []
    for idx, (img_path, xml_path) in enumerate(zip(image_paths, xml_paths)):
        bbox = get_bbox_from_xml(xml_path)
        if bbox is None:
            continue
        with Image.open(img_path) as img:
            tensor = pil_to_tensor(img)
            _, h, w = tensor.shape
            sizes.append((w, h))
            bboxes.append(bbox)
            tensors.append(tensor)
            valid_indices.append(idx)

    if not tensors:
        return

    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)
    pad_h = (SR_SCALE - max_h % SR_SCALE) % SR_SCALE
    pad_w = (SR_SCALE - max_w % SR_SCALE) % SR_SCALE
    batch_h = max_h + pad_h
    batch_w = max_w + pad_w

    padded = []
    for tensor in tensors:
        _, h, w = tensor.shape
        pad_right = batch_w - w
        pad_bottom = batch_h - h
        padded.append(F.pad(tensor, (0, pad_right, 0, pad_bottom), mode='constant', value=0))

    batch = torch.stack(padded).to(device, non_blocking=True)
    with torch.inference_mode():
        if device.type == 'cuda' and SR_USE_FP16:
            with torch.cuda.amp.autocast():
                upscaled = model(batch).cpu()
        else:
            upscaled = model(batch).cpu()

    for out_tensor, (w, h), bbox, original_idx in zip(
        upscaled, sizes, bboxes, valid_indices
    ):
        out_w = w * SR_SCALE
        out_h = h * SR_SCALE
        cropped = out_tensor[:, :out_h, :out_w]
        xmin, ymin, xmax, ymax = bbox
        bbox_up = (xmin * SR_SCALE, ymin * SR_SCALE, xmax * SR_SCALE, ymax * SR_SCALE)
        sr_pil = tensor_to_pil(cropped)
        centered = center_bbox_in_square(sr_pil, bbox_up)
        if centered is None:
            continue
        output_paths[original_idx].parent.mkdir(parents=True, exist_ok=True)
        centered.save(output_paths[original_idx], quality=95)

def super_resolve_then_center_dataset():
    output_dir = SR_OUTPUT_DIR

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.backends.cudnn.benchmark = True
        if SR_ALLOW_TF32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    else:
        print("Warning: running SR on CPU will be slow.")

    print(f"Loading super-resolution model (scale={SR_SCALE}x)...")
    try:
        model = DrlnModel.from_pretrained('eugenesiow/drln', scale=SR_SCALE)
        print("Loaded model")
    except Exception as e:
        print(f"Could not load model: {e}")
        raise

    model = model.to(device)
    model.eval()

    for split in ['train', 'val']:
        print(f"\nProcessing {split} set...")
        img_dir = IMAGENET_ROOT / 'Data/CLS-LOC' / split
        anno_dir = IMAGENET_ROOT / 'Annotations/CLS-LOC' / split

        output_split_dir = output_dir / split
        output_split_dir.mkdir(parents=True, exist_ok=True)

        if split == 'val':
            done_marker = output_split_dir / "__done__.txt"
            if done_marker.exists():
                print(f"Skipping {split}; done marker found at {done_marker}")
                continue
            pending = []
            for xml_file in tqdm(list(anno_dir.glob('*.xml')), desc="Collecting validation images"):
                if not check_bbox_requirements(xml_file):
                    continue
                class_id = get_class_from_xml(xml_file)
                if class_id is None:
                    continue
                img_name = f"{xml_file.stem}.JPEG"
                img_path = img_dir / img_name
                if img_path.exists():
                    out_path = output_split_dir / class_id / img_name
                    if not out_path.exists():
                        size = get_size_from_xml(xml_file)
                        max_side = max(size) if size else 0
                        pending.append((img_path, xml_file, out_path, max_side))

            if SR_SORT_BY_SIZE:
                pending.sort(key=lambda item: item[3])

            desc = "SR+Center val"
            for i in tqdm(range(0, len(pending), SR_BATCH_SIZE), desc=desc):
                batch = pending[i:i + SR_BATCH_SIZE]
                batch_paths = [p for p, _, _, _ in batch]
                batch_xmls = [x for _, x, _, _ in batch]
                batch_outputs = [o for _, _, o, _ in batch]
                current_bs = len(batch_paths)
                while current_bs > 0:
                    try:
                        process_sr_then_center_batch(
                            batch_paths[:current_bs],
                            batch_xmls[:current_bs],
                            batch_outputs[:current_bs],
                            model,
                            device
                        )
                        if current_bs < len(batch_paths):
                            batch_paths = batch_paths[current_bs:]
                            batch_xmls = batch_xmls[current_bs:]
                            batch_outputs = batch_outputs[current_bs:]
                            current_bs = len(batch_paths)
                            continue
                        break
                    except torch.cuda.OutOfMemoryError:
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        if current_bs == 1:
                            raise
                        current_bs = max(1, current_bs // 2)
            done_marker.write_text("completed\n")
        else:
            for class_dir in sorted(anno_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                class_id = class_dir.name
                img_class_dir = img_dir / class_id
                output_class_dir = output_split_dir / class_id
                done_marker = output_class_dir / "__done__.txt"
                if done_marker.exists():
                    continue
                output_class_dir.mkdir(exist_ok=True)

                xml_files = list(class_dir.glob('*.xml'))
                pending = []
                for xml_file in xml_files:
                    if not check_bbox_requirements(xml_file):
                        continue
                    img_path = img_class_dir / f"{xml_file.stem}.JPEG"
                    if not img_path.exists():
                        continue
                    out_path = output_class_dir / f"{xml_file.stem}.JPEG"
                    if not out_path.exists():
                        size = get_size_from_xml(xml_file)
                        max_side = max(size) if size else 0
                        pending.append((img_path, xml_file, out_path, max_side))

                if not pending:
                    continue

                if SR_SORT_BY_SIZE:
                    pending.sort(key=lambda item: item[3])

                desc = f"SR+Center train/{class_id}"
                for i in tqdm(range(0, len(pending), SR_BATCH_SIZE), desc=desc):
                    batch = pending[i:i + SR_BATCH_SIZE]
                    batch_paths = [p for p, _, _, _ in batch]
                    batch_xmls = [x for _, x, _, _ in batch]
                    batch_outputs = [o for _, _, o, _ in batch]
                    current_bs = len(batch_paths)
                    while current_bs > 0:
                        try:
                            process_sr_then_center_batch(
                                batch_paths[:current_bs],
                                batch_xmls[:current_bs],
                                batch_outputs[:current_bs],
                                model,
                                device
                            )
                            if current_bs < len(batch_paths):
                                batch_paths = batch_paths[current_bs:]
                                batch_xmls = batch_xmls[current_bs:]
                                batch_outputs = batch_outputs[current_bs:]
                                current_bs = len(batch_paths)
                                continue
                            break
                        except torch.cuda.OutOfMemoryError:
                            if device.type == 'cuda':
                                torch.cuda.empty_cache()
                            if current_bs == 1:
                                raise
                            current_bs = max(1, current_bs // 2)
                done_marker.write_text("completed\n")

def check_bbox_requirements(xml_path):
    """Check if image has a bbox and size <= MAX_IMAGE_SIDE."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        if width > MAX_IMAGE_SIDE or height > MAX_IMAGE_SIDE:
            return False

        # Require at least one object with a bbox
        objects = root.findall('object')
        if len(objects) < 1:
            return False
        bbox = objects[0].find('bndbox')
        if bbox is None:
            return False

        return True
        
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")
        return False

def create_filtered_dataset():
    # Set up paths
    base_dir = IMAGENET_ROOT
    output_dir = CENTERED_DIR

    train_count = 0
    val_count = 0

    for split in ['train', 'val']:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing {split} set...")

        img_dir = base_dir / 'Data/CLS-LOC' / split
        anno_dir = base_dir / 'Annotations/CLS-LOC' / split

        if split == 'val':
            print("Processing validation set...")
            valid_items = []
            for xml_file in tqdm(list(anno_dir.glob('*.xml')), desc="Collecting validation images"):
                if check_bbox_requirements(xml_file):
                    class_id = get_class_from_xml(xml_file)
                    if class_id is None:
                        continue
                    img_name = f"{xml_file.stem}.JPEG"
                    img_path = img_dir / img_name
                    if img_path.exists():
                        class_output_dir = split_dir / class_id
                        class_output_dir.mkdir(exist_ok=True)
                        output_path = class_output_dir / img_name
                        valid_items.append((img_path, xml_file, output_path))

            print(f"Processing {len(valid_items)} validation images...")
            for img_path, xml_path, output_path in tqdm(valid_items, desc="Processing validation"):
                if process_image_center_only(img_path, xml_path, output_path):
                    val_count += 1
        else:
            # For training set, process by class
            for class_dir in anno_dir.iterdir():
                if not class_dir.is_dir():
                    continue

                class_id = class_dir.name
                class_output_dir = split_dir / class_id
                class_output_dir.mkdir(exist_ok=True)

                xml_files = list(class_dir.glob('*.xml'))
                for xml_file in tqdm(xml_files, desc=f"Processing {class_id}"):
                    if not check_bbox_requirements(xml_file):
                        continue
                    img_path = img_dir / class_id / f"{xml_file.stem}.JPEG"
                    if img_path.exists():
                        output_path = class_output_dir / f"{xml_file.stem}.JPEG"
                        if process_image_center_only(img_path, xml_file, output_path):
                            train_count += 1

    # Print summary
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Training samples: {train_count}")
    print(f"Validation samples: {val_count}")
    print(f"Total samples: {train_count + val_count}")
    print(f"Output directory: {output_dir}")
    print("="*50)

if __name__ == "__main__":
    if RUN_CENTERING:
        create_filtered_dataset()
    if RUN_SR:
        super_resolve_then_center_dataset()