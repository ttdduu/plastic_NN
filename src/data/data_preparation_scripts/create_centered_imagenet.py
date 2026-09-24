"""
Create a centered 256x256 ImageNet dataset.

For each image with a bounding box annotation:
  1. Center the bbox in a square (zero-padded if needed).
  2. Resize the square to OUTPUT_SIZE x OUTPUT_SIZE.
  3. Save to output directory, preserving class/split structure.
"""
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
from PIL import Image

OUTPUT_SIZE = 256
IMAGENET_ROOT = Path('/home/tomasdu/repos/datasets/imagenet-ILSVRC/ILSVRC')
OUTPUT_DIR = Path('/home/tomasdu/repos/datasets/imagenet-centered-256')


def parse_xml(xml_path):
    """Return (class_name, bbox) or (None, None) on failure."""
    try:
        root = ET.parse(xml_path).getroot()
        objects = root.findall('object')
        if not objects:
            return None, None
        name = objects[0].find('name').text
        b = objects[0].find('bndbox')
        bbox = (
            int(b.find('xmin').text),
            int(b.find('ymin').text),
            int(b.find('xmax').text),
            int(b.find('ymax').text),
        )
        return name, bbox
    except Exception:
        return None, None


def center_and_resize(img, bbox):
    """
    Place the image on a black square so the bbox centre aligns with the
    square's centre, then resize the square to OUTPUT_SIZE x OUTPUT_SIZE.
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')

    w, h = img.size
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2

    # Square side = 2 * max distance from bbox centre to any image edge
    side = int(2 * max(cx, w - cx, cy, h - cy))
    if side < 1:
        side = max(w, h)

    square = Image.new('RGB', (side, side), (0, 0, 0))
    paste_x = int(side / 2 - cx)
    paste_y = int(side / 2 - cy)
    square.paste(img, (paste_x, paste_y))

    return square.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)


def process_split(split):
    img_dir = IMAGENET_ROOT / 'Data/CLS-LOC' / split
    anno_dir = IMAGENET_ROOT / 'Annotations/CLS-LOC' / split
    out_dir = OUTPUT_DIR / split

    if split == 'val':
        # Annotations are flat files; images are also flat.
        xml_files = list(anno_dir.glob('*.xml'))
        items = []
        for xml_path in xml_files:
            class_name, bbox = parse_xml(xml_path)
            if class_name is None:
                continue
            img_path = img_dir / f'{xml_path.stem}.JPEG'
            if not img_path.exists():
                continue
            out_path = out_dir / class_name / img_path.name
            if not out_path.exists():
                items.append((img_path, bbox, out_path))

        for img_path, bbox, out_path in tqdm(items, desc='val'):
            try:
                with Image.open(img_path) as img:
                    result = center_and_resize(img, bbox)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                result.save(out_path, quality=95)
            except Exception as e:
                print(f'Error processing {img_path}: {e}')

    else:  # train
        class_dirs = sorted(d for d in anno_dir.iterdir() if d.is_dir())
        for class_dir in class_dirs:
            class_id = class_dir.name
            out_class_dir = out_dir / class_id
            done_marker = out_class_dir / '__done__.txt'
            if done_marker.exists():
                continue

            items = []
            for xml_path in class_dir.glob('*.xml'):
                _, bbox = parse_xml(xml_path)
                if bbox is None:
                    continue
                img_path = img_dir / class_id / f'{xml_path.stem}.JPEG'
                if not img_path.exists():
                    continue
                out_path = out_class_dir / img_path.name
                if not out_path.exists():
                    items.append((img_path, bbox, out_path))

            if not items:
                out_class_dir.mkdir(parents=True, exist_ok=True)
                done_marker.write_text('completed\n')
                continue

            out_class_dir.mkdir(parents=True, exist_ok=True)
            for img_path, bbox, out_path in tqdm(items, desc=f'train/{class_id}', leave=False):
                try:
                    with Image.open(img_path) as img:
                        result = center_and_resize(img, bbox)
                    result.save(out_path, quality=95)
                except Exception as e:
                    print(f'Error processing {img_path}: {e}')

            done_marker.write_text('completed\n')


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val']:
        print(f'\n=== {split} ===')
        process_split(split)
    print('\nDone.')
