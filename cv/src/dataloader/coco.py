import json
import os
from tqdm import tqdm
import shutil

def convert_bbox_coco_to_yolo(img_width, img_height, bbox):
    """
    Convert bounding box from COCO format to YOLO format
    COCO: [x_min, y_min, width, height]
    YOLO: [x_center, y_center, width, height] - normalized between 0 and 1
    """
    x_min, y_min, width, height = bbox
    
    # Calculate center points and normalize
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    
    # Normalize width and height
    norm_width = width / img_width
    norm_height = height / img_height
    
    return [x_center, y_center, norm_width, norm_height]

def convert_coco_to_yolo(coco_file, image_dir, output_dir):
    """
    Convert COCO annotations to YOLO format
    """
    # Load COCO annotations
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create category id mapping
    id_to_idx = {}
    for i, category in enumerate(coco_data['categories']):
        id_to_idx[category['id']] = i
    
    # Create class names file
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        for category in sorted(coco_data['categories'], key=lambda x: id_to_idx[x['id']]):
            f.write(f"{category['name']}\n")
    
    # Create directories for images and labels
    img_output_dir = os.path.join(output_dir, 'images')
    labels_output_dir = os.path.join(output_dir, 'labels')
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)
    
    # Map image id to file name
    img_id_to_file = {}
    for img in coco_data['images']:
        img_id_to_file[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }
    
    # Group annotations by image id
    annotations_by_img = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_img:
            annotations_by_img[img_id] = []
        annotations_by_img[img_id].append(ann)
    
    # Process each image
    print(f"Converting {len(img_id_to_file)} images...")
    for img_id, img_info in tqdm(img_id_to_file.items()):
        img_path = os.path.join(image_dir, img_info['file_name'])
        
        # Check if image exists
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found, skipping.")
            continue
        
        # Copy image to output directory
        shutil.copy(img_path, os.path.join(img_output_dir, img_info['file_name']))
        
        # Create YOLO annotation file
        base_name = os.path.splitext(img_info['file_name'])[0]
        txt_file = os.path.join(labels_output_dir, f"{base_name}.txt")
        
        with open(txt_file, 'w') as f:
            if img_id in annotations_by_img:
                for ann in annotations_by_img[img_id]:
                    category_idx = id_to_idx[ann['category_id']]
                    bbox = convert_bbox_coco_to_yolo(
                        img_info['width'], 
                        img_info['height'], 
                        ann['bbox']
                    )
                    # Write as: class_id x_center y_center width height
                    f.write(f"{category_idx} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
            else:
                # Create empty annotation file if no annotations for this image
                pass
    
    # Create train/val splits
    create_data_splits(output_dir, img_output_dir)
    
    # Create YAML configuration
    create_yaml_config(output_dir, len(coco_data['categories']))
    
    print(f"Conversion complete. Data saved to {output_dir}")

def create_data_splits(output_dir, img_dir):
    """Create train/val/test splits"""
    all_images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    total = len(all_images)
    
    # 80% train, 20% val
    train_size = int(0.8 * total)
    
    train_images = all_images[:train_size]
    val_images = all_images[train_size:]
    
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        for img in train_images:
            f.write(f"./images/{img}\n")
    
    with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
        for img in val_images:
            f.write(f"./images/{img}\n")

def create_yaml_config(output_dir, num_classes):
    """Create YAML configuration file for YOLOv11"""
    yaml_content = f"""# YOLOv11 configuration
train: ./train.txt  # train images
val: ./val.txt  # validation images

# Number of classes
nc: {num_classes}

# Class names
names:
"""
    
    # Add class names
    with open(os.path.join(output_dir, 'classes.txt'), 'r') as f:
        class_names = f.read().splitlines()
    
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content)

if __name__ == "__main__":
    # Config
    coco_file = "dataset/cv/annotations.json"
    image_dir = "dataset/cv/images"  # Directory containing all JPG images
    output_dir = "yolo_dataset"
    
    convert_coco_to_yolo(coco_file, image_dir, output_dir)