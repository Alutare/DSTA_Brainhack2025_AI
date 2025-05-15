import json
import os
import shutil
import random
import cv2
from tqdm import tqdm
import albumentations as A
from collections import defaultdict, Counter

def convert_bbox_coco_to_yolo(img_width, img_height, bbox):
    x_min, y_min, width, height = bbox
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    norm_width = width / img_width
    norm_height = height / img_height
    return [x_center, y_center, norm_width, norm_height]

def get_augmentations():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.HueSaturationValue(p=0.2)
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_id']))

def augment_and_save(img_path, anns, img_info, output_img_dir, output_lbl_dir, id_to_idx, augment_id):
    img = cv2.imread(img_path)
    if img is None:
        return

    transform = get_augmentations()
    transformed = transform(
        image=img,
        bboxes=[ann['bbox'] for ann in anns],
        category_id=[ann['category_id'] for ann in anns]
    )

    new_img = transformed['image']
    new_bboxes = transformed['bboxes']
    new_categories = transformed['category_id']

    base_name = os.path.splitext(img_info['file_name'])[0] + f"_aug{augment_id}"
    new_img_name = base_name + ".jpg"
    cv2.imwrite(os.path.join(output_img_dir, new_img_name), new_img)

    with open(os.path.join(output_lbl_dir, base_name + ".txt"), 'w') as f:
        for bbox, cat_id in zip(new_bboxes, new_categories):
            yolo_bbox = convert_bbox_coco_to_yolo(img_info['width'], img_info['height'], bbox)
            f.write(f"{id_to_idx[cat_id]} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")

def create_data_splits(output_dir, img_dir):
    all_images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(all_images)
    total = len(all_images)
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
    yaml_content = f"""# YOLO configuration
train: ./train.txt
val: ./val.txt
nc: {num_classes}
names:
"""
    with open(os.path.join(output_dir, 'classes.txt'), 'r') as f:
        class_names = f.read().splitlines()

    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"

    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content)

def convert_coco_to_yolo(coco_file, image_dir, output_dir):
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    id_to_idx = {cat['id']: i for i, cat in enumerate(coco_data['categories'])}
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        for category in sorted(coco_data['categories'], key=lambda x: id_to_idx[x['id']]):
            f.write(f"{category['name']}\n")

    img_output_dir = os.path.join(output_dir, 'images')
    lbl_output_dir = os.path.join(output_dir, 'labels')
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(lbl_output_dir, exist_ok=True)

    img_id_to_file = {
        img['id']: {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        } for img in coco_data['images']
    }

    # Count class instances
    class_instance_count = defaultdict(int)
    for ann in coco_data['annotations']:
        class_instance_count[ann['category_id']] += 1

    max_instances = max(class_instance_count.values())  # target class balance

    # Organize annotations per image
    annotations_by_img = defaultdict(list)
    for ann in coco_data['annotations']:
        annotations_by_img[ann['image_id']].append(ann)

    print(f"Processing {len(img_id_to_file)} images with class balancing...")
    for img_id, img_info in tqdm(img_id_to_file.items()):
        img_path = os.path.join(image_dir, img_info['file_name'])

        if not os.path.exists(img_path):
            print(f"Missing: {img_path}")
            continue

        shutil.copy(img_path, os.path.join(img_output_dir, img_info['file_name']))

        anns = annotations_by_img.get(img_id, [])
        base_name = os.path.splitext(img_info['file_name'])[0]

        # Write original label
        with open(os.path.join(lbl_output_dir, f"{base_name}.txt"), 'w') as f:
            for ann in anns:
                bbox = convert_bbox_coco_to_yolo(img_info['width'], img_info['height'], ann['bbox'])
                f.write(f"{id_to_idx[ann['category_id']]} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

        # Skip if no annotations
        if not anns:
            continue

        # Determine how rare this image is based on the rarest class it contains
        min_count = min(class_instance_count[ann['category_id']] for ann in anns)
        rarity_factor = max_instances // min_count
        augment_times = min(rarity_factor, 5)

        for aug_id in range(augment_times):
            augment_and_save(
                img_path=img_path,
                anns=anns,
                img_info=img_info,
                output_img_dir=img_output_dir,
                output_lbl_dir=lbl_output_dir,
                id_to_idx=id_to_idx,
                augment_id=aug_id + 1
            )

    create_data_splits(output_dir, img_output_dir)
    create_yaml_config(output_dir, len(coco_data['categories']))
    print(f"✅ YOLO dataset with balanced classes ready at: {output_dir}")

if __name__ == "__main__":
    coco_file = "advanced/cv/annotations.json"
    image_dir = "advanced/cv/images"
    output_dir = "yolo_dataset1"
    convert_coco_to_yolo(coco_file, image_dir, output_dir)
