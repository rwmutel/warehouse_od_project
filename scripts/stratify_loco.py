import argparse
import json
import os
import random
import shutil
from pathlib import Path

categories = [
        {
            "id": 3,
            "name": "small_load_carrier",
            "supercategory": "",
            "color": "#e8e047",
            "metadata": {}
        },
        {
            "id": 5,
            "name": "forklift",
            "supercategory": "",
            "color": "#f8c718",
            "metadata": {}
        },
        {
            "id": 7,
            "name": "pallet",
            "supercategory": "",
            "color": "#8dd708",
            "metadata": {}
        },
        {
            "id": 10,
            "name": "stillage",
            "supercategory": "",
            "color": "#1640aa",
            "metadata": {}
        },
        {
            "id": 11,
            "name": "pallet_truck",
            "supercategory": "",
            "color": "#6ba8dc",
            "metadata": {}
        }
    ]

category_id_to_pos = {cat["id"]: i for i, cat in enumerate(categories)}
category_id_to_pos_r = {i: cat["id"] for i, cat in enumerate(categories)}


def split_coco_dataset(
    input_file: str, 
    output_dir: str,
    images_dir: str,
    train_ratio: float = 0.8, 
    val_ratio: float = 0.1, 
    test_ratio: float = 0.1,
) -> None:
    random.seed(42)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
    
    with open(input_file, 'r') as f:
        coco_data = json.load(f)
    
    image_ids = [img['id'] for img in coco_data['images']]
    random.shuffle(image_ids)

    coco_annotations = coco_data['annotations']
    for ann in coco_annotations:
        # Update category_id to position, crucial for D-FINE and RT-DETR
        ann['category_id'] = category_id_to_pos[ann['category_id']]
    
    num_images = len(image_ids)
    train_size = int(num_images * train_ratio)
    val_size = int(num_images * val_ratio)
    
    train_ids = set(image_ids[:train_size])
    val_ids = set(image_ids[train_size:train_size + val_size])
    test_ids = set(image_ids[train_size + val_size:])
    
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }
    
    for split_name, split_ids in splits.items():
        images = []
        for img in coco_data['images']:
            if img['id'] in split_ids:
                src_path = Path(images_dir) / img['path'][9:]
                new_file_name = f"images/{split_name}/{os.path.basename(img['file_name'])}"
                dst_path = output_dir / new_file_name
                
                shutil.copy2(src_path, dst_path)
                
                # Update image path in annotations
                img_copy = img.copy()
                img_copy['path'] = new_file_name
                img_copy['file_name'] = os.path.basename(img['file_name'])
                images.append(img_copy)
        
        annotations = [ann for ann in coco_annotations if ann['image_id'] in split_ids]
        
        split_dataset = {
            'images': images,
            'annotations': annotations,
            'categories': coco_data['categories']
        }
        
        output_file = output_dir / f"{split_name}.json"
        with open(output_file, 'w') as f:
            json.dump(split_dataset, f)
        
        print(f"{split_name} set: {len(images)} images, {len(annotations)} annotations")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split COCO dataset into train, val, and test sets")
    parser.add_argument("input_file", help="Path to input COCO JSON file")
    parser.add_argument("--images-dir", required=True, help="Directory containing source images")
    parser.add_argument("--output-dir", default="./", help="Output directory for split files and images")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio for training set")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Ratio for validation set")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Ratio for test set")
    # train set: 4077 images, 121344 annotations
    # val set: 509 images, 14729 annotations
    # test set: 511 images, 15355 annotations
    args = parser.parse_args()
    
    split_coco_dataset(
        args.input_file,
        args.output_dir,
        args.images_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )
