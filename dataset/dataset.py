from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from transformers import RTDetrImageProcessor

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
image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r18vd")

def detection_collate_fn(batch):
    batch_dict = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            batch_dict[key] = torch.stack([item[key] for item in batch])
        else:
            # Handle other data types labels
            batch_dict[key] = [item[key][0] for item in batch]
    return batch_dict


def get_transforms(train=True):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            A.Rotate(limit=10, p=0.5),
            A.RandomScale(scale_limit=0.2, p=0.5),
            A.RandomResizedCrop(640, 640, scale=(0.8, 1.0)),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
    else:
        return A.Compose([
            A.Resize(640, 640),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))


class WOBDataset(Dataset):
    """
    Dataset for warehouse object detection
    """

    def __init__(self, base_dir: str, annotations_file: str, transform=None):
        self.base_dir = base_dir
        self.transform = transform
        self.processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r18vd")
        
        import json
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)
            
        # Create image_id to annotations mapping for efficient retrieval
        self.img_to_anns = {}
        for ann in self.coco_data["annotations"]:
            image_id = ann["image_id"]
            if image_id not in self.img_to_anns:
                self.img_to_anns[image_id] = []
            self.img_to_anns[image_id].append(ann)
            
        # Create a list of images that have annotations
        self.images = [img for img in self.coco_data["images"] 
                      if img["id"] in self.img_to_anns]

        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_info = self.images[idx]
        
        # omit /dataset/
        image_path = image_info["path"]
        if image_path.startswith("/dataset/"):
            image_path = image_path[9:]
        
        full_image_path = Path(self.base_dir, image_path)
        
        image = cv2.imread(str(full_image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {full_image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image_id = image_info["id"]
        annotations = self.img_to_anns.get(image_id, [])
        
        boxes = []
        labels = []
        
        for ann in annotations:
            # Convert category_id to position index
            category_id = category_id_to_pos[ann["category_id"]]
            bbox = ann["bbox"]  # [x, y, width, height] coco format
            boxes.append(bbox)
            labels.append(category_id)
            ann["category_id"] = category_id

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels) if labels else np.array([], dtype=np.int64)
        
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            image = transformed["image"]
            boxes = np.array(transformed["bboxes"], dtype=np.float32)
            labels = np.array(transformed["labels"])
            
            # Update annotations with transformed boxes
            for ann, box in zip(annotations, boxes):
                ann["bbox"] = box.tolist()
        
        coco_annotation = {
            "image_id": image_id,
            "annotations": annotations
        }
        encoding = self.processor(
            images=image,
            annotations=coco_annotation,
            return_tensors="pt",
            do_resize=False,
            )
        # Remove batch dimension
        for k, v in encoding.items():
            if isinstance(v, torch.Tensor):
                encoding[k] = v.squeeze(0)

        return encoding
