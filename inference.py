import argparse
import os

import cv2
import torch
from super_gradients.common.object_names import Models
from super_gradients.training import models
from tqdm import tqdm

from dataset import categories

MODELS = {
    "yolonas": {
        "s": {
            "model": Models.YOLO_NAS_S,
        },
        "m": {
            "model": Models.YOLO_NAS_M,
        },
        "l": {
            "model": Models.YOLO_NAS_L,
        },
    },
    "yolox": {
        "s": {
            "model": Models.YOLOX_S,
        },
        "m": {
            "model": Models.YOLOX_M,
        },
        "l": {
            "model": Models.YOLOX_L,
        },
    }
}

def load_model(model_type: str, model_size: str, checkpoint_path: str) -> torch.nn.Module:
    """Load a model with specified weights"""
    if model_type not in MODELS or model_size not in MODELS[model_type]:
        raise ValueError(f"Unsupported model type {model_type} or size {model_size}")
    
    model = models.get(
        MODELS[model_type][model_size]["model"],
        num_classes=len(categories),
        checkpoint_path=checkpoint_path
    )
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained YOLO models")
    parser.add_argument("--model-type", type=str, required=True, choices=["yolonas", "yolox"])
    parser.add_argument("--model-size", type=str, required=True, choices=["s", "m", "l"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    model = load_model(args.model_type, args.model_size, args.checkpoint)
    model = model.to(args.device)
    
    if os.path.isfile(args.input):
        image_paths = [args.input]
    else:
        image_paths = [
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
    
    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            continue
        output_path = str(os.path.join(args.output, os.path.basename(image_path)))
        with torch.no_grad():
            predictions = model.predict(image)
            predictions.save(str(os.path.join(args.output, os.path.basename(image_path))))
        
        print(f"Processed {image_path} -> {output_path}")

if __name__ == "__main__":
    main()