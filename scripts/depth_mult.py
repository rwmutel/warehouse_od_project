import os
import torch
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Process images with depth estimation and dim distant objects")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing input images")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save processed images")
    parser.add_argument("--model_name", type=str, default="depth-anything/Depth-Anything-V2-Base-hf", 
                        help="DepthAnything model name on HuggingFace")
    parser.add_argument("--depth_strength", type=float, default=1.0, 
                        help="Strength of depth dimming effect (0-1)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of images to process simultaneously")
    return parser.parse_args()

def process_batch(model, processor, image_batch, args):
    standard_size = (384, 384)
    resized_batch = []
    original_sizes = []
    
    for image in image_batch:
        original_sizes.append(image.size)
        resized = image.resize(standard_size, Image.Resampling.LANCZOS)
        resized_batch.append(resized)
    
    inputs = processor(images=resized_batch, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depths = outputs.predicted_depth

    results = []
    for i, (image, orig_size) in enumerate(zip(image_batch, original_sizes)):
        depth = torch.nn.functional.interpolate(
            predicted_depths[i:i+1].unsqueeze(1),
            size=orig_size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()
        
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
        depth_normalized = 1 - depth_normalized
        # depth_factor = 1 - (depth_normalized * args.depth_strength)
        depth_factor = (depth_normalized * args.depth_strength)
        
        img_array = np.array(image).astype(np.float32) / 255.0
        for c in range(3):
            img_array[:, :, c] = img_array[:, :, c] * depth_factor
            
        result_img = (img_array * 255).astype(np.uint8)
        results.append(Image.fromarray(result_img))
    
    return results

def main():
    args = parse_args()
    
    os.makedirs(args.output_folder, exist_ok=True)
    
    print(f"Loading {args.model_name}...")
    image_processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModelForDepthEstimation.from_pretrained(args.model_name)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    image_files = [f for f in os.listdir(args.input_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
    
    if not image_files:
        print(f"No image files found in {args.input_folder}")
        return
    
    print(f"Processing {len(image_files)} images in batches of {args.batch_size}...")
    
    for i in tqdm(range(0, len(image_files), args.batch_size)):
        batch_files = image_files[i:i + args.batch_size]
        batch_images = []
        
        for img_file in batch_files:
            input_path = os.path.join(args.input_folder, img_file)
            image = Image.open(input_path).convert("RGB")
            batch_images.append(image)
        
        results = process_batch(model, image_processor, batch_images, args)
        
        for img_file, result_img in zip(batch_files, results):
            output_path = os.path.join(args.output_folder, img_file)
            result_img.save(output_path)
    
    print(f"Processed images saved to {args.output_folder}")

if __name__ == "__main__":
    main()