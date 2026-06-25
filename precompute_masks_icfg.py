"""
Pre-compute segmentation masks for ICFG-PEDES dataset images.
Saves as .npy files in the mask_root directory.

Usage:
    python precompute_masks_icfg.py --image_root ../dataset/ICFG-PEDES/imgs \
        --output_dir ../dataset/ICFG-PEDES/masks
"""

import os
import glob
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

# SegFormer label → 7 body-part groups
LABEL_TO_PART = {
    0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 4, 6: 4, 7: 2, 8: 4,
    9: 5, 10: 5, 11: 1, 12: 4, 13: 4, 14: 3, 15: 3, 16: 6, 17: 2,
}


def load_model():
    print("Loading SegFormer...")
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, processor


def segment(model, processor, image):
    inputs = processor(images=image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    upsampled = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    seg_map = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    part_map = np.zeros_like(seg_map, dtype=np.uint8)
    for seg_label, part_id in LABEL_TO_PART.items():
        part_map[seg_map == seg_label] = part_id
    return part_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_root', default='../dataset/ICFG-PEDES/imgs')
    parser.add_argument('--output_dir', default='../dataset/ICFG-PEDES/masks')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model, processor = load_model()

    # Find all images
    images = sorted(
        glob.glob(os.path.join(args.image_root, '**', '*.jpg'), recursive=True) +
        glob.glob(os.path.join(args.image_root, '**', '*.png'), recursive=True)
    )
    print(f"Found {len(images)} images")

    # Skip already processed
    done = set(os.listdir(args.output_dir))
    remaining = [p for p in images
                 if os.path.basename(p).replace('.jpg', '.npy').replace('.png', '.npy') not in done]
    print(f"Already done: {len(images) - len(remaining)}, remaining: {len(remaining)}")

    for img_path in tqdm(remaining, desc="Computing masks"):
        try:
            image = Image.open(img_path).convert("RGB")
            part_map = segment(model, processor, image)
            filename = os.path.basename(img_path).replace('.jpg', '.npy').replace('.png', '.npy')
            np.save(os.path.join(args.output_dir, filename), part_map)
        except Exception as e:
            print(f"  Error: {img_path}: {e}")

    print(f"Done! {len(os.listdir(args.output_dir))} masks in {args.output_dir}")


if __name__ == "__main__":
    main()
