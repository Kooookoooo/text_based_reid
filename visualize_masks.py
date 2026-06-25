"""
Visualize pre-computed masks with original images and color-coded body part labels.
Saves side-by-side visualizations to output/mask_vis/

Usage:
    python visualize_masks.py
"""

import os
import glob
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

IMAGE_ROOT = "/mrsach_dang_quoc_minh_101/workspace/WS_DATASETS/REID/ICFG-PEDES/imgs"
MASK_ROOT = "/mrsach_dang_quoc_minh_101/workspace/pretrain_framework/RaSa/data/ICFG-PEDES/masks"
OUTPUT_DIR = "output/mask_vis"
NUM_SAMPLES = 20

PART_NAMES = ['background', 'head', 'upper_body', 'arms', 'lower_body', 'feet', 'accessory']
PART_COLORS = [
    (0, 0, 0),        # background - black
    (255, 100, 0),    # head - orange
    (0, 100, 255),    # upper_body - blue
    (0, 200, 100),    # arms - green
    (200, 0, 200),    # lower_body - purple
    (255, 255, 0),    # feet - yellow
    (255, 0, 100),    # accessory - pink
]


def visualize_sample(img_path, mask_path, output_path):
    img = Image.open(img_path).convert("RGB")
    mask = np.load(mask_path)

    # Resize mask to image size if different
    if mask.shape != (img.height, img.width):
        mask_pil = Image.fromarray(mask)
        mask_pil = mask_pil.resize((img.width, img.height), Image.NEAREST)
        mask = np.array(mask_pil)

    # Create colored mask
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    present_parts = []
    for part_id in range(len(PART_NAMES)):
        if part_id == 0:
            continue
        region = mask == part_id
        if region.any():
            colored[region] = PART_COLORS[part_id]
            present_parts.append((part_id, PART_NAMES[part_id], PART_COLORS[part_id]))

    # Create overlay
    img_array = np.array(img)
    overlay = (img_array * 0.5 + colored * 0.5).astype(np.uint8)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(colored)
    axes[1].set_title("Part Mask")
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis('off')

    # Legend
    patches = [mpatches.Patch(color=np.array(c)/255.0, label=n) for _, n, c in present_parts]
    fig.legend(handles=patches, loc='lower center', ncol=min(6, len(patches)),
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(os.path.basename(img_path), fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find masks
    mask_files = sorted(glob.glob(os.path.join(MASK_ROOT, "*.npy")))
    if not mask_files:
        print(f"No masks found in {MASK_ROOT}")
        return

    print(f"Found {len(mask_files)} masks")

    # Sample random ones
    random.seed(42)
    samples = random.sample(mask_files, min(NUM_SAMPLES, len(mask_files)))

    for i, mask_path in enumerate(samples):
        filename = os.path.basename(mask_path).replace('.npy', '.jpg')
        # Search for image (might be in subdirectories)
        img_path = None
        for ext in ['.jpg', '.png']:
            candidate = os.path.join(IMAGE_ROOT, filename.replace('.jpg', ext))
            if os.path.exists(candidate):
                img_path = candidate
                break
            # Try recursive search
            found = glob.glob(os.path.join(IMAGE_ROOT, '**', filename.replace('.jpg', ext)), recursive=True)
            if found:
                img_path = found[0]
                break

        if img_path is None:
            print(f"  Image not found for {filename}")
            continue

        out_path = os.path.join(OUTPUT_DIR, f"vis_{i+1:02d}_{filename.replace('.jpg', '.png')}")
        visualize_sample(img_path, mask_path, out_path)

    print(f"Saved {NUM_SAMPLES} visualizations to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
