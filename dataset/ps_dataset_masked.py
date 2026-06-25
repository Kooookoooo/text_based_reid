"""
Extended dataset that loads pre-computed segmentation masks alongside images.
Drop-in replacement for ps_train_dataset.
"""

import json
import os
import numpy as np
from PIL import Image
from PIL import ImageFile
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from dataset.utils import pre_caption

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class ps_train_dataset_masked(Dataset):
    """
    Training dataset with mask loading.
    Returns (image1, image2, caption1, caption2, person, replace, mask)
    where mask is [H, W] integer tensor with part IDs 0-6.
    """

    def __init__(self, ann_file, transform, image_root, mask_root,
                 max_words=30, weak_pos_pair_probability=0.1, image_res=384):
        anns = []
        for f in ann_file:
            anns += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.mask_root = mask_root
        self.max_words = max_words
        self.image_res = image_res
        self.weak_pos_pair_probability = weak_pos_pair_probability
        self.person2image = defaultdict(list)
        self.person2text = defaultdict(list)
        person_id2idx = {}
        n = 0
        self.pairs = []
        for ann in anns:
            person_id = ann['id']
            if person_id not in person_id2idx.keys():
                person_id2idx[person_id] = n
                n += 1
            person_idx = person_id2idx[person_id]
            self.person2image[person_idx].append(ann['file_path'])
            for cap in ann['captions']:
                self.pairs.append((ann['file_path'], cap, person_idx))
                self.person2text[person_idx].append(cap)

    def __len__(self):
        return len(self.pairs)

    def augment(self, caption, person):
        caption_aug = caption
        if np.random.random() < self.weak_pos_pair_probability:
            caption_aug = np.random.choice(self.person2text[person], 1).item()
        if caption_aug == caption:
            replace = 0
        else:
            replace = 1
        return caption_aug, replace

    def load_mask(self, image_path):
        """Load pre-computed mask for an image. Returns [image_res, image_res] tensor."""
        filename = os.path.basename(image_path).replace('.jpg', '.npy').replace('.png', '.npy')
        mask_path = os.path.join(self.mask_root, filename)

        if os.path.exists(mask_path):
            mask = np.load(mask_path)  # [H, W] with values 0-6
            # Resize mask to match image_res using nearest interpolation
            mask_pil = Image.fromarray(mask)
            mask_pil = mask_pil.resize((self.image_res, self.image_res), Image.NEAREST)
            mask = np.array(mask_pil)
        else:
            # No mask available — return all zeros (background)
            mask = np.zeros((self.image_res, self.image_res), dtype=np.uint8)

        return torch.from_numpy(mask).long()

    def __getitem__(self, index):
        image_path, caption, person = self.pairs[index]
        caption_aug, replace = self.augment(caption, person)

        full_image_path = os.path.join(self.image_root, image_path)
        image = Image.open(full_image_path).convert('RGB')
        image1 = self.transform(image)
        image2 = self.transform(image)

        caption1 = pre_caption(caption, self.max_words)
        caption2 = pre_caption(caption_aug, self.max_words)

        mask = self.load_mask(image_path)

        return image1, image2, caption1, caption2, person, replace, mask
