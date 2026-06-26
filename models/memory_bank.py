"""
Full-Dataset Memory Bank — stores EVERY image and EVERY text separately.

Handles imbalanced pairs (e.g. 1 image with 3 texts, another with 5).
Each image slot and each text slot has a person_id for contrastive matching.

Contrastive logic:
- Positive images = all images of the same person
- Positive texts = all texts describing the same person
- Everything else = negatives

Usage:
    bank = FullMemoryBank(num_images=N_img, num_texts=N_txt, embed_dim=256, momentum=0.9)
    bank.initialize(data_loader, model, tokenizer, device)
    loss = bank.contrastive_loss(img_feats, txt_feats, img_indices, txt_indices, person_ids, temp)
    bank.update_images(img_feats, img_indices, person_ids)
    bank.update_texts(txt_feats, txt_indices, person_ids)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class FullMemoryBank(nn.Module):
    """
    Stores one embedding per image and one embedding per text in the entire dataset.
    Supports imbalanced image-text pairs per person.
    """

    def __init__(self, num_images, num_texts, embed_dim=256, momentum=0.9):
        super().__init__()
        self.num_images = num_images
        self.num_texts = num_texts
        self.embed_dim = embed_dim
        self.momentum = momentum

        # Image bank
        self.register_buffer("image_bank", torch.zeros(num_images, embed_dim))
        self.register_buffer("image_id_bank", torch.full((num_images,), -1, dtype=torch.long))

        # Text bank
        self.register_buffer("text_bank", torch.zeros(num_texts, embed_dim))
        self.register_buffer("text_id_bank", torch.full((num_texts,), -1, dtype=torch.long))

        self.register_buffer("is_initialized", torch.tensor(False))

    @torch.no_grad()
    def initialize(self, train_dataset, model, tokenizer, device, max_words=50, batch_size=128):
        """
        Populate banks by iterating through the dataset.
        Handles imbalanced pairs: each unique image gets a slot, each unique text gets a slot.
        """
        if self.is_initialized:
            print("Memory bank already initialized, skipping.")
            return

        print("Initializing full memory bank...")
        model.eval()

        from torch.utils.data import DataLoader

        # We need raw access to images and texts with their indices
        # Build index maps from the dataset's pairs
        # pairs = [(file_path, caption, person_idx), ...]
        pairs = train_dataset.pairs
        image_root = train_dataset.image_root
        transform = train_dataset.transform

        # Build unique image list and text list
        image_to_idx = {}  # file_path -> image_bank_index
        text_to_idx = {}   # (file_path, caption) -> text_bank_index
        img_idx = 0
        txt_idx = 0

        image_person_map = {}  # image_bank_idx -> person_id
        text_person_map = {}   # text_bank_idx -> person_id

        for file_path, caption, person_id in pairs:
            if file_path not in image_to_idx:
                image_to_idx[file_path] = img_idx
                image_person_map[img_idx] = person_id
                img_idx += 1

            text_key = (file_path, caption)
            if text_key not in text_to_idx:
                text_to_idx[text_key] = txt_idx
                text_person_map[txt_idx] = person_id
                txt_idx += 1

        print(f"  Unique images: {img_idx}, Unique texts: {txt_idx}")
        assert img_idx <= self.num_images, f"num_images too small: {img_idx} > {self.num_images}"
        assert txt_idx <= self.num_texts, f"num_texts too small: {txt_idx} > {self.num_texts}"

        # Populate image bank
        print("  Populating image bank...")
        from PIL import Image as PILImage
        image_paths = sorted(image_to_idx.keys(), key=lambda x: image_to_idx[x])

        for i in tqdm(range(0, len(image_paths), batch_size), desc="  Images"):
            batch_paths = image_paths[i:i + batch_size]
            images = []
            for fp in batch_paths:
                img = PILImage.open(f"{image_root}/{fp}").convert("RGB")
                images.append(transform(img))

            image_tensor = torch.stack(images).to(device)
            image_embeds = model.visual_encoder(image_tensor)
            image_feat = F.normalize(model.vision_proj(image_embeds[:, 0, :]), dim=-1)

            for j, fp in enumerate(batch_paths):
                bank_idx = image_to_idx[fp]
                self.image_bank[bank_idx] = image_feat[j].cpu()
                self.image_id_bank[bank_idx] = image_person_map[bank_idx]

        # Populate text bank
        print("  Populating text bank...")
        text_items = sorted(text_to_idx.keys(), key=lambda x: text_to_idx[x])
        captions_list = [cap for (_, cap) in text_items]

        for i in tqdm(range(0, len(captions_list), batch_size), desc="  Texts"):
            batch_captions = captions_list[i:i + batch_size]
            text_input = tokenizer(batch_captions, padding='longest', max_length=max_words,
                                   truncation=True, return_tensors="pt").to(device)
            text_output = model.text_encoder.bert(
                text_input.input_ids, attention_mask=text_input.attention_mask,
                return_dict=True, mode='text'
            )
            text_feat = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

            for j in range(len(batch_captions)):
                bank_idx = i + j
                self.text_bank[bank_idx] = text_feat[j].cpu()
                self.text_id_bank[bank_idx] = text_person_map[bank_idx]

        # Normalize
        self.image_bank = F.normalize(self.image_bank, dim=-1)
        self.text_bank = F.normalize(self.text_bank, dim=-1)
        self.is_initialized = torch.tensor(True)

        # Store maps for use during training
        self.image_to_idx = image_to_idx
        self.text_to_idx = text_to_idx

        model.train()
        print(f"Memory bank initialized: {img_idx} images, {txt_idx} texts")

    @torch.no_grad()
    def update_images(self, image_feats, file_paths):
        """EMA-update image bank for given file paths."""
        image_feats = F.normalize(image_feats.detach(), dim=-1)
        for j, fp in enumerate(file_paths):
            if fp in self.image_to_idx:
                idx = self.image_to_idx[fp]
                self.image_bank[idx] = (
                    self.momentum * self.image_bank[idx] +
                    (1 - self.momentum) * image_feats[j].cpu()
                )
                self.image_bank[idx] = F.normalize(self.image_bank[idx], dim=-1)

    @torch.no_grad()
    def update_texts(self, text_feats, file_paths, captions):
        """EMA-update text bank for given (file_path, caption) pairs."""
        text_feats = F.normalize(text_feats.detach(), dim=-1)
        for j in range(len(captions)):
            key = (file_paths[j], captions[j])
            if key in self.text_to_idx:
                idx = self.text_to_idx[key]
                self.text_bank[idx] = (
                    self.momentum * self.text_bank[idx] +
                    (1 - self.momentum) * text_feats[j].cpu()
                )
                self.text_bank[idx] = F.normalize(self.text_bank[idx], dim=-1)

    def contrastive_loss(self, image_feats, text_feats, person_ids, temp=0.07):
        """
        Global contrastive loss against entire bank.

        Positives:
        - For image query: all texts in text_bank with same person_id
        - For text query: all images in image_bank with same person_id

        Args:
            image_feats: [B, D] current batch image features
            text_feats: [B, D] current batch text features
            person_ids: [B] person identity for each sample in batch
            temp: temperature

        Returns:
            loss: scalar
        """
        image_feats = F.normalize(image_feats, dim=-1)
        text_feats = F.normalize(text_feats, dim=-1)

        # Get full banks (on same device as feats)
        all_text_feats = self.text_bank.to(image_feats.device)      # [N_txt, D]
        all_text_ids = self.text_id_bank.to(image_feats.device)     # [N_txt]
        all_image_feats = self.image_bank.to(image_feats.device)    # [N_img, D]
        all_image_ids = self.image_id_bank.to(image_feats.device)   # [N_img]

        # Filter out uninitialized slots (id == -1)
        valid_txt_mask = all_text_ids >= 0
        valid_img_mask = all_image_ids >= 0
        all_text_feats = all_text_feats[valid_txt_mask]
        all_text_ids = all_text_ids[valid_txt_mask]
        all_image_feats = all_image_feats[valid_img_mask]
        all_image_ids = all_image_ids[valid_img_mask]

        # Image-to-Text: sim against all texts
        sim_i2t = image_feats @ all_text_feats.t() / temp  # [B, N_txt]
        # Positive mask: same person
        pos_mask_i2t = person_ids.unsqueeze(1) == all_text_ids.unsqueeze(0)  # [B, N_txt]
        pos_count_i2t = pos_mask_i2t.float().sum(dim=1, keepdim=True).clamp(min=1)
        targets_i2t = pos_mask_i2t.float() / pos_count_i2t

        # Text-to-Image: sim against all images
        sim_t2i = text_feats @ all_image_feats.t() / temp  # [B, N_img]
        pos_mask_t2i = person_ids.unsqueeze(1) == all_image_ids.unsqueeze(0)  # [B, N_img]
        pos_count_t2i = pos_mask_t2i.float().sum(dim=1, keepdim=True).clamp(min=1)
        targets_t2i = pos_mask_t2i.float() / pos_count_t2i

        # Cross-entropy with soft targets
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * targets_i2t, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * targets_t2i, dim=1).mean()

        loss = (loss_i2t + loss_t2i) / 2
        return loss
