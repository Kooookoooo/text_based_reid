"""
Full-Dataset Memory Bank with EMA Updates.

Maintains a representation for every image and every text in the dataset.
Updated via exponential moving average each time a sample appears in a batch.
Contrastive loss is computed against the entire memory bank (global).

Initialization: On first epoch, the bank is populated by forwarding all samples
through the encoders (warm-start pass).

Usage:
    bank = MemoryBank(num_samples=34674, embed_dim=256, momentum=0.9)

    # Warm-start (before training):
    bank.initialize(data_loader, model, device)

    # In training loop:
    loss = bank.contrastive_loss(image_feats, text_feats, indices, person_ids, temp=0.07)
    bank.update(image_feats, text_feats, indices)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class MemoryBank(nn.Module):
    """
    Full-dataset memory bank for global contrastive learning.

    Stores one embedding per image-text pair in the dataset.
    EMA-updated when a sample appears in a batch.
    Contrastive loss computed against ALL stored embeddings.
    """

    def __init__(self, num_samples, embed_dim=256, momentum=0.9):
        """
        Args:
            num_samples: total number of image-text pairs in training set
            embed_dim: dimensionality of stored features
            momentum: EMA momentum (0.9 = 90% old + 10% new)
        """
        super().__init__()
        self.num_samples = num_samples
        self.embed_dim = embed_dim
        self.momentum = momentum

        # Memory banks — not model parameters, just persistent buffers
        self.register_buffer("image_bank", torch.zeros(num_samples, embed_dim))
        self.register_buffer("text_bank", torch.zeros(num_samples, embed_dim))
        self.register_buffer("id_bank", torch.zeros(num_samples, dtype=torch.long))
        self.register_buffer("is_initialized", torch.tensor(False))

    @torch.no_grad()
    def initialize(self, data_loader, model, tokenizer, device, max_words=50):
        """
        Warm-start: forward all training samples through the encoders
        to populate the memory bank before training begins.

        Args:
            data_loader: training data loader (iterates full dataset)
            model: the model (needs visual_encoder, vision_proj, text_encoder, text_proj)
            tokenizer: text tokenizer
            device: cuda device
            max_words: max text length for tokenizer
        """
        if self.is_initialized:
            print("Memory bank already initialized, skipping.")
            return

        print("Initializing memory bank (forwarding all samples through encoders)...")
        model.eval()

        sample_idx = 0
        for batch in tqdm(data_loader, desc="Populating memory bank"):
            # Unpack — handle both masked (7 items) and non-masked (6 items) datasets
            if len(batch) == 7:
                image1, image2, text1, text2, person_ids, replace, masks = batch
            else:
                image1, image2, text1, text2, person_ids, replace = batch

            image1 = image1.to(device)
            text_input = tokenizer(text2, padding='longest', max_length=max_words,
                                   return_tensors="pt").to(device)

            # Forward through encoders
            image_embeds = model.visual_encoder(image1)
            image_feat = F.normalize(model.vision_proj(image_embeds[:, 0, :]), dim=-1)

            text_output = model.text_encoder.bert(
                text_input.input_ids, attention_mask=text_input.attention_mask,
                return_dict=True, mode='text'
            )
            text_feat = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

            # Store in bank
            bs = image_feat.size(0)
            end_idx = min(sample_idx + bs, self.num_samples)
            actual_bs = end_idx - sample_idx

            self.image_bank[sample_idx:end_idx] = image_feat[:actual_bs].cpu()
            self.text_bank[sample_idx:end_idx] = text_feat[:actual_bs].cpu()
            self.id_bank[sample_idx:end_idx] = person_ids[:actual_bs]

            sample_idx = end_idx
            if sample_idx >= self.num_samples:
                break

        # Move banks to device and normalize
        self.image_bank = F.normalize(self.image_bank, dim=-1)
        self.text_bank = F.normalize(self.text_bank, dim=-1)
        self.is_initialized = torch.tensor(True)

        model.train()
        print(f"Memory bank initialized: {sample_idx} samples populated.")

    @torch.no_grad()
    def update(self, image_feats, text_feats, indices, person_ids=None):
        """
        EMA-update the memory bank for the given sample indices.

        Args:
            image_feats: [B, D] — normalized image features from current batch
            text_feats: [B, D] — normalized text features from current batch
            indices: [B] — dataset indices for these samples
            person_ids: [B] — person identity labels
        """
        image_feats = F.normalize(image_feats.detach(), dim=-1)
        text_feats = F.normalize(text_feats.detach(), dim=-1)

        # EMA update at specific positions
        self.image_bank[indices] = (
            self.momentum * self.image_bank[indices] +
            (1 - self.momentum) * image_feats
        )
        self.text_bank[indices] = (
            self.momentum * self.text_bank[indices] +
            (1 - self.momentum) * text_feats
        )

        # Re-normalize
        self.image_bank[indices] = F.normalize(self.image_bank[indices], dim=-1)
        self.text_bank[indices] = F.normalize(self.text_bank[indices], dim=-1)

        if person_ids is not None:
            self.id_bank[indices] = person_ids

    def contrastive_loss(self, image_feats, text_feats, indices, person_ids, temp=0.07):
        """
        Compute global contrastive loss against the entire memory bank.

        Args:
            image_feats: [B, D] — current batch image features (has gradients)
            text_feats: [B, D] — current batch text features (has gradients)
            indices: [B] — dataset indices
            person_ids: [B] — person identity labels
            temp: temperature scalar or nn.Parameter

        Returns:
            loss: scalar contrastive loss
        """
        image_feats = F.normalize(image_feats, dim=-1)
        text_feats = F.normalize(text_feats, dim=-1)

        # Full bank (detached + cloned to prevent in-place modification issues)
        all_image_feats = self.image_bank.clone().detach()  # [N, D]
        all_text_feats = self.text_bank.clone().detach()    # [N, D]
        all_ids = self.id_bank.clone().detach()             # [N]

        # Similarity matrices against entire bank
        sim_i2t = image_feats @ all_text_feats.t() / temp  # [B, N]
        sim_t2i = text_feats @ all_image_feats.t() / temp  # [B, N]
        sim_i2i = image_feats @ all_image_feats.t() / temp  # [B, N]
        sim_t2t = text_feats @ all_text_feats.t() / temp    # [B, N]

        # Build soft targets: same person_id = positive
        pos_mask = person_ids.unsqueeze(1) == all_ids.unsqueeze(0)  # [B, N]
        pos_count = pos_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        targets = pos_mask.float() / pos_count  # [B, N]

        # Cross-entropy with soft targets
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * targets, dim=1).mean()
        loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1) * targets, dim=1).mean()
        loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1) * targets, dim=1).mean()

        loss = (loss_i2t + loss_t2i + loss_i2i + loss_t2t) / 4
        return loss
