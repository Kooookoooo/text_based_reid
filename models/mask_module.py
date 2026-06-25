"""
Mask-Guided Part Pooling + Part-Level Contrastive Loss Module.

Integrates body-part segmentation masks into the ALBEF/RaSa model:
1. Pools image patch tokens by body-part regions using pre-computed masks
2. Computes part-level contrastive loss (part image features ↔ text features)

Body parts (7 groups, index 0 is background, skipped):
  1: head, 2: upper_body, 3: arms, 4: lower_body, 5: feet, 6: accessory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


NUM_PARTS = 6  # Excluding background (parts 1-6)
PART_NAMES = ['head', 'upper_body', 'arms', 'lower_body', 'feet', 'accessory']


class MaskGuidedPartPooling(nn.Module):
    """
    Pools ViT patch tokens into body-part embeddings using segmentation masks.

    Input:
        image_tokens: [B, N+1, D] (N patches + CLS token)
        masks: [B, H, W] with values 0-6 (0=background, 1-6=body parts)
        patch_grid: (patch_h, patch_w) — spatial dims of the patch grid

    Output:
        part_features: [B, NUM_PARTS, D] — one embedding per body part
        part_visibility: [B, NUM_PARTS] — binary mask of which parts are visible
    """

    def __init__(self, embed_dim, num_parts=NUM_PARTS, proj_dim=256):
        super().__init__()
        self.num_parts = num_parts
        self.part_proj = nn.Linear(embed_dim, proj_dim)

    def forward(self, image_tokens, masks, patch_grid):
        """
        Args:
            image_tokens: [B, N+1, D] — includes CLS at position 0
            masks: [B, H, W] — integer mask with part IDs (0=bg, 1-6=parts)
            patch_grid: tuple (patch_h, patch_w)
        Returns:
            part_features: [B, num_parts, proj_dim]
            part_visibility: [B, num_parts] — 1 if part has any pixels, 0 otherwise
        """
        B, N_plus_1, D = image_tokens.shape
        patch_h, patch_w = patch_grid
        N = patch_h * patch_w

        # Remove CLS token — only use patch tokens
        patch_tokens = image_tokens[:, 1:N+1, :]  # [B, N, D]

        # Convert masks to one-hot and downsample to patch grid
        # masks: [B, H, W] → part_weights: [B, num_parts, N]
        part_weights = self._mask_to_patch_weights(masks, patch_h, patch_w)  # [B, num_parts, N]

        # Track which parts are visible (have any non-zero weight)
        part_visibility = (part_weights.sum(dim=-1) > 0).float()  # [B, num_parts]

        # Normalize weights per part (so they sum to 1)
        weight_sum = part_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        part_weights_norm = part_weights / weight_sum  # [B, num_parts, N]

        # Weighted average pooling: [B, num_parts, N] × [B, N, D] → [B, num_parts, D]
        part_features = torch.bmm(part_weights_norm, patch_tokens)  # [B, num_parts, D]

        # Project to contrastive space
        part_features = self.part_proj(part_features)  # [B, num_parts, proj_dim]
        part_features = F.normalize(part_features, dim=-1)

        return part_features, part_visibility

    def _mask_to_patch_weights(self, masks, patch_h, patch_w):
        """
        Convert integer masks [B, H, W] to per-part patch weights [B, num_parts, N].
        Uses one-hot encoding + adaptive average pooling.
        """
        B, H, W = masks.shape
        device = masks.device

        # One-hot encode parts 1-6 (skip background=0)
        # masks: [B, H, W] → one_hot: [B, num_parts, H, W]
        one_hot = torch.zeros(B, self.num_parts, H, W, device=device)
        for part_id in range(self.num_parts):
            one_hot[:, part_id] = (masks == (part_id + 1)).float()

        # Downsample to patch grid using average pooling
        # [B, num_parts, H, W] → [B, num_parts, patch_h, patch_w]
        part_weights = F.adaptive_avg_pool2d(one_hot, (patch_h, patch_w))

        # Flatten spatial dims: [B, num_parts, patch_h, patch_w] → [B, num_parts, N]
        part_weights = part_weights.view(B, self.num_parts, -1)

        return part_weights


class PartContrastiveLoss(nn.Module):
    """
    Part-level contrastive loss.
    Aligns body-part image features with text features using cross-attention
    to find which text tokens correspond to each body part.

    The text encoder's token embeddings are used to compute part-specific
    text features via attention between part image features and text tokens.
    """

    def __init__(self, proj_dim=256, text_dim=768, temp=0.07):
        super().__init__()
        self.text_part_proj = nn.Linear(text_dim, proj_dim)
        self.temp = nn.Parameter(torch.tensor(temp))

    def forward(self, part_features, part_visibility, text_tokens, text_mask):
        """
        Args:
            part_features: [B, num_parts, proj_dim] — from MaskGuidedPartPooling
            part_visibility: [B, num_parts] — which parts are visible
            text_tokens: [B, L, text_dim] — text encoder hidden states (all tokens)
            text_mask: [B, L] — attention mask for text

        Returns:
            loss_part: scalar — part-level contrastive loss
        """
        B, K, proj_dim = part_features.shape
        L = text_tokens.shape[1]

        # Project text tokens to same space as part features
        text_proj = self.text_part_proj(text_tokens)  # [B, L, proj_dim]
        text_proj = F.normalize(text_proj, dim=-1)

        # Compute cross-attention: which text tokens correspond to each part
        # [B, K, proj_dim] × [B, proj_dim, L] → [B, K, L]
        attn_scores = torch.bmm(part_features, text_proj.transpose(1, 2))  # [B, K, L]

        # Mask padding tokens
        text_mask_expanded = text_mask.unsqueeze(1).expand(-1, K, -1)  # [B, K, L]
        attn_scores = attn_scores.masked_fill(~text_mask_expanded.bool(), -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, K, L]

        # Compute part-specific text features via attention
        part_text_features = torch.bmm(attn_weights, text_proj)  # [B, K, proj_dim]
        part_text_features = F.normalize(part_text_features, dim=-1)

        # Part-level contrastive loss (only for visible parts)
        # Similarity between image part and its text part
        sim = (part_features * part_text_features).sum(dim=-1) / self.temp  # [B, K]

        # Only compute loss for visible parts
        visible_mask = part_visibility.bool()  # [B, K]

        if visible_mask.sum() == 0:
            return torch.tensor(0.0, device=part_features.device)

        # Positive: matching image-text part pairs
        # For simplicity, use InfoNCE across the batch for each part
        loss = torch.tensor(0.0, device=part_features.device)
        num_parts_used = 0

        for k in range(K):
            # Get visibility for this part across batch
            vis_k = visible_mask[:, k]  # [B]
            if vis_k.sum() < 2:
                continue

            # Get features for visible samples
            img_k = part_features[vis_k, k]  # [M, proj_dim]
            txt_k = part_text_features[vis_k, k]  # [M, proj_dim]

            # Cross-modal similarity matrix
            sim_matrix = img_k @ txt_k.t() / self.temp  # [M, M]

            # InfoNCE: diagonal are positives
            labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
            loss_i2t = F.cross_entropy(sim_matrix, labels)
            loss_t2i = F.cross_entropy(sim_matrix.t(), labels)
            loss = loss + (loss_i2t + loss_t2i) / 2
            num_parts_used += 1

        if num_parts_used > 0:
            loss = loss / num_parts_used

        return loss
