"""
ALBEF with:
1. Full-dataset Memory Bank (replaces FIFO queue + momentum model for contrastive loss)
2. Mask-Guided Part Pooling + Part-Level Contrastive Loss

The memory bank stores a representation for every sample in the dataset,
EMA-updated each time a sample appears. Contrastive loss is computed
against the ENTIRE dataset at every step.

No momentum encoder needed for contrastive loss (bank handles consistency).
Momentum encoder is still used for MLM soft labels and MRTD.
"""

from functools import partial
import torch
import torch.nn.functional as F
from torch import nn
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertForMaskedLM
from models.mask_module import MaskGuidedPartPooling, PartContrastiveLoss
from models.memory_bank import MemoryBank


class ALBEF_MemBank(nn.Module):
    def __init__(self, text_encoder=None, tokenizer=None, config=None):
        super().__init__()

        self.tokenizer = tokenizer
        embed_dim = config['embed_dim']
        vision_width = config['vision_width']
        self.image_res = config['image_res']
        self.patch_size = 16

        # === Main encoders ===
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=self.patch_size, embed_dim=768,
            depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)
        self.text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(self.text_width, embed_dim)
        self.temp = nn.Parameter(torch.ones([]) * config['temp'])

        self.mlm_probability = config['mlm_probability']
        self.mrtd_mask_probability = config['mrtd_mask_probability']
        self.momentum = config['momentum']

        # === Heads ===
        self.itm_head = nn.Linear(self.text_width, 2)
        self.prd_head = nn.Linear(self.text_width, 2)
        self.mrtd_head = nn.Linear(self.text_width, 2)

        # === Memory Bank (replaces queue) ===
        num_samples = config.get('num_train_samples', 35000)
        bank_momentum = config.get('bank_momentum', 0.9)
        self.memory_bank = MemoryBank(
            num_samples=num_samples,
            embed_dim=embed_dim,
            momentum=bank_momentum,
        )

        # === Mask-guided modules ===
        part_proj_dim = config.get('part_proj_dim', 256)
        self.part_pooling = MaskGuidedPartPooling(embed_dim=vision_width, proj_dim=part_proj_dim)
        self.part_loss_fn = PartContrastiveLoss(
            proj_dim=part_proj_dim, text_dim=self.text_width,
            temp=config.get('part_temp', 0.07)
        )
        self.part_loss_weight = config.get('part_loss_weight', 0.5)

        # === Momentum encoder (only for MLM soft labels + MRTD) ===
        self.visual_encoder_m = VisionTransformer(
            img_size=config['image_res'], patch_size=self.patch_size, embed_dim=768,
            depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        self.text_encoder_m = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)

        self.model_pairs = [
            [self.visual_encoder, self.visual_encoder_m],
            [self.text_encoder, self.text_encoder_m],
        ]
        self.copy_params()

    @property
    def patch_grid(self):
        p = self.patch_size
        return (self.image_res // p, self.image_res // p)

    def forward(self, image1, image2, text1, text2, alpha, idx, replace, masks=None):
        """
        Args:
            image1, image2: augmented image pairs [B, 3, H, W]
            text1, text2: tokenized text (text2 may be augmented caption)
            alpha: soft label interpolation weight
            idx: [B] — dataset sample indices (position in memory bank)
            replace: [B] — whether text was replaced (for PRD loss)
            masks: [B, H, W] — segmentation masks (optional)

        Returns:
            loss_cl, loss_pitm, loss_mlm, loss_prd, loss_mrtd, loss_part
        """
        # ===== Image encoding =====
        image_embeds = self.visual_encoder(image1)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image1.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        # ===== Text encoding =====
        text_output = self.text_encoder.bert(
            text2.input_ids, attention_mask=text2.attention_mask,
            return_dict=True, mode='text'
        )
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        # ===== Global Contrastive Loss via Memory Bank =====
        # Get person IDs from idx (stored in memory bank)
        person_ids = idx  # idx here IS the person ID from the dataset

        loss_cl = self.memory_bank.contrastive_loss(
            image_feat, text_feat, idx, person_ids, temp=self.temp
        )

        # Update memory bank with current features
        self.memory_bank.update(image_feat, text_feat, idx, person_ids)

        # ===== Part-level contrastive loss =====
        loss_part = torch.tensor(0.0, device=image1.device)
        if masks is not None:
            part_features, part_visibility = self.part_pooling(
                image_embeds, masks, self.patch_grid
            )
            loss_part = self.part_loss_fn(
                part_features, part_visibility,
                text_embeds, text2.attention_mask
            )
            loss_part = self.part_loss_weight * loss_part

        # ===== ITM (Image-Text Matching) =====
        output_pos = self.text_encoder.bert(
            encoder_embeds=text_embeds, attention_mask=text2.attention_mask,
            encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts,
            return_dict=True, mode='fusion',
        )

        # Hard negative mining using memory bank similarities
        bs = image1.size(0)
        with torch.no_grad():
            # Use memory bank for negative sampling
            valid_mask = self.memory_bank.initialized
            all_text_feats = self.memory_bank.text_bank[valid_mask]
            all_image_feats = self.memory_bank.image_bank[valid_mask]

            sim_i2t = image_feat @ all_text_feats.t()  # [B, M]
            sim_t2i = text_feat @ all_image_feats.t()  # [B, M]

            # Mask out same-person samples
            all_ids = self.memory_bank.id_bank[valid_mask]
            same_person_mask = person_ids.unsqueeze(1) == all_ids.unsqueeze(0)
            sim_i2t.masked_fill_(same_person_mask, -1e9)
            sim_t2i.masked_fill_(same_person_mask, -1e9)

            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)

        # Select negatives from current batch
        image_neg_idx = torch.multinomial(weights_t2i[:, :bs], 1).flatten()
        image_embeds_neg = image_embeds[image_neg_idx]
        text_neg_idx = torch.multinomial(weights_i2t[:, :bs], 1).flatten()
        text_embeds_neg = text_embeds[text_neg_idx]
        text_atts_neg = text2.attention_mask[text_neg_idx]

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text2.attention_mask, text_atts_neg], dim=0)
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        output_neg_cross = self.text_encoder.bert(
            encoder_embeds=text_embeds_all, attention_mask=text_atts_all,
            encoder_hidden_states=image_embeds_all, encoder_attention_mask=image_atts_all,
            return_dict=True, mode='fusion',
        )

        vl_embeddings = torch.cat([
            output_pos.last_hidden_state[:, 0, :],
            output_neg_cross.last_hidden_state[:, 0, :]
        ], dim=0)
        vl_output = self.itm_head(vl_embeddings)
        itm_labels = torch.cat([
            torch.ones(bs, dtype=torch.long),
            torch.zeros(2 * bs, dtype=torch.long)
        ], dim=0).to(image1.device)
        loss_pitm = F.cross_entropy(vl_output, itm_labels)

        # PRD loss
        prd_output = self.prd_head(output_pos.last_hidden_state[:, 0, :])
        loss_prd = F.cross_entropy(prd_output, replace)

        # ===== MLM + MRTD (uses momentum encoder) =====
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image2)

        input_ids = text1.input_ids.clone()
        labels = input_ids.clone()
        mrtd_input_ids = input_ids.clone()

        # MLM
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size,
                                      targets=labels, probability_matrix=probability_matrix)

        with torch.no_grad():
            logits_m = self.text_encoder_m(
                input_ids, attention_mask=text1.attention_mask,
                encoder_hidden_states=image_embeds_m, encoder_attention_mask=image_atts,
                return_dict=True, return_logits=True,
            )
            prediction = F.softmax(logits_m, dim=-1)

        mlm_output = self.text_encoder(
            input_ids, attention_mask=text1.attention_mask,
            encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts,
            return_dict=True, labels=labels, soft_labels=prediction, alpha=alpha
        )
        loss_mlm = mlm_output.loss

        # MRTD
        with torch.no_grad():
            probability_matrix = torch.full(labels.shape, self.mrtd_mask_probability)
            mrtd_input_ids = self.mask(mrtd_input_ids, self.text_encoder.config.vocab_size,
                                      probability_matrix=probability_matrix)
            mrtd_logits_m = self.text_encoder_m(
                mrtd_input_ids, attention_mask=text1.attention_mask,
                encoder_hidden_states=image_embeds_m, encoder_attention_mask=image_atts,
                return_dict=True, return_logits=True,
            )
            weights = F.softmax(mrtd_logits_m, dim=-1)
            mrtd_input_ids, mrtd_labels = self.mrtd_mask_modeling(
                mrtd_input_ids, text1.input_ids, text1.attention_mask, weights
            )

        output_mrtd = self.text_encoder.bert(
            mrtd_input_ids, attention_mask=text1.attention_mask,
            encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts,
            return_dict=True,
        )
        mrtd_output = self.mrtd_head(output_mrtd.last_hidden_state.view(-1, self.text_width))
        loss_mrtd = F.cross_entropy(mrtd_output, mrtd_labels.view(-1))

        return loss_cl, loss_pitm, loss_mlm, loss_prd, loss_mrtd, loss_part

    # === Helper methods ===

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    def mask(self, input_ids, vocab_size, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        if targets is not None:
            targets[~masked_indices] = -100
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(input_ids.device)
        input_ids[indices_random] = random_words[indices_random]
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    def mrtd_mask_modeling(self, mrtd_input_ids, ori_input_ids, attention_mask, weights):
        bs = mrtd_input_ids.size(0)
        weights = weights.view(-1, weights.size(-1))
        pred = torch.multinomial(weights, 1).view(bs, -1)
        pred[:, 0] = self.tokenizer.cls_token_id
        mrtd_input_ids = pred * attention_mask
        mrtd_labels = (pred != ori_input_ids) * attention_mask
        mrtd_labels[mrtd_input_ids == self.tokenizer.pad_token_id] = -100
        mrtd_labels[mrtd_input_ids == self.tokenizer.cls_token_id] = -100
        return mrtd_input_ids, mrtd_labels
