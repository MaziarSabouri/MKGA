import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry

from mkga.models.lora_linear import inject_lora_into_sam


class SAM(nn.Module):
    def __init__(self, checkpoint_path="medsam_vit_b.pth", num_classes=2, img_size=512, lora_rank=4):
        super().__init__()
        self.sam_model = sam_model_registry["vit_b"](checkpoint=None)
        self.image_encoder = self.sam_model.image_encoder
        self.prompt_encoder = self.sam_model.prompt_encoder
        self.mask_decoder = self.sam_model.mask_decoder
        inject_lora_into_sam(self.sam_model, rank=lora_rank)

        target_size = img_size // 16
        if self.image_encoder.pos_embed.shape[1] != target_size:
            new_pos = self.image_encoder.pos_embed.permute(0, 3, 1, 2)
            new_pos = F.interpolate(new_pos, size=(target_size, target_size), mode='bicubic', align_corners=False).permute(0, 2, 3, 1)
            self.image_encoder.pos_embed = nn.Parameter(new_pos)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head_tirads = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes))
        self.head_pos = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 3))

        if checkpoint_path:
            self._load_weights(checkpoint_path, target_size)

    def _load_weights(self, path, size):
        try:
            state = torch.load(path, map_location="cpu")
            new_state = {k.replace("medsam_model.", ""): v for k, v in state.items()}
            if "image_encoder.pos_embed" in new_state:
                pos_embed = new_state["image_encoder.pos_embed"]
                if pos_embed.shape[1] != size:
                    pos_embed = pos_embed.permute(0, 3, 1, 2)
                    pos_embed = F.interpolate(pos_embed, size=(size, size), mode='bicubic', align_corners=False)
                    pos_embed = pos_embed.permute(0, 2, 3, 1)
                    new_state["image_encoder.pos_embed"] = pos_embed
            self.sam_model.load_state_dict(new_state, strict=False)
            print("MedSAM weights loaded successfully (with resized pos_embed).")
        except Exception as e:
            print(f"Weight loading warning: {e}")

    def forward(self, img, external_boxes=None):
        img_feat = self.image_encoder(img)
        B, _, H, W = img.shape
        if external_boxes is None:
            boxes = torch.tensor([0, 0, W, H], dtype=torch.float, device=img.device).repeat(B, 1).unsqueeze(1)
        else:
            boxes = external_boxes

        sparse, dense = self.prompt_encoder(points=None, boxes=boxes, masks=None)
        if dense.shape[-1] != img_feat.shape[-1]:
            dense = F.interpolate(dense, size=(img_feat.shape[2], img_feat.shape[3]), mode='bilinear')
        dense_pe = self.prompt_encoder.get_dense_pe()
        if dense_pe.shape[-1] != img_feat.shape[-1]:
            dense_pe = F.interpolate(dense_pe, size=(img_feat.shape[2], img_feat.shape[3]), mode='bilinear')

        low_res_masks = []
        for i in range(B):
            lr, _ = self.mask_decoder(
                image_embeddings=img_feat[i].unsqueeze(0),
                image_pe=dense_pe,
                sparse_prompt_embeddings=sparse[i].unsqueeze(0),
                dense_prompt_embeddings=dense[i].unsqueeze(0),
                multimask_output=False
            )
            low_res_masks.append(lr)

        low_res = torch.cat(low_res_masks, dim=0)
        mask_logits = F.interpolate(low_res, size=(H, W), mode="bilinear", align_corners=False)
        feat_pooled = self.global_avg_pool(img_feat).flatten(1)
        return mask_logits, self.head_pos(feat_pooled), self.head_tirads(feat_pooled)
