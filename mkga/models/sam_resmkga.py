import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry

from mkga.models.lora_linear import inject_lora_into_sam


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MultiKernelFeatureExtractor(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.branch3x3 = nn.Conv2d(in_c, out_c // 2, 3, 1, 1)
        self.branch5x5 = nn.Conv2d(in_c, out_c // 2, 3, 1, 2, dilation=2)
        self.fuse = nn.Sequential(nn.Conv2d(out_c, out_c, 1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.fuse(torch.cat([self.branch3x3(x), self.branch5x5(x)], dim=1))


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        return x * self.psi(psi)


class MKGABlock(nn.Module):
    def __init__(self, in_c_high, in_c_skip, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.reduce = nn.Conv2d(in_c_high, out_c, 1)
        self.skip_cleaner = MultiKernelFeatureExtractor(in_c_skip, out_c)
        self.attn = AttentionGate(F_g=out_c, F_l=out_c, F_int=out_c // 2)
        self.res_block = nn.Sequential(
            nn.Conv2d(out_c * 2, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(True),
            nn.Conv2d(out_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(True)
        )

    def forward(self, high, skip):
        high_up = self.reduce(self.up(high))
        if high_up.shape[-2:] != skip.shape[-2:]:
            high_up = F.interpolate(high_up, size=skip.shape[-2:], mode='bilinear')
        skip_clean = self.skip_cleaner(skip)
        skip_gated = self.attn(g=high_up, x=skip_clean)
        return self.res_block(torch.cat([high_up, skip_gated], dim=1))


class SAM_ResMKGA(nn.Module):
    def __init__(self, checkpoint_path="medsam_vit_b.pth", num_classes=2, img_size=512, lora_rank=4):
        super().__init__()
        self.sam_model = sam_model_registry["vit_b"](checkpoint=None)
        self.image_encoder = self.sam_model.image_encoder

        for param in self.image_encoder.parameters():
            param.requires_grad = False
        if lora_rank > 0:
            print(f"   >>> Injecting LoRA (Rank={lora_rank}) into SAM Encoder...")
            inject_lora_into_sam(self.sam_model, rank=lora_rank)
        else:
            print(f"   >>> LoRA Disabled (Rank={lora_rank}). Encoder is FROZEN.")

        target_size = img_size // 16
        if self.image_encoder.pos_embed.shape[1] != target_size:
            new_pos = self.image_encoder.pos_embed.permute(0, 3, 1, 2)
            new_pos = F.interpolate(new_pos, size=(target_size, target_size), mode='bicubic', align_corners=False).permute(0, 2, 3, 1)
            self.image_encoder.pos_embed = nn.Parameter(new_pos)

        self.adapter_conv = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.adapter_se = SEBlock(256)
        self.make_skip3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 2, 2), nn.BatchNorm2d(128), nn.ReLU())
        self.make_skip2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, 2), nn.BatchNorm2d(64), nn.ReLU())
        self.make_skip1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 2, 2), nn.BatchNorm2d(32), nn.ReLU())
        self.dec4 = MKGABlock(256, 256, 256)
        self.dec3 = MKGABlock(256, 128, 128)
        self.dec2 = MKGABlock(128, 64, 64)
        self.dec1 = MKGABlock(64, 32, 32)
        self.head_seg = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(32, 1, 1))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head_pos = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 3))
        self.head_tirads = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, num_classes))

        if checkpoint_path:
            self._load_weights(checkpoint_path, target_size)

    def _load_weights(self, path, size):
        try:
            state = torch.load(path, map_location="cpu")
            if 'model_state_dict' in state:
                state = state['model_state_dict']
            new_state = {}
            for k, v in state.items():
                k = k.replace("medsam_model.", "").replace("module.", "")
                new_state[k] = v
            if "image_encoder.pos_embed" in new_state:
                pos_embed = new_state["image_encoder.pos_embed"]
                if pos_embed.shape[1] != size:
                    pos_embed = pos_embed.permute(0, 3, 1, 2)
                    pos_embed = F.interpolate(pos_embed, size=(size, size), mode='bicubic', align_corners=False)
                    pos_embed = pos_embed.permute(0, 2, 3, 1)
                    new_state["image_encoder.pos_embed"] = pos_embed
            self.sam_model.load_state_dict(new_state, strict=False)
            print("MedSAM weights loaded successfully.")
        except Exception as e:
            print(f"Weight loading warning: {e}")

    def forward(self, img):
        x_frozen = self.image_encoder(img)
        x_adapted = x_frozen + self.adapter_se(self.adapter_conv(x_frozen))
        s4 = x_adapted
        s3 = self.make_skip3(s4)
        s2 = self.make_skip2(s3)
        s1 = self.make_skip1(s2)
        d4 = self.dec4(x_adapted, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        mask = self.head_seg(d1)
        feat_pooled = self.gap(x_adapted).flatten(1)
        return mask, self.head_pos(feat_pooled), self.head_tirads(feat_pooled)
