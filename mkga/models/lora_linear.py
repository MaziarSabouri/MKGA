import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=4, alpha=4):
        super().__init__()
        self.original_layer = original_layer
        self.r = rank
        self.alpha = alpha
        in_dim = original_layer.in_features
        out_dim = original_layer.out_features
        self.lora_A = nn.Parameter(torch.zeros(in_dim, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        original_out = self.original_layer(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_out + lora_out


def inject_lora_into_sam(sam_model, rank=4):
    for param in sam_model.image_encoder.parameters():
        param.requires_grad = False
    if rank == 0:
        print("LoRA rank is 0. Encoder backbone hard-frozen (No LoRA injected).")
        return sam_model
    for block in sam_model.image_encoder.blocks:
        original_qkv = block.attn.qkv
        block.attn.qkv = LoRALinear(original_qkv, rank=rank)
    print(f"LoRA injected with rank {rank}. Encoder backbone frozen, LoRA params trainable.")
    return sam_model
