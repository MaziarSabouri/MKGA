#!/usr/bin/env python3
"""Training entry point for MKGA multi-task models."""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from mkga.dataset import ThyroidDataset
from mkga.models import ResNet34, ResNet34_MKGA, ResNet34_ResMKGA
from mkga.optim import PCGrad
from mkga.paths import CHECKPOINTS_DIR, resolve_data_root, resolve_medsam_weights


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class RobustLoss(nn.Module):
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        pred = torch.sigmoid(logits)
        smooth = 1.0
        intersection = (pred * targets).sum()
        dice = 1 - (2. * intersection + smooth) / (pred.sum() + targets.sum() + smooth)
        return 0.5 * bce + 0.5 * dice


class UncertaintyLoss(nn.Module):
    def __init__(self, num_tasks=3):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, loss_seg, loss_tirads, loss_pos):
        total_loss = 0
        for i, task_loss in enumerate([loss_seg, loss_tirads, loss_pos]):
            precision = torch.exp(-self.log_vars[i])
            total_loss += task_loss * precision + self.log_vars[i]
        return total_loss


def get_args():
    parser = argparse.ArgumentParser(description="Train MKGA multi-task thyroid models.")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to directory containing Dataset/ (or set MKGA_DATA_ROOT).")
    parser.add_argument("--medsam_weights", type=str, default=None,
                        help="Path to medsam_vit_b.pth (default: weights/medsam_vit_b.pth).")
    parser.add_argument("--model", type=str, required=True,
                        choices=["ResNet34", "SAM", "ResNet34_MKGA", "SAM_MKGA", "SAM_ResMKGA", "ResNet34_ResMKGA"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--source", type=str, default="ThyroidXL")
    parser.add_argument("--binary_tirads", type=str, default="True")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--use_lora", type=str, default="False")
    parser.add_argument("--freeze_resnet", type=str, default="False")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--use_pcgrad", type=str, default="False")
    parser.add_argument("--ablate_gate", type=str, default="False")
    parser.add_argument("--ablate_multi", type=str, default="False")
    parser.add_argument("--ablate_se", type=str, default="False")
    parser.add_argument("--kernel_combo", type=str, default="3_5", choices=["1_3", "3_5", "3_7"])
    parser.add_argument("--use_uncertainty", type=str, default="False")
    parser.add_argument("--use_mixstyle", type=str, default="False")
    return parser.parse_args()


def build_model(args, num_classes, ablation_dict, lora_rank, medsam_weights):
    if args.model == "ResNet34":
        return ResNet34(num_classes=num_classes, use_mixstyle=(args.use_mixstyle == "True"))
    if args.model == "ResNet34_MKGA":
        return ResNet34_MKGA(num_classes=num_classes, ablation_dict=ablation_dict)
    if args.model == "ResNet34_ResMKGA":
        return ResNet34_ResMKGA(num_classes=num_classes, ablation_dict=ablation_dict)
    if args.model == "SAM":
        from mkga.models import SAM
        return SAM(medsam_weights, num_classes=num_classes, lora_rank=lora_rank)
    if args.model == "SAM_MKGA":
        from mkga.models import SAM_MKGA
        return SAM_MKGA(medsam_weights, num_classes=num_classes, lora_rank=lora_rank)
    if args.model == "SAM_ResMKGA":
        from mkga.models import SAM_ResMKGA
        return SAM_ResMKGA(medsam_weights, num_classes=num_classes, lora_rank=lora_rank)
    raise ValueError("Invalid model choice.")


def main():
    args = get_args()
    set_seed()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = resolve_data_root(args.data_root)
    medsam_weights = resolve_medsam_weights(args.medsam_weights)

    binary_mode = (args.binary_tirads == "True")
    use_lora = (args.use_lora == "True")
    freeze_resnet = (args.freeze_resnet == "True")
    use_pcgrad = (args.use_pcgrad == "True")
    use_uncertainty = (args.use_uncertainty == "True")
    use_mixstyle = (args.use_mixstyle == "True")
    ablate_gate = (args.ablate_gate == "True")
    ablate_multi = (args.ablate_multi == "True")
    ablate_se = (args.ablate_se == "True")
    lora_rank = args.lora_rank if use_lora else 0

    ablation_dict = {
        "use_gate": not ablate_gate,
        "use_multi_kernel": not ablate_multi,
        "use_se_bridge": not ablate_se,
        "kernel_combo": args.kernel_combo,
    }
    num_classes = 2 if binary_mode else 6

    print(f"--- TRAINING {args.model} ---")
    print(f"Data root: {data_root}")
    print(f"Settings: Fraction={args.fraction} | LoRa={use_lora} (Rank {lora_rank}) | "
          f"FreezeResNet={freeze_resnet} | PCGrad={use_pcgrad} | MixStyle={use_mixstyle}")
    print(f"Ablations: NoGate={ablate_gate} | NoMultiKernel={ablate_multi} | "
          f"NoSE={ablate_se} | Kernels={args.kernel_combo}")

    train_ds = ThyroidDataset(
        data_root, mode='train', split='train', val_ratio=0.2, fraction=args.fraction,
        data_source=args.source, binary_tirads=binary_mode,
    )
    val_ds = ThyroidDataset(
        data_root, mode='train', split='val', val_ratio=0.2, fraction=args.fraction,
        data_source=args.source, binary_tirads=binary_mode,
    )
    print(f"   |-- Training Samples: {len(train_ds)}")
    print(f"   |-- Validation Samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    model = build_model(args, num_classes, ablation_dict, lora_rank, medsam_weights).to(device)

    if "ResNet" in args.model and freeze_resnet:
        print("   |-- [INFO] Freezing ResNet Encoder...")
        for name, param in model.named_parameters():
            if any(x in name for x in ["encoder", "resnet", "layer1", "layer2", "layer3", "layer4", "conv1", "bn1"]):
                param.requires_grad = False
            else:
                param.requires_grad = True

    dynamic_loss = UncertaintyLoss(num_tasks=3).to(device)
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if use_uncertainty:
        trainable_params += list(dynamic_loss.parameters())
    base_optimizer = optim.AdamW(trainable_params, lr=args.lr)
    optimizer = PCGrad(base_optimizer) if use_pcgrad else base_optimizer

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   |-- Trainable Params: {train_params:,} / {total_params:,} "
          f"({100 * train_params / total_params:.2f}%)")

    criterion_seg = RobustLoss()
    criterion_cls = nn.CrossEntropyLoss(ignore_index=-1)
    best_val_loss = float('inf')
    patience_counter = 0
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for batch in pbar:
            imgs, masks = batch['image'].to(device), batch['mask'].to(device)
            mask_valid = batch['mask_valid'].to(device)
            pos, tirads = batch['position'].to(device), batch['tirads'].to(device)

            optimizer.zero_grad()
            out, pred_pos, pred_tirads = model(imgs)

            loss_seg = torch.tensor(0.0, device=device)
            if mask_valid.sum() > 0:
                if isinstance(out, list):
                    loss_seg = loss_seg + criterion_seg(out[0][mask_valid], masks[mask_valid])
                    for i in range(1, len(out)):
                        target_aux = (
                            F.interpolate(masks, size=out[i].shape[-2:], mode='nearest')
                            if out[i].shape[-2:] != masks.shape[-2:] else masks
                        )
                        loss_seg = loss_seg + 0.4 * criterion_seg(out[i][mask_valid], target_aux[mask_valid])
                else:
                    loss_seg = loss_seg + criterion_seg(out[mask_valid], masks[mask_valid])

            loss_pos = 0.5 * criterion_cls(pred_pos, pos)
            loss_tirads = 0.5 * criterion_cls(pred_tirads, tirads)
            total_loss = (
                dynamic_loss(loss_seg, loss_tirads, loss_pos) if use_uncertainty
                else loss_seg + loss_pos + loss_tirads
            )
            if torch.isnan(total_loss):
                print("!!! NaN detected in loss !!! Skipping step.")
                continue

            if use_pcgrad:
                active_losses = []
                if loss_seg.requires_grad:
                    active_losses.append(loss_seg)
                if loss_pos.requires_grad:
                    active_losses.append(loss_pos)
                if loss_tirads.requires_grad:
                    active_losses.append(loss_tirads)
                optimizer.pc_backward(active_losses)
            else:
                total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += total_loss.item()
            pbar.set_postfix({'Train Loss': total_loss.item()})

        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0

        model.eval()
        val_loss = 0
        valid_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs, masks = batch['image'].to(device), batch['mask'].to(device)
                mask_valid = batch['mask_valid'].to(device)
                pos, tirads = batch['position'].to(device), batch['tirads'].to(device)
                out, pred_pos, pred_tirads = model(imgs)
                v_loss = 0
                if mask_valid.sum() > 0:
                    if isinstance(out, list):
                        out = out[0]
                    v_loss += criterion_seg(out[mask_valid], masks[mask_valid])
                v_loss += 0.5 * criterion_cls(pred_pos, pos)
                v_loss += 0.5 * criterion_cls(pred_tirads, tirads)
                if not torch.isnan(v_loss):
                    val_loss += v_loss.item()
                    valid_batches += 1

        avg_val_loss = val_loss / valid_batches if valid_batches > 0 else float('inf')
        print(f"   |-- Epoch {epoch + 1} Summary | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            mode_suffix = ""
            if use_lora:
                mode_suffix += f"_LoRa_Rank{lora_rank}"
            if freeze_resnet:
                mode_suffix += "_Frozen"
            if use_pcgrad:
                mode_suffix += "_PCGrad"
            ablation_suffix = ""
            if ablate_gate:
                ablation_suffix += "_NoGate"
            if ablate_multi:
                ablation_suffix += "_NoMulti"
            if ablate_se:
                ablation_suffix += "_NoSE"
            if args.kernel_combo != "3_5":
                ablation_suffix += f"_K{args.kernel_combo}"
            save_path = CHECKPOINTS_DIR / f"{args.model}{mode_suffix}{ablation_suffix}_Frac{args.fraction}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"   |-- [SAVED] Best Val Loss: {best_val_loss:.4f}. Saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n--- Early stopping triggered at Epoch {epoch + 1} ---")
                break


if __name__ == "__main__":
    main()
