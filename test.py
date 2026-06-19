#!/usr/bin/env python3
"""Evaluation entry point for MKGA multi-task models."""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from mkga.dataset import ThyroidDataset
from mkga.models import ResNet34, ResNet34_MKGA, ResNet34_ResMKGA
from mkga.paths import RESULTS_DIR, resolve_data_root


def calculate_segmentation_metrics(pred, target):
    pred = (pred > 0.5).float().view(-1)
    target = target.view(-1)
    TP = (pred * target).sum()
    FP = (pred * (1 - target)).sum()
    FN = ((1 - pred) * target).sum()
    TN = ((1 - pred) * (1 - target)).sum()
    smooth = 1e-6
    dice = (2. * TP + smooth) / (pred.sum() + target.sum() + smooth)
    iou = (TP + smooth) / (TP + FP + FN + smooth)
    recall = (TP + smooth) / (TP + FN + smooth)
    precision = (TP + smooth) / (TP + FP + smooth)
    specificity = (TN + smooth) / (TN + FP + smooth)
    return {
        "Dice": dice.item(),
        "IoU": iou.item(),
        "Recall": recall.item(),
        "Precision": precision.item(),
        "Specificity": specificity.item(),
    }


def get_detailed_classification_metrics(y_true, y_pred, y_prob, classes):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    metrics_dict = {}
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    if len(classes) != len(unique_classes):
        classes = sorted(unique_classes)
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    for i, cls in enumerate(classes):
        if i >= cm.shape[0]:
            continue
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        sensitivity = tp / (tp + fn + 1e-6)
        specificity = tn / (tn + fp + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-6)
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
        try:
            if len(classes) == 2:
                if y_prob.ndim > 1 and y_prob.shape[1] > 1:
                    auc = roc_auc_score(
                        (y_true == cls).astype(int),
                        y_prob[:, 1] if cls == classes[1] else y_prob[:, 0],
                    )
                else:
                    auc = 0.0
            elif y_prob.shape[1] > i:
                auc = roc_auc_score((y_true == cls).astype(int), y_prob[:, i])
            else:
                auc = 0.0
        except ValueError:
            auc = 0.0
        metrics_dict[f"Class {cls}"] = {
            "Accuracy": acc,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Precision": precision,
            "F1-Score": f1,
            "ROC AUC": auc,
        }

    for avg in ['macro', 'weighted']:
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        try:
            if len(classes) == 2:
                auc_avg = roc_auc_score(y_true, y_prob[:, 1]) if y_prob.ndim > 1 and y_prob.shape[1] > 1 else 0.0
            else:
                auc_avg = roc_auc_score(y_true, y_prob, multi_class='ovr', average=avg)
        except ValueError:
            auc_avg = 0.0
        metrics_dict[f"{avg.capitalize()} Avg"] = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Sensitivity": rec,
            "Specificity": 0.0,
            "Precision": prec,
            "F1-Score": f1,
            "ROC AUC": auc_avg,
        }
    return pd.DataFrame(metrics_dict).transpose()


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate MKGA multi-task thyroid models.")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to directory containing Dataset/ (or set MKGA_DATA_ROOT).")
    parser.add_argument("--model", type=str, required=True,
                        choices=["ResNet34", "SAM", "ResNet34_MKGA", "SAM_MKGA", "SAM_ResMKGA", "ResNet34_ResMKGA"])
    parser.add_argument("--test_on", type=str, required=True, choices=["ThyroidXL", "DDTI"])
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--binary_tirads", type=str, default="True")
    parser.add_argument("--use_masks", type=str, default="False")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--use_lora", type=str, default="False")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--ablate_gate", type=str, default="False")
    parser.add_argument("--ablate_multi", type=str, default="False")
    parser.add_argument("--ablate_se", type=str, default="False")
    parser.add_argument("--kernel_combo", type=str, default="3_5", choices=["1_3", "3_5", "3_7"])
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for result CSVs (default: results/).")
    return parser.parse_args()


def build_model(args, num_classes, ablation_dict, current_rank):
    if args.model == "ResNet34":
        return ResNet34(num_classes=num_classes)
    if args.model == "ResNet34_MKGA":
        return ResNet34_MKGA(num_classes=num_classes, ablation_dict=ablation_dict)
    if args.model == "ResNet34_ResMKGA":
        return ResNet34_ResMKGA(num_classes=num_classes, ablation_dict=ablation_dict)
    if args.model == "SAM":
        from mkga.models import SAM
        return SAM(checkpoint_path=None, num_classes=num_classes, lora_rank=current_rank)
    if args.model == "SAM_MKGA":
        from mkga.models import SAM_MKGA
        return SAM_MKGA(checkpoint_path=None, num_classes=num_classes, lora_rank=current_rank)
    if args.model == "SAM_ResMKGA":
        from mkga.models import SAM_ResMKGA
        return SAM_ResMKGA(checkpoint_path=None, num_classes=num_classes, lora_rank=current_rank)
    raise ValueError("Invalid model choice.")


def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            checkpoint = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
    expected_keys = model.state_dict().keys()
    new_state_dict = {}
    for k, v in checkpoint.items():
        k = k.replace("module.", "")
        if "branch1" in k and k.replace("branch1", "branch3x3") in expected_keys:
            k = k.replace("branch1", "branch3x3")
        elif "branch2" in k and k.replace("branch2", "branch5x5") in expected_keys:
            k = k.replace("branch2", "branch5x5")
        elif "branch3x3" in k and k.replace("branch3x3", "branch1") in expected_keys:
            k = k.replace("branch3x3", "branch1")
        elif "branch5x5" in k and k.replace("branch5x5", "branch2") in expected_keys:
            k = k.replace("branch5x5", "branch2")
        new_state_dict[k] = v
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("Weights loaded successfully (strict).")
    except RuntimeError as e:
        print(f"Strict loading warning: {e}")
        print("Trying non-strict loading...")
        model.load_state_dict(new_state_dict, strict=False)
        print("Weights loaded (non-strict).")


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = resolve_data_root(args.data_root)
    output_dir = args.output_dir or str(RESULTS_DIR)
    os.makedirs(output_dir, exist_ok=True)

    binary_mode = (args.binary_tirads == "True")
    num_classes_tirads = 2 if binary_mode else 6
    use_lora = (args.use_lora == "True")
    current_rank = args.lora_rank if use_lora else 0
    ablation_dict = {
        "use_gate": not (args.ablate_gate == "True"),
        "use_multi_kernel": not (args.ablate_multi == "True"),
        "use_se_bridge": not (args.ablate_se == "True"),
        "kernel_combo": args.kernel_combo,
    }

    ckpt_name = os.path.splitext(os.path.basename(args.path))[0]
    print(f"--- TESTING {ckpt_name} ---")
    print(f"Data root: {data_root}")
    print(f"Target Dataset: {args.test_on}")
    print(f"Configuration: LoRA={use_lora} (Rank {current_rank})")

    test_ds = ThyroidDataset(
        data_root,
        mode='test',
        data_source=args.test_on,
        fraction=args.fraction,
        use_external_masks=(args.use_masks == "True"),
        binary_tirads=binary_mode,
    )
    loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    print(f"   |-- Test Samples: {len(test_ds)}")

    model = build_model(args, num_classes_tirads, ablation_dict, current_rank).to(device)
    if not os.path.exists(args.path):
        print(f"ERROR: Checkpoint not found: {args.path}")
        return
    print(f"Loading checkpoint: {args.path}")
    load_checkpoint(model, args.path, device)
    model.eval()

    seg_metrics = []
    y_true_tirads, y_pred_tirads, y_prob_tirads = [], [], []
    y_true_pos, y_pred_pos, y_prob_pos = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            imgs = batch['image'].to(device)
            masks = batch['mask'].to(device)
            mask_valid = batch['mask_valid'].to(device)
            out_tuple = model(imgs)
            if len(out_tuple) == 3:
                out, pred_pos_logits, pred_tirads_logits = out_tuple
            else:
                out = out_tuple[0]
                pred_pos_logits, pred_tirads_logits = None, None
            if isinstance(out, list):
                out = out[0]
            batch_size = imgs.size(0)
            for i in range(batch_size):
                tirads_i = batch['tirads'][i].item()
                pos_i = batch['position'][i].item()
                valid_i = mask_valid[i].item()
                if valid_i > 0:
                    seg_metrics.append(
                        calculate_segmentation_metrics(torch.sigmoid(out[i:i + 1]), masks[i:i + 1])
                    )
                if tirads_i != -1 and pred_tirads_logits is not None:
                    probs = torch.softmax(pred_tirads_logits[i:i + 1], dim=1).cpu().numpy()[0]
                    y_true_tirads.append(tirads_i)
                    y_pred_tirads.append(np.argmax(probs))
                    y_prob_tirads.append(probs)
                if pos_i != -1 and pred_pos_logits is not None:
                    pos_probs = torch.softmax(pred_pos_logits[i:i + 1], dim=1).cpu().numpy()[0]
                    y_true_pos.append(pos_i)
                    y_pred_pos.append(np.argmax(pos_probs))
                    y_prob_pos.append(pos_probs)

    prefix = f"{ckpt_name}_{args.test_on}"
    if seg_metrics:
        df_seg = pd.DataFrame(seg_metrics)
        means, stds = df_seg.mean(), df_seg.std()
        print(f"\n   [Segmentation] Dice: {means['Dice']:.4f} ± {stds['Dice']:.4f}")
        pd.DataFrame({'Mean': means, 'Std': stds}).to_csv(os.path.join(output_dir, f"{prefix}_seg.csv"))
    if y_true_tirads and len(np.unique(y_true_tirads)) > 1:
        print("\n   [TIRADS Classification]")
        df_tirads = get_detailed_classification_metrics(
            y_true_tirads, y_pred_tirads, y_prob_tirads,
            classes=sorted(np.unique(y_true_tirads)),
        )
        print(df_tirads[['Accuracy', 'F1-Score', 'ROC AUC']])
        df_tirads.to_csv(os.path.join(output_dir, f"{prefix}_tirads.csv"))
    if y_true_pos and len(np.unique(y_true_pos)) > 1:
        print("\n   [Position Classification]")
        df_pos = get_detailed_classification_metrics(
            y_true_pos, y_pred_pos, y_prob_pos,
            classes=sorted(np.unique(y_true_pos)),
        )
        print(df_pos[['Accuracy', 'F1-Score', 'ROC AUC']])
        df_pos.to_csv(os.path.join(output_dir, f"{prefix}_pos.csv"))


if __name__ == "__main__":
    main()
