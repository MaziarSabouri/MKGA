#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -z "${MKGA_DATA_ROOT:-}" ]; then
    echo "ERROR: Set MKGA_DATA_ROOT to the directory containing Dataset/"
    exit 1
fi

resnet_models=("ResNet34" "ResNet34_MKGA" "ResNet34_ResMKGA")
sam_models=("SAM" "SAM_MKGA" "SAM_ResMKGA")

LORA_RANKS=(4 16 32)
EPOCHS=100
FRACTION=1.0
LR=0.0001
PATIENCE=15
BATCH_SIZE_RESNET=16
BATCH_SIZE_SAM=16

COMMON_ARGS=(--data_root "$MKGA_DATA_ROOT" --source ThyroidXL --epochs "$EPOCHS" --lr "$LR"
             --binary_tirads True --fraction "$FRACTION" --patience "$PATIENCE")

echo "========================================================"
echo "Starting experiment suite"
echo "========================================================"

for model in "${resnet_models[@]}"; do
    for freeze in "False" "True"; do
        suffix=""
        [ "$freeze" == "True" ] && suffix="_Frozen"
        ckpt_name="checkpoints/${model}${suffix}_Frac${FRACTION}.pth"
        if [ -f "$ckpt_name" ]; then
            echo "Skipping $model (freeze=$freeze) — checkpoint exists."
            continue
        fi
        python train.py --model "$model" --batch_size "$BATCH_SIZE_RESNET" \
            --freeze_resnet "$freeze" "${COMMON_ARGS[@]}"
    done
done

for model in "${sam_models[@]}"; do
    ckpt_name="checkpoints/${model}_Frac${FRACTION}.pth"
    if [ ! -f "$ckpt_name" ]; then
        python train.py --model "$model" --batch_size "$BATCH_SIZE_SAM" \
            --use_lora False --lora_rank 0 "${COMMON_ARGS[@]}"
    fi
    for rank in "${LORA_RANKS[@]}"; do
        ckpt_name="checkpoints/${model}_LoRa_Rank${rank}_Frac${FRACTION}.pth"
        if [ ! -f "$ckpt_name" ]; then
            python train.py --model "$model" --batch_size "$BATCH_SIZE_SAM" \
                --use_lora True --lora_rank "$rank" "${COMMON_ARGS[@]}"
        fi
    done
done

echo "Done."
