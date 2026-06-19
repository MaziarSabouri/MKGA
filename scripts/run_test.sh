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
datasets=("ThyroidXL" "DDTI")
FRACTION=1.0
LORA_RANKS=(4 16 32)

run_test() {
    local model_name=$1
    local checkpoint_path=$2
    local rank=$3

    if [ ! -f "$checkpoint_path" ]; then
        echo "Skipping $model_name — checkpoint not found: $checkpoint_path"
        return
    fi

    for dataset in "${datasets[@]}"; do
        python test.py \
            --data_root "$MKGA_DATA_ROOT" \
            --model "$model_name" \
            --test_on "$dataset" \
            --path "$checkpoint_path" \
            --fraction "$FRACTION" \
            --binary_tirads True \
            --use_masks False \
            --lora_rank "$rank" \
            --use_lora "$(if [ "$rank" -gt 0 ]; then echo True; else echo False; fi)"
    done
}

for model in "${resnet_models[@]}"; do
    run_test "$model" "checkpoints/${model}_Frac${FRACTION}.pth" 0
    run_test "$model" "checkpoints/${model}_Frozen_Frac${FRACTION}.pth" 0
done

for model in "${sam_models[@]}"; do
    run_test "$model" "checkpoints/${model}_Frac${FRACTION}.pth" 0
    for rank in "${LORA_RANKS[@]}"; do
        run_test "$model" "checkpoints/${model}_LoRa_Rank${rank}_Frac${FRACTION}.pth" "$rank"
    done
done

echo "Done."
