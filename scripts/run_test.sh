#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
resolve_mkga_data_root
init_experiment_paths

resnet_models=("ResNet34" "ResNet34_MKGA" "ResNet34_ResMKGA")
sam_models=("SAM" "SAM_MKGA" "SAM_ResMKGA")
datasets=("ThyroidXL" "DDTI")
LORA_RANKS=(4 16 32)

echo "Evaluating checkpoints in $CHECKPOINT_DIR (Frac${TRAIN_FRACTION})"
echo "Test data fraction: $TEST_FRACTION"

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
            --output_dir "$RESULTS_DIR" \
            --fraction "$TEST_FRACTION" \
            --binary_tirads True \
            --use_masks False \
            --lora_rank "$rank" \
            --use_lora "$(if [ "$rank" -gt 0 ]; then echo True; else echo False; fi)"
    done
}

for model in "${resnet_models[@]}"; do
    run_test "$model" "${CHECKPOINT_DIR}/${model}_Frac${TRAIN_FRACTION}.pth" 0
    run_test "$model" "${CHECKPOINT_DIR}/${model}_Frozen_Frac${TRAIN_FRACTION}.pth" 0
done

for model in "${sam_models[@]}"; do
    run_test "$model" "${CHECKPOINT_DIR}/${model}_Frac${TRAIN_FRACTION}.pth" 0
    for rank in "${LORA_RANKS[@]}"; do
        run_test "$model" "${CHECKPOINT_DIR}/${model}_LoRa_Rank${rank}_Frac${TRAIN_FRACTION}.pth" "$rank"
    done
done

echo "Done."
