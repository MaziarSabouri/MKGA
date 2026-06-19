#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
resolve_mkga_data_root
init_experiment_paths

ABLATION_FRACTION="${ABLATION_FRACTION:-1.0}"
ABLATION_EPOCHS="${ABLATION_EPOCHS:-100}"
ABLATION_PATIENCE="${ABLATION_PATIENCE:-15}"
ABLATION_LR="${ABLATION_LR:-0.0001}"
BATCH_SIZE=16

COMMON=(--data_root "$MKGA_DATA_ROOT" --source ThyroidXL --fraction "$ABLATION_FRACTION"
        --epochs "$ABLATION_EPOCHS" --patience "$ABLATION_PATIENCE" --lr "$ABLATION_LR"
        --batch_size "$BATCH_SIZE" --binary_tirads True)

echo "=============================================="
echo " MKGA / ResMKGA Ablation Study"
echo " Fraction: $ABLATION_FRACTION | Epochs: $ABLATION_EPOCHS | Patience: $ABLATION_PATIENCE"
echo "=============================================="

python train.py --model ResNet34_MKGA "${COMMON[@]}"
python test.py --data_root "$MKGA_DATA_ROOT" --model ResNet34_MKGA --test_on ThyroidXL --path "${CHECKPOINT_DIR}/ResNet34_MKGA_Frac${ABLATION_FRACTION}.pth" --output_dir "$RESULTS_DIR"
python test.py --data_root "$MKGA_DATA_ROOT" --model ResNet34_MKGA --test_on DDTI --path "${CHECKPOINT_DIR}/ResNet34_MKGA_Frac${ABLATION_FRACTION}.pth" --output_dir "$RESULTS_DIR"

python train.py --model ResNet34_MKGA --ablate_gate True "${COMMON[@]}"
python test.py --data_root "$MKGA_DATA_ROOT" --model ResNet34_MKGA --test_on ThyroidXL --path "${CHECKPOINT_DIR}/ResNet34_MKGA_NoGate_Frac${ABLATION_FRACTION}.pth" --ablate_gate True --output_dir "$RESULTS_DIR"
python test.py --data_root "$MKGA_DATA_ROOT" --model ResNet34_MKGA --test_on DDTI --path "${CHECKPOINT_DIR}/ResNet34_MKGA_NoGate_Frac${ABLATION_FRACTION}.pth" --ablate_gate True --output_dir "$RESULTS_DIR"

python train.py --model ResNet34_MKGA --ablate_multi True "${COMMON[@]}"
python test.py --data_root "$MKGA_DATA_ROOT" --model ResNet34_MKGA --test_on ThyroidXL --path "${CHECKPOINT_DIR}/ResNet34_MKGA_NoMulti_Frac${ABLATION_FRACTION}.pth" --ablate_multi True --output_dir "$RESULTS_DIR"
python test.py --data_root "$MKGA_DATA_ROOT" --model ResNet34_MKGA --test_on DDTI --path "${CHECKPOINT_DIR}/ResNet34_MKGA_NoMulti_Frac${ABLATION_FRACTION}.pth" --ablate_multi True --output_dir "$RESULTS_DIR"

python train.py --model ResNet34_ResMKGA --ablate_se True "${COMMON[@]}"
python test.py --data_root "$MKGA_DATA_ROOT" --model ResNet34_ResMKGA --test_on ThyroidXL --path "${CHECKPOINT_DIR}/ResNet34_ResMKGA_NoSE_Frac${ABLATION_FRACTION}.pth" --ablate_se True --output_dir "$RESULTS_DIR"
python test.py --data_root "$MKGA_DATA_ROOT" --model ResNet34_ResMKGA --test_on DDTI --path "${CHECKPOINT_DIR}/ResNet34_ResMKGA_NoSE_Frac${ABLATION_FRACTION}.pth" --ablate_se True --output_dir "$RESULTS_DIR"

python train.py --model ResNet34_MKGA --kernel_combo "1_3" "${COMMON[@]}"
python test.py --data_root "$MKGA_DATA_ROOT" --model ResNet34_MKGA --test_on ThyroidXL --path "${CHECKPOINT_DIR}/ResNet34_MKGA_K1_3_Frac${ABLATION_FRACTION}.pth" --kernel_combo "1_3" --output_dir "$RESULTS_DIR"
python test.py --data_root "$MKGA_DATA_ROOT" --model ResNet34_MKGA --test_on DDTI --path "${CHECKPOINT_DIR}/ResNet34_MKGA_K1_3_Frac${ABLATION_FRACTION}.pth" --kernel_combo "1_3" --output_dir "$RESULTS_DIR"

python train.py --model ResNet34_MKGA --kernel_combo "3_7" "${COMMON[@]}"
python test.py --data_root "$MKGA_DATA_ROOT" --model ResNet34_MKGA --test_on ThyroidXL --path "${CHECKPOINT_DIR}/ResNet34_MKGA_K3_7_Frac${ABLATION_FRACTION}.pth" --kernel_combo "3_7" --output_dir "$RESULTS_DIR"
python test.py --data_root "$MKGA_DATA_ROOT" --model ResNet34_MKGA --test_on DDTI --path "${CHECKPOINT_DIR}/ResNet34_MKGA_K3_7_Frac${ABLATION_FRACTION}.pth" --kernel_combo "3_7" --output_dir "$RESULTS_DIR"

echo "All ablation studies completed."
