#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -z "${MKGA_DATA_ROOT:-}" ]; then
    echo "ERROR: Set MKGA_DATA_ROOT to the directory containing Dataset/"
    exit 1
fi

COMMON=(--data_root "$MKGA_DATA_ROOT" --source ThyroidXL)

echo "=============================================="
echo " MKGA / ResMKGA Ablation Study"
echo "=============================================="

python train.py --model ResNet34_MKGA "${COMMON[@]}"
python test.py --data_root "$MKGA_DATA_ROOT" --model ResNet34_MKGA --test_on ThyroidXL --path checkpoints/ResNet34_MKGA_Frac1.0.pth
python test.py --data_root "$MKGA_DATA_ROOT" --model ResNet34_MKGA --test_on DDTI --path checkpoints/ResNet34_MKGA_Frac1.0.pth

python train.py --model ResNet34_MKGA --ablate_gate True "${COMMON[@]}"
python test.py --data_root "$MKGA_DATA_ROOT" --model ResNet34_MKGA --test_on ThyroidXL --path checkpoints/ResNet34_MKGA_NoGate_Frac1.0.pth --ablate_gate True
python test.py --data_root "$MKGA_DATA_ROOT" --model ResNet34_MKGA --test_on DDTI --path checkpoints/ResNet34_MKGA_NoGate_Frac1.0.pth --ablate_gate True

python train.py --model ResNet34_MKGA --ablate_multi True "${COMMON[@]}"
python test.py --data_root "$MKGA_DATA_ROOT" --model ResNet34_MKGA --test_on ThyroidXL --path checkpoints/ResNet34_MKGA_NoMulti_Frac1.0.pth --ablate_multi True
python test.py --data_root "$MKGA_DATA_ROOT" --model ResNet34_MKGA --test_on DDTI --path checkpoints/ResNet34_MKGA_NoMulti_Frac1.0.pth --ablate_multi True

python train.py --model ResNet34_ResMKGA --ablate_se True "${COMMON[@]}"
python test.py --data_root "$MKGA_DATA_ROOT" --model ResNet34_ResMKGA --test_on ThyroidXL --path checkpoints/ResNet34_ResMKGA_NoSE_Frac1.0.pth --ablate_se True
python test.py --data_root "$MKGA_DATA_ROOT" --model ResNet34_ResMKGA --test_on DDTI --path checkpoints/ResNet34_ResMKGA_NoSE_Frac1.0.pth --ablate_se True

python train.py --model ResNet34_MKGA --kernel_combo "1_3" "${COMMON[@]}"
python test.py --data_root "$MKGA_DATA_ROOT" --model ResNet34_MKGA --test_on ThyroidXL --path checkpoints/ResNet34_MKGA_K1_3_Frac1.0.pth --kernel_combo "1_3"
python test.py --data_root "$MKGA_DATA_ROOT" --model ResNet34_MKGA --test_on DDTI --path checkpoints/ResNet34_MKGA_K1_3_Frac1.0.pth --kernel_combo "1_3"

python train.py --model ResNet34_MKGA --kernel_combo "3_7" "${COMMON[@]}"
python test.py --data_root "$MKGA_DATA_ROOT" --model ResNet34_MKGA --test_on ThyroidXL --path checkpoints/ResNet34_MKGA_K3_7_Frac1.0.pth --kernel_combo "3_7"
python test.py --data_root "$MKGA_DATA_ROOT" --model ResNet34_MKGA --test_on DDTI --path checkpoints/ResNet34_MKGA_K3_7_Frac1.0.pth --kernel_combo "3_7"

echo "All ablation studies completed."
