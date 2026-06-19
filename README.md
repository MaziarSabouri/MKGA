# MKGA: Multi-Kernel Gated Attention for Thyroid Multi-Task Learning

Official implementation for simultaneous thyroid nodule **segmentation**, **TIRADS classification**, and **anatomical position** prediction.

## Project structure

```
MKGA/
├── mkga/                  # Core Python package
│   ├── dataset.py         # ThyroidDataset and augmentations
│   ├── paths.py           # Path helpers
│   ├── models/            # ResNet34, MedSAM, and MKGA variants
│   └── optim/             # PCGrad optimizer wrapper
├── train.py               # Training entry point
├── test.py                # Evaluation entry point
├── scripts/               # Batch experiment runners
├── weights/               # MedSAM pretrained weights (not committed)
├── checkpoints/           # Saved models (gitignored)
└── results/               # Evaluation CSVs (gitignored)
```

## Setup

```bash
cd MKGA
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Download [MedSAM ViT-B](https://github.com/bowang-lab/MedSAM) weights to `weights/medsam_vit_b.pth`.

## Data layout

Set `MKGA_DATA_ROOT` to the directory that contains a `Dataset/` folder:

```
$MKGA_DATA_ROOT/
└── Dataset/
    ├── ThyroidXL/
    │   ├── stats/id2info_eng_clean.json
    │   ├── train/images/, train/masks/
    │   └── test/images/, test/masks/
    └── DDTI/
        ├── stats/ddti_dataset.json
        ├── train/images/, train/masks/
        └── test/images/, test/masks/
```

Alternatively, pass `--data_root /path/to/parent` on every command.

## Training

```bash
export MKGA_DATA_ROOT=/path/to/MultiTaskNet

python train.py \
  --model ResNet34_MKGA \
  --source ThyroidXL \
  --epochs 100 \
  --batch_size 16
```

**Models:** `ResNet34`, `ResNet34_MKGA`, `ResNet34_ResMKGA`, `SAM`, `SAM_MKGA`, `SAM_ResMKGA`

**Key flags:** `--use_lora`, `--lora_rank`, `--freeze_resnet`, `--use_pcgrad`, `--use_uncertainty`, `--use_mixstyle`, ablation toggles (`--ablate_gate`, `--ablate_multi`, `--ablate_se`, `--kernel_combo`)

Checkpoints are saved to `checkpoints/`.

## Evaluation

```bash
python test.py \
  --model ResNet34_MKGA \
  --test_on DDTI \
  --path checkpoints/ResNet34_MKGA_Frac1.0.pth
```

Results are written to `results/` as CSV files (`*_seg.csv`, `*_tirads.csv`, `*_pos.csv`).

## Batch scripts

```bash
export MKGA_DATA_ROOT=/path/to/MultiTaskNet
bash scripts/run_train.sh
bash scripts/run_test.sh
bash scripts/run_ablations.sh
```

## Citation

If you use this code, please cite our paper (details to be added upon publication).
