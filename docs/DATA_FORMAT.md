# Dataset and JSON Format

This document describes the directory layout and JSON schema expected by `ThyroidDataset` (`mkga/dataset.py`). Both **ThyroidXL** (in-domain) and **DDTI** (external test) follow the same annotation structure; only the on-disk paths and default JSON filenames differ.

## Directory layout

Set `MKGA_DATA_ROOT` (or `--data_root`) to the parent folder that contains `Dataset/`:

```
$MKGA_DATA_ROOT/
└── Dataset/
    ├── ThyroidXL/
    │   ├── stats/
    │   │   └── id2info_eng_clean.json    # annotation file
    │   ├── train/
    │   │   ├── images/                   # training ultrasound images
    │   │   └── masks/                    # binary nodule masks (optional per file)
    │   └── test/
    │       ├── images/
    │       └── masks/
    └── DDTI/
        ├── stats/
        │   └── ddti_dataset.json         # annotation file
        ├── train/
        │   ├── images/
        │   └── masks/
        └── test/
            ├── images/
            └── masks/
```

### Image and mask files

- **Images** listed in the JSON `images` array must exist under the corresponding `images/` folder (`train` or `test`, depending on `mode`).
- **Masks** are matched per image in `masks/` using either:
  - the **same filename** as the image (e.g. `00000058_2201CE11_0.png`), or
  - the **same stem with `.png`** (e.g. image `2_1.jpg` → mask `2_1.png`).
- Masks are read as grayscale; pixels `> 127` are treated as foreground.
- If no mask file is found, an all-zero mask is used and `mask_valid=False` for that sample.

### Train / validation split

For `mode='train'` with `split='train'` or `split='val'`, the code splits **image files** (not patient IDs) in `train/images/` using a fixed seed (`42`) and `val_ratio` (default `0.2`). Test mode uses all files in `test/images/`.

---

## JSON schema

Both datasets use a **single JSON object** whose keys are **patient / case IDs** (strings). Each value is one patient record.

### Top-level patient record

| Field       | Type            | Required | Used by loader | Description |
|-------------|-----------------|----------|----------------|-------------|
| `age`       | `int` or `null` | No       | Yes*           | Patient age in years. `null` → encoded as `-1.0` after normalization. |
| `gender`    | `int` or `null` | No       | Yes*           | Numeric gender code (`1` / `2` in our releases). `null` → `-1.0`. |
| `conclusion`| `int` or `null` | No       | No             | Stored in JSON for reference; not used during training/eval. |
| `nodule_1`  | `object` or `null` | Yes   | Yes            | Primary nodule metadata (see below). |
| `nodule_2`  | `object` or `null` | No    | No             | Ignored by the current loader. |
| `images`    | `list` of `str` | Yes      | Yes            | Filenames of ultrasound images for this patient in `images/`. |

\* `age` and `gender` are returned by the dataloader but are **not** consumed by the default MKGA training loop. They are included for compatibility and future use.

### `nodule_1` object

Only **`nodule_1`** is read for labels. Additional nodules (`nodule_2`, …) are not used.

| Field            | Type              | Required | Used by loader | Description |
|------------------|-------------------|----------|----------------|-------------|
| `Position`       | `str` or `null`   | No       | Yes            | Anatomical location (see position labels below). |
| `TIRADS`         | `int`, `str`, or `null` | No  | Yes            | TIRADS score (typically 1–5). Parsed with `int()`. `null` or missing → `0`. |
| `Width`          | `float` or `null` | No       | No             | Optional size metadata. |
| `Height`         | `float` or `null` | No       | No             | Optional size metadata. |
| `Depth`          | `float` or `null` | No       | No             | Optional size metadata. |
| `FNAC`           | `str` or `null`   | No       | No             | Cytology result. |
| `Conclusion`     | `str` or `null`   | No       | No             | Clinical conclusion text. |
| `Histopathology` | `str` or `null`   | No       | No             | Histopathology label. |
| `Composition`    | `str` or `null`   | No       | No             | DDTI-specific TI-RADS feature (optional). |
| `Echogenicity`   | `str` or `null`   | No       | No             | DDTI-specific TI-RADS feature (optional). |
| `Margins`        | `str` or `null`   | No       | No             | DDTI-specific TI-RADS feature (optional). |
| `Calcifications` | `str` or `null`   | No       | No             | DDTI-specific TI-RADS feature (optional). |

Extra keys are allowed and ignored.

---

## Label encoding

### Position (`nodule_1.Position`)

Case-insensitive string match:

| JSON value (examples) | Class index | Notes |
|-----------------------|-------------|-------|
| `"Right lobe"`        | `0`         | |
| `"Left lobe"`         | `1`         | |
| `"Isthmus"`           | `2`         | |
| `null` or unknown     | `-1`        | Sample excluded from position loss / metrics |

### TIRADS (`nodule_1.TIRADS`)

Controlled by `--binary_tirads` (default `True`):

**Binary mode (`binary_tirads=True`, default in training scripts):**

| Raw TIRADS | Label | Notes |
|------------|-------|-------|
| `0` or missing | `-1` | Ignored in classification loss |
| `1`, `2`, `3`  | `0`  | Low-risk bucket |
| `4`, `5`       | `1`  | High-risk bucket |

**Multi-class mode (`binary_tirads=False`):**

Raw integer TIRADS (1–6) is used directly as the class index. Values that cannot be parsed default to `0`.

---

## Minimal examples

### ThyroidXL-style record

File: `Dataset/ThyroidXL/stats/id2info_eng_clean.json`

```json
{
  "00000058": {
    "age": 60,
    "gender": 2,
    "conclusion": 3,
    "nodule_1": {
      "Position": "Left lobe",
      "Width": 22.0,
      "Height": 14.0,
      "Depth": null,
      "TIRADS": 4,
      "FNAC": "2",
      "Conclusion": "Benign (FNAC) - 3",
      "Histopathology": null
    },
    "nodule_2": null,
    "images": [
      "00000058_2201CE11_0.png",
      "00000058_A73CED93_1.png"
    ]
  }
}
```

### DDTI-style record

File: `Dataset/DDTI/stats/ddti_dataset.json`

```json
{
  "2": {
    "age": 49,
    "gender": 2,
    "conclusion": null,
    "nodule_1": {
      "Position": null,
      "Width": null,
      "Height": null,
      "Depth": null,
      "TIRADS": "2",
      "FNAC": null,
      "Conclusion": null,
      "Histopathology": null,
      "Composition": "solid",
      "Echogenicity": "hyperechogenicity",
      "Margins": "well defined",
      "Calcifications": "non"
    },
    "nodule_2": null,
    "images": [
      "2_1.jpg"
    ]
  }
}
```

`TIRADS` may be an integer or a string; both are accepted.

---

## Dataset-specific behaviour

### ThyroidXL

- JSON path: `Dataset/ThyroidXL/stats/id2info_eng_clean.json`
- Images: `Dataset/ThyroidXL/{train,test}/images/`
- Masks: `Dataset/ThyroidXL/{train,test}/masks/`
- Segmentation masks are used for training and evaluation when present.

### DDTI

- JSON path: `Dataset/DDTI/stats/ddti_dataset.json`
- Images: `Dataset/DDTI/{train,test}/images/`
- Masks: `Dataset/DDTI/{train,test}/masks/`
- During **training** on DDTI, segmentation loss is **disabled by default** (`mask_valid=False`) because masks are reserved for external evaluation. Pass `use_external_masks=True` to the dataset (or `--use_masks True` in `test.py`) to enable mask-based metrics.

---

## Adapting your own data

To use a custom cohort with this codebase:

1. Create `Dataset/<YourDataset>/stats/your_annotations.json` following the schema above.
2. Place images under `train/images/` and `test/images/`.
3. Optionally add masks under `train/masks/` and `test/masks/`.
4. Either extend `ThyroidDataset._load_source()` in `mkga/dataset.py` to recognize your dataset name, or mirror the ThyroidXL folder layout and JSON filename and point `--source ThyroidXL` at your prepared tree.

At minimum, each patient entry needs a non-empty `images` list and a `nodule_1` object with `Position` and `TIRADS` if you want position and TIRADS metrics.

---

## Quick validation checklist

- [ ] JSON is a single object (not a list of records).
- [ ] Every string in `images` exists in the correct `images/` folder.
- [ ] `Position` uses one of: `Right lobe`, `Left lobe`, `Isthmus` (or `null` if unknown).
- [ ] `TIRADS` is an integer or numeric string when available.
- [ ] Mask files use the same name as the image, or the same stem with `.png`.
- [ ] `MKGA_DATA_ROOT` points to the parent of `Dataset/`, not to `Dataset/` itself.
