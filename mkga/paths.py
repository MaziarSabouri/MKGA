"""Repository and runtime path helpers."""

import os
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = REPO_ROOT / "weights"
DEFAULT_MEDSAM_WEIGHTS = WEIGHTS_DIR / "medsam_vit_b.pth"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "results"


def resolve_data_root(cli_value: Optional[str] = None) -> str:
    """Return absolute path to the directory that contains ``Dataset/``."""
    if cli_value:
        return os.path.abspath(cli_value)
    env = os.environ.get("MKGA_DATA_ROOT")
    if env:
        return os.path.abspath(env)
    raise ValueError(
        "Data root not set. Pass --data_root or set the MKGA_DATA_ROOT environment variable."
    )


def resolve_medsam_weights(cli_value: Optional[str] = None) -> str:
    """Return path to MedSAM ViT-B pretrained weights."""
    if cli_value:
        return os.path.abspath(cli_value)
    return str(DEFAULT_MEDSAM_WEIGHTS)
