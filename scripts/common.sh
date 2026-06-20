#!/bin/bash
# Shared helpers for MKGA experiment scripts.
# Source from scripts/*.sh after setting ROOT to the MKGA repo directory.

# Training fraction (suffix in checkpoint filenames). Override: TRAIN_FRACTION=1.0 bash scripts/run_train.sh
TRAIN_FRACTION="${TRAIN_FRACTION:-1.0}"
# Fraction of the test split to evaluate (independent of training fraction)
TEST_FRACTION="${TEST_FRACTION:-1.0}"

resolve_mkga_data_root() {
    if [ -n "${MKGA_DATA_ROOT:-}" ]; then
        MKGA_DATA_ROOT="$(cd "$MKGA_DATA_ROOT" && pwd)"
        export MKGA_DATA_ROOT
        return 0
    fi

    # Auto-detect: parent of MKGA repo contains Dataset/
    local candidate
    candidate="$(cd "${ROOT}/.." && pwd)"
    if [ -d "${candidate}/Dataset" ]; then
        MKGA_DATA_ROOT="$candidate"
        export MKGA_DATA_ROOT
        echo "Auto-detected MKGA_DATA_ROOT=${MKGA_DATA_ROOT}"
        return 0
    fi

    echo "ERROR: Data root not set."
    echo "  Pass --data_root to train.py/test.py, set MKGA_DATA_ROOT, or place Dataset/ at:"
    echo "  ${candidate}/Dataset"
    exit 1
}

init_experiment_paths() {
    CHECKPOINT_DIR="${ROOT}/checkpoints"
    RESULTS_DIR="${ROOT}/results"
}
