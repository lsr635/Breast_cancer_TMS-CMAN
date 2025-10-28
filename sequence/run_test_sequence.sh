#!/usr/bin/env bash
# Wrapper script to launch test_sequence.py with the common configuration.
#
# Usage:
#   ./run_test_sequence.sh [DATASET_ROOT] [WEIGHTS_PATH] [OUTPUT_DIR]
#
# Arguments (all optional, position-based):
#   DATASET_ROOT  Path to the dataset root that contains train/val/test folders.
#   WEIGHTS_PATH  Path to the trained checkpoint (.pth) to evaluate.
#   OUTPUT_DIR    Folder to write per-case results and CSV reports.
#
# Defaults below can be edited to match your workspace.

set -euo pipefail

# ---------- User-configurable defaults ----------
ENV_NAME="bctc"
DATASET_ROOT_DEFAULT="/root/autodl-tmp/class_dataset_exp_1"
WEIGHTS_DEFAULT="/root/autodl-tmp/Breast_cancer_TMS-CMAN/sequence/model/3/tms_cman_best_auc.pth"
OUTPUT_DEFAULT="test_results_$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=4
IMG_SIZE=224
MAX_T=9
DEVICE="cuda:0"
SEQ_SOURCE="exp1_vibrant"
COPY_MODE="copy"   # copy|link|none
NUM_WORKERS=4
USE_TTA=1           # 1 to enable horizontal-flip TTA, 0 to disable

# ---------- Resolve positional overrides ----------
DATASET_ROOT="${1:-$DATASET_ROOT_DEFAULT}"
WEIGHTS_PATH="${2:-$WEIGHTS_DEFAULT}"
OUTPUT_DIR="${3:-$OUTPUT_DEFAULT}"

# ---------- Environment setup ----------
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
else
    echo "[WARN] conda command not found; proceeding without activating \"$ENV_NAME\"." >&2
fi

cd "$(dirname "$0")"

# ---------- Build command ----------
CMD=(
    python test_sequence.py
    --path "$DATASET_ROOT"
    --weights "$WEIGHTS_PATH"
    --output "$OUTPUT_DIR"
    --batch-size "$BATCH_SIZE"
    --img-size "$IMG_SIZE"
    --max-T "$MAX_T"
    --device "$DEVICE"
    --seq-source "$SEQ_SOURCE"
    --num-workers "$NUM_WORKERS"
    --copy-mode "$COPY_MODE"
    --overwrite
)

if [[ "$USE_TTA" == "1" ]]; then
    CMD+=(--tta)
fi

printf 'Running: %s\n' "${CMD[*]}"
"${CMD[@]}"


# chmod +x run_test_sequence.sh
# ./run_test_sequence.sh /root/autodl-tmp/dataset_exp_1 model/tms_cman/tms_cman_best_auc.pth test_results_run1