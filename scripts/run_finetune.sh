#!/usr/bin/env bash

# ==============================================================================
# INPUT CONFIGURATION
# ==============================================================================
CODE_DIR="/path/to/code/directory"
DEFAULT_PRETRAINED="${CODE_DIR}/checkpoints/pretrain/ckpt_latest.pt"
DEFAULT_PARAMS_JSON="hcp_finetune.json"

print_usage() {
    echo "[Usage] $0 <dataset> <task> <splits> <tuning_mode> [pretrained_path] [params_json] [resume_path]"
    echo "dataset:         Dataset name, e.g. UKB, SINGER, HCP"
    echo "task:            Downstream task column, e.g. age_when_attended_assessment_centre_f.21003.2.0"
    echo "splits:          Data split index"
    echo "tuning_mode:     full (finetune from pretrained checkpoint) or tfs (train from scratch)"
    echo "pretrained_path: Optional; default: checkpoints/pretrain/ckpt_latest.pt (ignored when tuning_mode=tfs)"
    echo "params_json:     Optional; default: ${DEFAULT_PARAMS_JSON} (also sets task_type and num_outputs)"
    echo "resume_path:     Optional checkpoint path for resuming training"
    echo "Example: bash scripts/run_finetune.sh HCP age 1 full checkpoints/pretrain/ckpt_latest.pt hcp_finetune.json"
}

if [ $# -lt 4 ] || [ $# -gt 7 ]; then
    print_usage
    exit 1
fi

DATASET=$1
TASK=$2
SPLITS=$3
TUNING_MODE=$4
PRETRAINED_PATH=${5:-"${DEFAULT_PRETRAINED}"}
PARAMS_JSON=${6:-"${DEFAULT_PARAMS_JSON}"}
RESUME_PATH=${7:-""}

if [[ "${TUNING_MODE}" != "full" && "${TUNING_MODE}" != "tfs" ]]; then
    echo "Error: tuning_mode must be 'full' or 'tfs', got '${TUNING_MODE}'"
    exit 1
fi

if [ "${TUNING_MODE}" == "tfs" ]; then
    PRETRAINED_PATH=""
else
    if [[ "${PRETRAINED_PATH}" != /* ]]; then
        PRETRAINED_PATH="${CODE_DIR}/${PRETRAINED_PATH}"
    fi
    if [ ! -f "${PRETRAINED_PATH}" ]; then
        echo "Error: pretrained checkpoint not found: ${PRETRAINED_PATH}"
        exit 1
    fi
fi

# ==============================================================================
# RUN CONFIGURATION & PATHS
# ==============================================================================
DATA_ROOT="/path/to/dataset/folder"
PARAMS_PATH="${CODE_DIR}/config/${PARAMS_JSON}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${CODE_DIR}/output/finetune_${TUNING_MODE}/${DATASET}/${TASK}/split${SPLITS}_${RUN_TS}/"

# ==============================================================================
# LAUNCH TRAINING
# ==============================================================================
cd "${CODE_DIR}" || exit 1

torchrun --nproc_per_node=1 --master_port=29500 finetune.py \
    --data_root "${DATA_ROOT}" \
    --run_dir "${RUN_DIR}" \
    --resume_path "${RESUME_PATH}" \
    --pretrained_path "${PRETRAINED_PATH}" --task "${TASK}" \
    --splits "${SPLITS}" --params_json "${PARAMS_PATH}" \
    --dataset_name "${DATASET}"

echo "Finished executing task: ${TASK}"
