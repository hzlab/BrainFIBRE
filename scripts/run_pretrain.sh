#!/usr/bin/env bash
CODE_DIR="/path/to/code/directory"
DATA_ROOT="/path/to/dataset/folder"
NUM_GPUS=8

OUT_DIR="${CODE_DIR}/output"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUT_DIR}/${RUN_TS}"
mkdir -p "${RUN_DIR}"

torchrun --nproc_per_node=${NUM_GPUS} --master_port=29500 train.py \
    --data_root "${DATA_ROOT}" \
    --run_root "${OUT_DIR}" \
    --run_dir "${RUN_DIR}" 