#!/bin/bash
# OVIS: 15 base classes + 5 incremental classes per task

SCRIPT_DIR=$(cd "$(dirname "$0")" &> /dev/null && pwd)
cd "$SCRIPT_DIR/.."

export DETECTRON2_DATASETS=datasets
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NGPUS=8
CFG_FILE="configs/ovis/vita_R50_bs8.yaml"
OUTPUT_BASE="output/ovis_continual"
EXP_NAME="ovis_15_5"

STEP_ARGS="CONT.BASE_CLS 15 CONT.INC_CLS 5 CONT.MODE overlap"
BASE_QUERIES=100

WEIGHT_ARGS="MODEL.MASK_FORMER.NUM_OBJECT_QUERIES ${BASE_QUERIES} \
             MODEL.VITA.NUM_OBJECT_QUERIES ${BASE_QUERIES}"

COMM_ARGS="OUTPUT_DIR ${OUTPUT_BASE} ${STEP_ARGS} ${WEIGHT_ARGS}"

# Task 0
echo ">>> Training Task 0"
python train_net_vita_continual.py --num-gpus ${NGPUS} \
    --config-file ${CFG_FILE} \
    ${COMM_ARGS} \
    CONT.TASK 0 \
    TEST.EVAL_PERIOD 5000 \
    SOLVER.CHECKPOINT_PERIOD 5000 \
    CONT.WEIGHTS vita_r50_coco.pth \
    SOLVER.MAX_ITER 80000 \
    NAME ${EXP_NAME}

# Incremental tasks
for t in {1..2}; do
    echo ">>> Training Task ${t}"
    python train_net_vita_continual.py --num-gpus ${NGPUS} \
        --config-file ${CFG_FILE} \
        ${COMM_ARGS} \
        CONT.TASK ${t} \
        TEST.EVAL_PERIOD 5000 \
        SOLVER.CHECKPOINT_PERIOD 5000 \
        SOLVER.MAX_ITER 30000 \
        NAME ${EXP_NAME}
done
