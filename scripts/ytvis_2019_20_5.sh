#!/bin/bash
# YouTube-VIS 2019: 20 base classes + 5 incremental classes per task
# Total: 5 tasks (20 -> 25 -> 30 -> 35 -> 40 classes)

SCRIPT_DIR=$(cd "$(dirname "$0")" &> /dev/null && pwd)
cd "$SCRIPT_DIR/.."

export DETECTRON2_DATASETS=datasets
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NGPUS=8
CFG_FILE="configs/youtubevis_2019/vita_R50_bs8.yaml"
OUTPUT_BASE="output/ytvis_2019_continual"
EXP_NAME="ytvis_2019_20_5"

STEP_ARGS="CONT.BASE_CLS 20 CONT.INC_CLS 5 CONT.MODE overlap"
BASE_QUERIES=100
ITER_BASE=80000

WEIGHT_ARGS="MODEL.MASK_FORMER.NUM_OBJECT_QUERIES ${BASE_QUERIES} \
             MODEL.VITA.NUM_OBJECT_QUERIES ${BASE_QUERIES}"

COMM_ARGS="OUTPUT_DIR ${OUTPUT_BASE} ${STEP_ARGS} ${WEIGHT_ARGS}"

# Task 0
echo ">>> Training Task 0: Base Classes (0-19)"
python train_net_vita_continual.py --num-gpus ${NGPUS} \
    --config-file ${CFG_FILE} \
    ${COMM_ARGS} \
    CONT.TASK 0 \
    TEST.EVAL_PERIOD 5000 \
    SOLVER.CHECKPOINT_PERIOD 5000 \
    CONT.WEIGHTS vita_r50_coco.pth \
    SOLVER.STEPS (55000,) \
    SOLVER.MAX_ITER ${ITER_BASE} \
    NAME ${EXP_NAME}

# Incremental tasks
ITER_INC=30000
for t in {1..4}; do
    echo ">>> Training Task ${t}: Classes $((20 + (t-1)*5))-$((20 + t*5 - 1))"
    python train_net_vita_continual.py --num-gpus ${NGPUS} \
        --config-file ${CFG_FILE} \
        ${COMM_ARGS} \
        CONT.TASK ${t} \
        TEST.EVAL_PERIOD 5000 \
        SOLVER.CHECKPOINT_PERIOD 5000 \
        SOLVER.MAX_ITER ${ITER_INC} \
        NAME ${EXP_NAME}
done

echo ">>> Training completed!"
