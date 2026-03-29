#!/bin/bash
# YouTube-VIS 2019: 20 base classes + 2 incremental classes per task
# Total: 5 tasks (20 -> 22 -> 24 -> 26 -> 28 classes)

SCRIPT_DIR=$(cd "$(dirname "$0")" &> /dev/null && pwd)
cd "$SCRIPT_DIR/.."

# Environment setup
export DETECTRON2_DATASETS=datasets
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Basic configuration
NGPUS=8
CFG_FILE="configs/youtubevis_2019/vita_R50_bs8.yaml"
OUTPUT_BASE="output/ytvis_2019_continual"
EXP_NAME="ytvis_2019_20_2"

# Continual learning settings
STEP_ARGS="CONT.BASE_CLS 20 CONT.INC_CLS 2 CONT.MODE overlap"

# Base task training parameters
BASE_QUERIES=100
ITER_BASE=80000
EVAL_PERIOD=5000
CHECKPOINT_PERIOD=5000

WEIGHT_ARGS="MODEL.MASK_FORMER.NUM_OBJECT_QUERIES ${BASE_QUERIES} \
             MODEL.VITA.NUM_OBJECT_QUERIES ${BASE_QUERIES}"

COMM_ARGS="OUTPUT_DIR ${OUTPUT_BASE} ${STEP_ARGS} ${WEIGHT_ARGS}"

# Task 0: Base task training
echo "=========================================="
echo "Training Task 0: Base Classes (0-19)"
echo "=========================================="

INC_ARGS_0="CONT.TASK 0 \
            TEST.EVAL_PERIOD ${EVAL_PERIOD} \
            SOLVER.CHECKPOINT_PERIOD ${CHECKPOINT_PERIOD} \
            CONT.WEIGHTS vita_r50_coco.pth \
            SOLVER.STEPS (55000,) \
            SOLVER.MAX_ITER ${ITER_BASE}"

python train_net_vita_continual.py --num-gpus ${NGPUS} \
    --config-file ${CFG_FILE} \
    ${COMM_ARGS} ${INC_ARGS_0} \
    NAME ${EXP_NAME}

# Incremental task training parameters
ITER_INC=10000
EVAL_PERIOD_INC=2000
CHECKPOINT_PERIOD_INC=2000

# Task 1-10: Incremental tasks
for t in {1..10}; do
    echo "=========================================="
    echo "Training Task ${t}: Classes $((20 + (t-1)*2))-$((20 + t*2 - 1))"
    echo "=========================================="

    python train_net_vita_continual.py --num-gpus ${NGPUS} \
        --config-file ${CFG_FILE} \
        ${COMM_ARGS} \
        CONT.TASK ${t} \
        TEST.EVAL_PERIOD ${EVAL_PERIOD_INC} \
        SOLVER.CHECKPOINT_PERIOD ${CHECKPOINT_PERIOD_INC} \
        SOLVER.MAX_ITER ${ITER_INC} \
        NAME ${EXP_NAME}
done

echo "=========================================="
echo "Training completed for all tasks!"
echo "=========================================="
