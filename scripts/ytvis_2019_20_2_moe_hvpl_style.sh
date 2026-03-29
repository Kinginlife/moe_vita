#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "$0")" &> /dev/null && pwd)
cd "$SCRIPT_DIR/.."

export DETECTRON2_DATASETS=/data1/lsh/VITA-main/datasets
export CUDA_VISIBLE_DEVICES=6,7

NGPUS=2
CFG_FILE="configs/youtubevis_2019/vita_R50_bs8.yaml"
OUTPUT_BASE="output/ytvis_2019_moe_hvpl"
EXP_NAME="VITA_MoE_20_2"

STEP_ARGS="CONT.BASE_CLS 20 CONT.INC_CLS 2 CONT.MODE overlap SEED 42"

BASE_QUERIES=100
ITER_BASE=10000

# MoE configuration
MOE_ARGS="MOE.ENABLED True \
          MOE.ROUTER_DIM 512 \
          MOE.TOP_K 1 \
          MOE.ROUTING_LOSS_WEIGHT 0.1 \
          MOE.FREEZE_OLD_EXPERTS True \
          MOE.INIT_FROM_PREVIOUS True \
          MOE.NUM_MOE_LAYERS 1 \
          MOE.SOFT_ROUTING_TEMP 2.0"

WEIGHT_ARGS="MODEL.MASK_FORMER.NUM_OBJECT_QUERIES ${BASE_QUERIES} \
             MODEL.VITA.NUM_OBJECT_QUERIES ${BASE_QUERIES} \
             MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME VitaMoEMultiScaleMaskedTransformerDecoder"

COMM_ARGS="OUTPUT_DIR ${OUTPUT_BASE} ${STEP_ARGS} ${WEIGHT_ARGS} ${MOE_ARGS}"

INC_ARGS_0="CONT.TASK 0 \
            TEST.EVAL_PERIOD 5000 \
            SOLVER.CHECKPOINT_PERIOD 5000 \
            CONT.WEIGHTS vita_r50_coco.pth \
            SOLVER.MAX_ITER ${ITER_BASE} \
            MOE.NUM_EXPERTS 1"

# Train the base model
echo ">>> Training Task 0 (Base)"
python train_net_vita.py --num-gpus ${NGPUS} \
    --dist-url tcp://127.0.0.1:50164 \
    --config-file ${CFG_FILE} \
    ${COMM_ARGS} ${INC_ARGS_0}


ITER_INC=2500
BASE_QUERIES_INC=100

# Freeze arguments for incremental tasks
FREEZE_ARGS="CONT.FREEZE_BACKBONE True"

WEIGHT_ARGS_INC="MODEL.MASK_FORMER.NUM_OBJECT_QUERIES ${BASE_QUERIES_INC} \
                 MODEL.VITA.NUM_OBJECT_QUERIES ${BASE_QUERIES_INC} \
                 MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME VitaMoEMultiScaleMaskedTransformerDecoder"

COMM_ARGS_INC="OUTPUT_DIR ${OUTPUT_BASE} ${STEP_ARGS} ${WEIGHT_ARGS_INC} ${MOE_ARGS}"

# Train first incremental step
PRETRAINED_PATH="${OUTPUT_BASE}/coco2ytvis2019_train_20-2-ov/step0/model_final.pth"
echo ">>> Training Task 1"
python train_net_vita.py --num-gpus ${NGPUS} \
    --config-file ${CFG_FILE} \
    ${COMM_ARGS_INC} ${FREEZE_ARGS} \
    CONT.TASK 1 \
    MOE.NUM_EXPERTS 2 \
    SOLVER.MAX_ITER ${ITER_INC} \
    CONT.WEIGHTS ${PRETRAINED_PATH}

# Train the rest incremental steps
for t in {2..10}; do
    num_experts=$((t + 1))
    echo ">>> Training Task ${t}"
    python train_net_vita.py --num-gpus ${NGPUS} \
        --config-file ${CFG_FILE} \
        ${COMM_ARGS_INC} ${FREEZE_ARGS} \
        CONT.TASK ${t} \
        MOE.NUM_EXPERTS ${num_experts} \
        SOLVER.MAX_ITER ${ITER_INC}
done
