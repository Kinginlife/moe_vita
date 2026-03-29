#!/bin/bash
# YouTube-VIS 2019: 20 base + 2 incremental classes with MoE
# Task 0: 80000 iterations (base task)
# Task 1-10: 10000 iterations each (incremental tasks)

SCRIPT_DIR=$(cd "$(dirname "$0")" &> /dev/null && pwd)
cd "$SCRIPT_DIR/.."

export DETECTRON2_DATASETS=datasets
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NGPUS=8
CFG_FILE="configs/youtubevis_2019/vita_R50_bs8.yaml"
OUTPUT_BASE="output/ytvis_2019_moe"

# Task 0: Base task (80000 iterations)
echo ">>> Training Task 0 (Base) - 80000 iterations"
python train_net_vita.py --num-gpus ${NGPUS} \
    --config-file ${CFG_FILE} \
    OUTPUT_DIR ${OUTPUT_BASE}/task0 \
    MOE.ENABLED True \
    MOE.NUM_EXPERTS 1 \
    CONT.TASK 0 \
    CONT.BASE_CLS 20 \
    CONT.INC_CLS 2 \
    MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME VitaMoEMultiScaleMaskedTransformerDecoder \
    SOLVER.MAX_ITER 80000 \
    SOLVER.STEPS 55000 \
    TEST.EVAL_PERIOD 5000 \
    SOLVER.CHECKPOINT_PERIOD 5000

# Incremental tasks: Task 1-10 (10000 iterations each)
for t in {1..10}; do
    prev_t=$((t - 1))

    echo ">>> Training Task ${t} - 10000 iterations"
    python train_net_vita.py --num-gpus ${NGPUS} \
        --config-file ${CFG_FILE} \
        OUTPUT_DIR ${OUTPUT_BASE}/task${t} \
        MOE.ENABLED True \
        MOE.NUM_EXPERTS $((t + 1)) \
        CONT.TASK ${t} \
        CONT.BASE_CLS 20 \
        CONT.INC_CLS 2 \
        CONT.WEIGHTS ${OUTPUT_BASE}/task${prev_t}/model_final.pth \
        MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME VitaMoEMultiScaleMaskedTransformerDecoder \
        SOLVER.MAX_ITER 10000 \
        TEST.EVAL_PERIOD 2000 \
        SOLVER.CHECKPOINT_PERIOD 2000
done




